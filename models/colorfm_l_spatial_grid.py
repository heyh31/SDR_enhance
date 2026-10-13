from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.colorfm_l import ColorFM_L as BaseColorFM_L


def _parse_grid_size(value):
    if isinstance(value, (int, float)):
        size = int(value)
        result = (size, size, size)
    else:
        values = list(value)
        if len(values) != 3:
            raise ValueError(
                "model.spatial_grid_size must be an integer or "
                "[guide_bins, height, width]."
            )
        result = tuple(int(item) for item in values)
    if any(item <= 1 for item in result):
        raise ValueError(
            "All model.spatial_grid_size dimensions must be greater than one, "
            f"got {result}."
        )
    return result


class SpatialAffineGridHead(nn.Module):
    """Fuse ViT maps and predict a 3D grid of 3x4 affine residuals."""

    def __init__(
        self,
        feature_count,
        embed_dim,
        hidden_dim,
        grid_size,
        max_coefficient,
    ):
        super().__init__()
        self.grid_depth, self.grid_height, self.grid_width = grid_size
        self.max_coefficient = float(max_coefficient)
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(embed_dim, hidden_dim, 1),
                nn.SiLU(),
            )
            for _ in range(int(feature_count))
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * int(feature_count), hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
        )
        # Each bilateral-grid cell stores a 3x4 affine transform: a 3x3 RGB
        # matrix plus a three-channel bias.
        self.to_grid = nn.Conv2d(hidden_dim, 12 * self.grid_depth, 1)

        # Loading an original ColorFM-L checkpoint leaves this new branch at
        # zero, so the initial model exactly follows the global branch.
        nn.init.zeros_(self.to_grid.weight)
        nn.init.zeros_(self.to_grid.bias)

    def forward(self, features):
        if len(features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} encoder maps, got {len(features)}."
            )
        target_size = features[-1].shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, features):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(feature)

        fused = self.fuse(torch.cat(projected, dim=1))
        fused = F.interpolate(
            fused,
            size=(self.grid_height, self.grid_width),
            mode="bilinear",
            align_corners=False,
        )
        grid = self.to_grid(fused)
        grid = grid.reshape(
            grid.shape[0],
            12,
            self.grid_depth,
            self.grid_height,
            self.grid_width,
        )
        return torch.tanh(grid) * self.max_coefficient


class LearnedGuideMap(nn.Module):
    """Paper-style pointwise, single-channel full-resolution guide.

    A learned 3x3 color transform is followed by three learned piecewise-linear
    transfer functions. Their outputs are summed into g(x, y) in [0, 1].
    """

    def __init__(self, curve_bins=16):
        super().__init__()
        self.curve_bins = int(curve_bins)
        if self.curve_bins < 2:
            raise ValueError("model.guide_curve_bins must be at least two.")
        self.color_matrix = nn.Parameter(torch.eye(3))
        self.color_bias = nn.Parameter(torch.zeros(3))
        thresholds = torch.linspace(0.0, 1.0, self.curve_bins)
        self.thresholds = nn.Parameter(thresholds.repeat(3, 1))
        self.slopes = nn.Parameter(torch.zeros(3, self.curve_bins))
        self.guide_bias = nn.Parameter(torch.zeros(()))

        # rho_c(x)=x/3 initially, so the starting guide is RGB intensity mean
        # rather than a constant/degenerate map. All parameters remain learned.
        with torch.no_grad():
            self.slopes[:, 0].fill_(1.0 / 3.0)

    def _forward_chunk(self, image):
        transformed = torch.einsum(
            "oc,bchw->bohw",
            self.color_matrix,
            image,
        )
        transformed = transformed + self.color_bias.view(1, 3, 1, 1)
        hinge_inputs = transformed.unsqueeze(2) - self.thresholds.view(
            1,
            3,
            self.curve_bins,
            1,
            1,
        )
        channel_curves = (
            F.relu(hinge_inputs) * self.slopes.view(
                1,
                3,
                self.curve_bins,
                1,
                1,
            )
        ).sum(dim=2)
        guide = channel_curves.sum(dim=1) + self.guide_bias
        return guide.clamp(0.0, 1.0)

    def forward(self, image, max_points=1024 * 1024):
        batch, _, height, width = image.shape
        rows_per_chunk = max(
            min(height, int(max_points) // max(batch * width, 1)),
            1,
        )
        chunks = [
            self._forward_chunk(image[:, :, top : top + rows_per_chunk])
            for top in range(0, height, rows_per_chunk)
        ]
        return torch.cat(chunks, dim=1)


class ColorFM_L(BaseColorFM_L):
    """ColorFM-L plus an HDRNet-style learned bilateral-grid residual.

    The original final-layer/global-kernel branch is preserved. Intermediate
    ViT maps predict a compact grid of local 3x4 affine coefficients. A learned
    single-channel full-resolution guidance map supplies the third coordinate
    for trilinear slicing at (x, y, guide). The sliced affine transform predicts
    a residual over the original global ColorFM-L result.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        model_cfg = cfg.model
        layers = [
            int(item)
            for item in model_cfg.get("spatial_encoder_layers", [2, 5, 8, 11])
        ]
        if not layers:
            raise ValueError("model.spatial_encoder_layers cannot be empty.")
        if len(set(layers)) != len(layers):
            raise ValueError(
                f"model.spatial_encoder_layers contains duplicates: {layers}"
            )
        if min(layers) < 0 or max(layers) >= self.encoder.n_blocks:
            raise ValueError(
                "model.spatial_encoder_layers must use zero-based block indices "
                f"between 0 and {self.encoder.n_blocks - 1}, got {layers}."
            )
        self.spatial_encoder_layers = sorted(layers)
        self.encoder_output_layer = self.encoder.n_blocks - 1
        self.requested_encoder_layers = sorted(
            set(self.spatial_encoder_layers + [self.encoder_output_layer])
        )
        self.spatial_grid_size = _parse_grid_size(
            model_cfg.get("spatial_grid_size", [16, 24, 24])
        )
        self.spatial_grid_head = SpatialAffineGridHead(
            feature_count=len(self.spatial_encoder_layers),
            embed_dim=self.encoder.embed_dim,
            hidden_dim=int(model_cfg.get("spatial_hidden_dim", 256)),
            grid_size=self.spatial_grid_size,
            max_coefficient=float(
                model_cfg.get(
                    "spatial_max_affine_coefficient",
                    model_cfg.get("spatial_max_residual", 0.25),
                )
            ),
        )
        self.guide_map = LearnedGuideMap(
            curve_bins=int(model_cfg.get("guide_curve_bins", 16))
        )
        self.max_query_points = int(
            model_cfg.get("spatial_max_query_points", 1024 * 1024)
        )
        if self.max_query_points <= 0:
            raise ValueError("model.spatial_max_query_points must be positive.")

    def predict_single_parameters(self, context):
        context = F.interpolate(
            context,
            size=(self.context_size, self.context_size),
            mode="bilinear",
            align_corners=False,
        )
        feature_tuple = self.encoder.get_intermediate_layers(
            context,
            n=self.requested_encoder_layers,
            reshape=True,
            norm=True,
        )
        feature_by_layer = dict(zip(self.requested_encoder_layers, feature_tuple))
        spatial_features = [
            feature_by_layer[layer] for layer in self.spatial_encoder_layers
        ]
        final_feature = feature_by_layer[self.encoder_output_layer]
        final_tokens = final_feature.flatten(2).transpose(1, 2).contiguous()

        global_kernels = self.param_gen(final_tokens)
        spatial_grid = self.spatial_grid_head(spatial_features)
        self.params = {
            "kernel1_single": global_kernels[0],
            "kernel2_single": global_kernels[1],
            "kernel3_single": global_kernels[2],
            "kernel4_single": global_kernels[3],
            "spatial_grid": spatial_grid,
        }
        return global_kernels, spatial_grid

    def predict_single_kernels(self, context):
        """Compatibility name used by the existing solver and test UI."""
        return self.predict_single_parameters(context)

    def _apply_spatial_grid(self, source_image, global_image, spatial_grid):
        batch, _, height, width = global_image.shape
        if spatial_grid.shape[0] != batch:
            raise ValueError(
                f"Grid/image batch mismatch: {spatial_grid.shape[0]} vs {batch}."
            )

        guide = self.guide_map(
            source_image.clamp(0.0, 1.0),
            max_points=self.max_query_points,
        )
        self.params["guide_map"] = guide
        # Pixel-center coordinates with align_corners=False follow bilateral
        # slicing's cell-center convention more closely than corner alignment.
        x_coordinates = (
            (torch.arange(
                width,
                device=global_image.device,
                dtype=global_image.dtype,
            ) + 0.5)
            * (2.0 / width)
            - 1.0
        ).view(1, 1, width)
        y_coordinates = (
            (torch.arange(
                height,
                device=global_image.device,
                dtype=global_image.dtype,
            ) + 0.5)
            * (2.0 / height)
            - 1.0
        )

        rows_per_chunk = max(
            min(height, self.max_query_points // max(batch * width, 1)),
            1,
        )
        output_chunks = []
        for top in range(0, height, rows_per_chunk):
            bottom = min(top + rows_per_chunk, height)
            chunk_height = bottom - top
            x = x_coordinates.expand(batch, chunk_height, width)
            y = y_coordinates[top:bottom].view(1, chunk_height, 1).expand(
                batch,
                chunk_height,
                width,
            )
            guide_coordinate = guide[:, top:bottom] * 2.0 - 1.0
            # Five-dimensional grid_sample expects coordinates ordered x/y/z,
            # where z indexes the learned-guide/depth dimension.
            sampling_coordinates = torch.stack(
                (x, y, guide_coordinate),
                dim=-1,
            ).unsqueeze(1)
            affine_residual = F.grid_sample(
                spatial_grid,
                sampling_coordinates,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).squeeze(2)
            affine_residual = affine_residual.reshape(
                batch,
                3,
                4,
                chunk_height,
                width,
            )
            global_chunk = global_image[:, :, top:bottom]
            residual = (
                affine_residual[:, :, :3]
                * global_chunk.unsqueeze(1)
            ).sum(dim=2)
            residual = residual + affine_residual[:, :, 3]
            output_chunks.append(global_chunk + residual)
        return torch.cat(output_chunks, dim=2)

    def forward_single_stages(self, image, parameters=None):
        """Return the global output, signed grid residual, and final output.

        Keeping these stages explicit is useful for inspecting what the spatial
        branch learns. ``grid_residual`` is signed and therefore should be
        visualized around a neutral midpoint instead of being clamped directly
        as an RGB image.
        """
        parameters = (
            self.predict_single_parameters(image)
            if parameters is None
            else parameters
        )
        if not isinstance(parameters, (tuple, list)) or len(parameters) != 2:
            raise ValueError(
                "Spatial ColorFM-L expects (global_kernels, spatial_grid)."
            )
        global_kernels, spatial_grid = parameters
        global_image = self._apply_velocity(image, global_kernels)
        final_image = self._apply_spatial_grid(image, global_image, spatial_grid)
        grid_residual = final_image - global_image
        return global_image, grid_residual, final_image

    def forward_single(self, image, kernels=None):
        _, _, final_image = self.forward_single_stages(
            image,
            parameters=kernels,
        )
        return final_image

    def spatial_grid_total_variation(self):
        grid = self.params.get("spatial_grid")
        if grid is None:
            raise RuntimeError("No spatial grid is available before model forward.")
        terms = [
            (grid[:, :, 1:] - grid[:, :, :-1]).abs().mean(),
            (grid[:, :, :, 1:] - grid[:, :, :, :-1]).abs().mean(),
            (grid[:, :, :, :, 1:] - grid[:, :, :, :, :-1]).abs().mean(),
        ]
        return sum(terms) / len(terms)
