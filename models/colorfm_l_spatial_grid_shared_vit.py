"""One final ViT feature map, separate Global and local affine Grid heads."""
from __future__ import annotations

from functools import partial
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.colorfm_l import DinoVisionTransformer, Swish
from models.layers.attention import Attention
from models.layers.block import Block
from models.colorfm_l_spatial_grid import (
    ColorFM_L as SpatialColorFM,
    LearnedGuideMap,
    _parse_grid_size,
)


class GlobalKernelHead(nn.Module):
    """Mean-pool patch tokens and predict the four existing RGB kernels."""

    kernel_shapes = ((3, 16), (16, 16), (16, 16), (16, 3))

    def __init__(self, embed_dim, hidden_dim=128, dropout=0.05):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, rows * columns)
            for rows, columns in self.kernel_shapes
        ])
        # The applied RGB MLP is residual. Only its last kernel starts at
        # zero; zeroing every kernel would block useful gradient propagation.
        nn.init.zeros_(self.heads[-1].weight)
        nn.init.zeros_(self.heads[-1].bias)

    def forward(self, tokens):
        embedding = self.project(tokens.mean(dim=1))
        return tuple(
            head(embedding).reshape(tokens.shape[0], *shape)
            for head, shape in zip(self.heads, self.kernel_shapes)
        )


class SpatialResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, features):
        return F.silu(features + self.convs(features))


class SharedFeatureGridHead(nn.Module):
    """Keep token positions; each cell emits 12 coefficients per guide bin."""

    def __init__(self, embed_dim, hidden_dim, grid_size, max_coefficient):
        super().__init__()
        self.grid_size = grid_size
        self.max_coefficient = float(max_coefficient)
        self.project = nn.Sequential(nn.Conv2d(embed_dim, hidden_dim, 1), nn.SiLU())
        self.block = SpatialResidualBlock(hidden_dim)
        self.to_grid = nn.Conv2d(hidden_dim, 12 * grid_size[0], 1)
        nn.init.zeros_(self.to_grid.weight)
        nn.init.zeros_(self.to_grid.bias)

    def forward(self, feature_map):
        depth, height, width = self.grid_size
        features = self.block(self.project(feature_map))
        if features.shape[-2:] != (height, width):
            features = F.interpolate(features, size=(height, width),
                                     mode="bilinear", align_corners=False)
        grid = self.to_grid(features).reshape(feature_map.shape[0], 12, depth, height, width)
        return torch.tanh(grid) * self.max_coefficient


class ColorFM_L(SpatialColorFM):
    """Predict Global/Grid together, then apply the Grid residual to Global.

    Both heads receive the same final normalized ViT patch tokens. The
    pointwise learned guide reads the full-resolution original SDR frame.
    Training is per-frame: no temporal parameter averaging is performed.
    """

    def __init__(self, cfg):
        nn.Module.__init__(self)
        model_cfg = cfg.model
        self.context_size = int(cfg.data.get("context_size", 256))
        self.patch_size = 16
        embed_dim = int(model_cfg.get("embed_dim", 192))
        depth = int(model_cfg.get("depth", 6))
        heads = int(model_cfg.get("num_heads", 3))
        mlp_ratio = float(model_cfg.get("mlp_ratio", 4.0))
        global_hidden = int(model_cfg.get("global_hidden_dim", 128))
        grid_hidden = int(model_cfg.get("grid_hidden_dim", 64))
        dropout = float(model_cfg.get("dropout", 0.05))
        drop_path = float(model_cfg.get("drop_path_rate", 0.1))
        coefficient = float(model_cfg.get("spatial_max_affine_coefficient", 0.25))
        self.max_query_points = int(model_cfg.get("spatial_max_query_points", 262144))
        if self.context_size < 64 or self.context_size % self.patch_size:
            raise ValueError("context_size must be >=64 and divisible by 16.")
        if min(embed_dim, depth, heads, global_hidden, grid_hidden) <= 0 or embed_dim % heads:
            raise ValueError("Positive dimensions and embed_dim % num_heads == 0 required.")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive and finite.")
        if not (0 <= dropout < 1 and 0 <= drop_path < 1):
            raise ValueError("dropout and drop_path_rate must be in [0,1).")
        if not math.isfinite(coefficient) or coefficient <= 0 or self.max_query_points <= 0:
            raise ValueError("Grid coefficient bound and max_query_points must be positive.")
        self.spatial_grid_size = _parse_grid_size(model_cfg.get("spatial_grid_size", [8, 16, 16]))
        self.encoder = DinoVisionTransformer(
            img_size=self.context_size, patch_size=self.patch_size,
            embed_dim=embed_dim, depth=depth, num_heads=heads,
            mlp_ratio=mlp_ratio, drop_path_rate=drop_path,
            block_fn=partial(Block, attn_class=Attention, drop=dropout),
            block_chunks=0, num_register_tokens=0,
            use_cls_token=bool(model_cfg.get("use_cls_token", False)),
        )
        self.param_gen = GlobalKernelHead(embed_dim, global_hidden, dropout)
        self.spatial_grid_head = SharedFeatureGridHead(
            embed_dim, grid_hidden, self.spatial_grid_size, coefficient,
        )
        self.guide_map = LearnedGuideMap(int(model_cfg.get("guide_curve_bins", 16)))
        self.act = Swish()
        self.params = {}

    def predict_single_parameters(self, context):
        context = F.interpolate(context, size=(self.context_size, self.context_size),
                                mode="bilinear", align_corners=False)
        tokens = self.encoder.forward_features(context)["x_norm_patchtokens"]
        kernels = self.param_gen(tokens)
        side = self.context_size // self.patch_size
        features = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, side, side)
        grid = self.spatial_grid_head(features)
        self.params = {f"kernel{i}_single": kernel for i, kernel in enumerate(kernels, 1)}
        self.params["spatial_grid"] = grid
        return kernels, grid

    def forward_single_stages(self, image, parameters=None):
        parameters = self.predict_single_parameters(image) if parameters is None else parameters
        if not isinstance(parameters, (tuple, list)) or len(parameters) != 2:
            raise ValueError("Shared-ViT model expects (four Global kernels, spatial grid).")
        kernels, grid = parameters
        if not isinstance(kernels, (tuple, list)) or len(kernels) != 4:
            raise ValueError("Shared-ViT Global head must provide four kernels.")
        self.params.update({f"kernel{i}_single": kernel for i, kernel in enumerate(kernels, 1)})
        self.params["spatial_grid"] = grid
        global_image = self._apply_velocity(image, kernels)
        final_image = self._apply_spatial_grid(image, global_image, grid)
        return global_image, final_image - global_image, final_image


def parameter_counts(model):
    """Return trainable parameter counts grouped by top-level model module."""
    result = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in model.named_children()
    }
    result["total"] = sum(parameter.numel() for parameter in model.parameters())
    result["trainable"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return result
