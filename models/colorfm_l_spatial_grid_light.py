from __future__ import annotations

from functools import partial
import math

import torch
from torch import nn

from models.colorfm_l import DinoVisionTransformer, Swish
from models.layers.block import Block
from models.layers.attention import Attention
from models.colorfm_l_spatial_grid import (
    ColorFM_L as SpatialColorFM,
    LearnedGuideMap,
    SpatialAffineGridHead,
    _parse_grid_size,
)


class LiteParamGenerator(nn.Module):
    """A narrow global head, retaining all four dynamic RGB kernel shapes."""

    def __init__(self, embed_dim, feature_grid_size, hidden_dim=96, dropout=0.05):
        super().__init__()
        self.feature_grid_size = feature_grid_size
        self.embed_dim = embed_dim
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.kernel_shapes = ((3, 16), (16, 16), (16, 16), (16, 3))
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, rows * cols),
            )
            for rows, cols in self.kernel_shapes
        ])
        # Only the final RGB kernel is zero: the initial global transform is
        # identity, and its output head can learn immediately. Zeroing all
        # four kernels would prevent useful gradient propagation.
        nn.init.zeros_(self.heads[-1][-1].weight)
        nn.init.zeros_(self.heads[-1][-1].bias)

    def forward(self, tokens):
        batch, count, channels = tokens.shape
        side = self.feature_grid_size
        if count != side * side or channels != self.embed_dim:
            raise ValueError(f"Unexpected encoder features: {tokens.shape}")
        features = tokens.transpose(1, 2).reshape(batch, channels, side, side)
        pooled = self.fuse(features)
        return tuple(
            head(pooled).reshape(batch, *shape)
            for head, shape in zip(self.heads, self.kernel_shapes)
        )


class ColorFM_L(SpatialColorFM):
    """Small ViT + global dynamic kernels + learned local affine grid.

    Reuses the existing parameter prediction, guide, slicing and image-space
    residual implementation. Only the trainable architecture is replaced.
    """

    def __init__(self, cfg):
        # Do not construct the large parent model only to discard its weights.
        nn.Module.__init__(self)
        model_cfg = cfg.model
        self.context_size = int(cfg.data.get("context_size", 256))
        self.patch_size = 16
        embed_dim = int(model_cfg.get("embed_dim", 192))
        depth = int(model_cfg.get("depth", 6))
        heads = int(model_cfg.get("num_heads", 3))
        mlp_ratio = float(model_cfg.get("mlp_ratio", 4.0))
        hidden = int(model_cfg.get("global_hidden_dim", 96))
        grid_hidden = int(model_cfg.get("spatial_hidden_dim", 64))
        dropout = float(model_cfg.get("dropout", 0.05))
        drop_path = float(model_cfg.get("drop_path_rate", 0.1))
        if self.context_size < 64 or self.context_size % self.patch_size:
            raise ValueError("context_size must be >=64 and divisible by 16.")
        if min(embed_dim, depth, heads, hidden, grid_hidden) <= 0 or embed_dim % heads:
            raise ValueError("Positive dimensions and embed_dim % num_heads == 0 required.")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive and finite.")
        if not (0 <= dropout < 1 and 0 <= drop_path < 1):
            raise ValueError("dropout and drop_path_rate must be in [0,1).")
        self.spatial_encoder_layers = [
            int(i) for i in model_cfg.get("spatial_encoder_layers", [1, 3, 5])
        ]
        layers = self.spatial_encoder_layers
        if not layers or len(set(layers)) != len(layers) or min(layers) < 0 or max(layers) >= depth:
            raise ValueError("spatial_encoder_layers must be unique zero-based indices < depth.")
        self.spatial_encoder_layers = sorted(layers)
        self.encoder_output_layer = depth - 1
        self.requested_encoder_layers = sorted(set(layers + [depth - 1]))
        self.encoder = DinoVisionTransformer(
            img_size=self.context_size, patch_size=self.patch_size,
            embed_dim=embed_dim, depth=depth, num_heads=heads,
            mlp_ratio=mlp_ratio, drop_path_rate=drop_path,
            block_fn=partial(Block, attn_class=Attention, drop=dropout),
            block_chunks=0, num_register_tokens=0,
        )
        self.param_gen = LiteParamGenerator(
            embed_dim, self.context_size // self.patch_size, hidden, dropout,
        )
        self.spatial_grid_size = _parse_grid_size(
            model_cfg.get("spatial_grid_size", [8, 16, 16])
        )
        coefficient = float(model_cfg.get("spatial_max_affine_coefficient", 0.25))
        if not math.isfinite(coefficient) or coefficient <= 0:
            raise ValueError("spatial_max_affine_coefficient must be positive and finite.")
        self.spatial_grid_head = SpatialAffineGridHead(
            feature_count=len(layers), embed_dim=embed_dim,
            hidden_dim=grid_hidden, grid_size=self.spatial_grid_size,
            max_coefficient=coefficient,
        )
        self.guide_map = LearnedGuideMap(int(model_cfg.get("guide_curve_bins", 16)))
        self.max_query_points = int(model_cfg.get("spatial_max_query_points", 262144))
        if self.max_query_points <= 0:
            raise ValueError("spatial_max_query_points must be positive.")
        self.act = Swish()
        self.params = {}


def parameter_counts(model):
    result = {name: sum(p.numel() for p in module.parameters())
              for name, module in model.named_children()}
    result["total"] = sum(p.numel() for p in model.parameters())
    result["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return result

