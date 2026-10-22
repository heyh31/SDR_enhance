"""Supervised clip solver for previous-enhanced-input Shared ViT."""
from solvers.colorfm_l_spatial_grid_shared_vit_recurrent_solver import (
    ColorFM_L as RecurrentSolver,
)


class ColorFM_L(RecurrentSolver):
    """Use the common Global/Grid losses with autoregressive output clips."""

