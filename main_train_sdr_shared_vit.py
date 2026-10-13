#!/usr/bin/env python
"""Train the single-ViT, two-head SDR model with random independent frames."""
from pathlib import Path

from main_train_sdr_light import main
from solvers.colorfm_l_spatial_grid_shared_vit_solver import ColorFM_L as Solver


if __name__ == "__main__":
    main(
        default_config=Path(__file__).resolve().parent / "configs/shared_vit_sdr.yaml",
        solver_type=Solver,
    )
