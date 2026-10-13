#!/usr/bin/env python
"""Train the shared-ViT Global/Grid model with recurrent consecutive clips."""
from pathlib import Path

from main_train_sdr_light import main
from solvers.colorfm_l_spatial_grid_shared_vit_recurrent_solver import ColorFM_L as Solver


if __name__ == "__main__":
    main(
        default_config=Path(__file__).resolve().parent
        / "configs/colorfm_l_spatial_grid_shared_vit_recurrent_sdr.yaml",
        solver_type=Solver,
    )
