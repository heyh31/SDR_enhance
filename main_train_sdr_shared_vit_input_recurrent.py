#!/usr/bin/env python
"""Train Shared ViT conditioned on the previous enhanced frame."""
from pathlib import Path

from main_train_sdr_light import main
from solvers.colorfm_l_spatial_grid_shared_vit_input_recurrent_solver import (
    ColorFM_L as Solver,
)


if __name__ == "__main__":
    main(
        default_config=Path(__file__).resolve().parent
        / "configs/input_recurrent.yaml",
        solver_type=Solver,
    )
