from pathlib import Path
import unittest

import torch
from omegaconf import OmegaConf

from models.colorfm_l_spatial_grid_shared_vit import ColorFM_L
from models.colorfm_l_spatial_grid_shared_vit_recurrent import ColorFM_L as RecurrentColorFM


class SharedViTModelTest(unittest.TestCase):
    def test_parameter_and_stage_shapes(self):
        root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(root / "configs" / "shared_vit_sdr.yaml")
        model = ColorFM_L(cfg).eval()

        with torch.inference_mode():
            parameters = model.predict_single_parameters(torch.rand(1, 3, 256, 256))
            stages = model.forward_single_stages(torch.rand(1, 3, 32, 48), parameters)

        kernels, grid = parameters
        self.assertEqual([tuple(kernel.shape) for kernel in kernels], [
            (1, 3, 16), (1, 16, 16), (1, 16, 16), (1, 16, 3),
        ])
        self.assertEqual(tuple(grid.shape), (1, 12, 8, 16, 16))
        self.assertTrue(all(tuple(image.shape) == (1, 3, 32, 48) for image in stages))

    def test_recurrent_clip_shapes(self):
        root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(root / "configs" / "colorfm_l_spatial_grid_shared_vit_recurrent_sdr.yaml")
        cfg.data.train_clip_length = 2
        model = RecurrentColorFM(cfg).eval()

        with torch.inference_mode():
            context = torch.rand(1, 2, 3, 256, 256)
            parameters = model.predict_clip_parameters(context)
            stages = model.forward_clip_stages(torch.rand(1, 2, 3, 32, 48), parameters=parameters)

        kernels, grid = parameters
        self.assertEqual([tuple(kernel.shape) for kernel in kernels], [
            (2, 3, 16), (2, 16, 16), (2, 16, 16), (2, 16, 3),
        ])
        self.assertEqual(tuple(grid.shape), (2, 12, 8, 16, 16))
        self.assertTrue(all(tuple(image.shape) == (1, 2, 3, 32, 48) for image in stages))


if __name__ == "__main__":
    unittest.main()
