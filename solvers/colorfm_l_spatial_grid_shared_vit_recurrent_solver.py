from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

from solvers.colorfm_l_spatial_grid_light_solver import (
    ColorFM_L as LightSolver,
    mean_frame_psnr,
)


class ColorFM_L(LightSolver):
    """Train recurrent Global/Grid prediction on [B,N,C,H,W] clips."""

    def _shared_step(self, batch, stage):
        source, target, context = batch["source"], batch["target"], batch["context"]
        if source.ndim == 4:
            if stage == "train":
                raise ValueError("Recurrent training requires consecutive [B,N,C,H,W] clip batches.")
            self.model.reset_temporal_state()
            try:
                return super()._shared_step(batch, stage)
            finally:
                self.model.reset_temporal_state()
        if source.ndim != 5 or target.ndim != 5 or context.ndim != 5:
            raise ValueError("Recurrent batches must have [B,N,C,H,W] image/context tensors.")

        global_image, residual, final = self.model.forward_clip_stages(source, context)
        residual_l1 = residual.abs().mean()
        grid_tv = self.model.spatial_grid_total_variation()
        global_l1 = F.l1_loss(global_image, target)
        final_l1 = F.l1_loss(final, target)
        objective = (final_l1 + self.weights["global"] * global_l1
                     + self.weights["residual"] * residual_l1
                     + self.weights["tv"] * grid_tv)
        loss = objective if stage == "train" else final_l1

        batch_frames = source.shape[0] * source.shape[1]
        flat_source = source.reshape(batch_frames, *source.shape[2:])
        flat_target = target.reshape(batch_frames, *target.shape[2:])
        flat_global = global_image.reshape(batch_frames, *global_image.shape[2:])
        flat_final = final.reshape(batch_frames, *final.shape[2:])
        with torch.no_grad():
            clipped_global = flat_global.clamp(0, 1)
            clipped_final = flat_final.clamp(0, 1)
            values = {
                "loss": loss,
                "final_l1": final_l1,
                "global_l1": global_l1,
                "grid_residual_l1": residual_l1,
                "grid_tv": grid_tv,
                "source_psnr": mean_frame_psnr(flat_source, flat_target),
                "global_psnr": mean_frame_psnr(clipped_global, flat_target),
                "psnr": mean_frame_psnr(clipped_final, flat_target),
            }
            values["grid_gain"] = values["psnr"] - values["global_psnr"]
            if stage != "train":
                values["ssim"] = ssim(clipped_final.float(), flat_target.float(),
                                       data_range=1.0, size_average=True)
        for name, value in values.items():
            self.log(
                f"{stage}_{name}", value,
                on_step=stage == "train" and name == "loss",
                on_epoch=True, batch_size=batch_frames, sync_dist=True,
                prog_bar=name in ("loss", "psnr", "global_psnr"),
            )
        return loss
