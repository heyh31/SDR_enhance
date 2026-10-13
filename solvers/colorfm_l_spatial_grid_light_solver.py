from __future__ import annotations

import math

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_msssim import ssim

from solvers.colorfm_l_solver import ColorFM_L as BaseSolver


def mean_frame_psnr(prediction, target):
    """Arithmetic mean of per-frame PSNR; not PSNR of batch-average MSE."""
    mse = (prediction.float() - target.float()).square().flatten(1).mean(1)
    return (-10 * torch.log10(mse.clamp_min(1e-10))).mean()


class ColorFM_L(BaseSolver):
    """Explicit global supervision plus a regularized local residual."""

    def __init__(self, cfg, model):
        L.LightningModule.__init__(self)
        self.cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
        self.model = model
        # Plain containers keep new checkpoints compatible with weights-only
        # torch loading; do not pickle DictConfig inside hyperparameters.
        self.save_hyperparameters({"cfg": OmegaConf.to_container(self.cfg, resolve=True)})
        cfg = self.cfg
        self.split_fingerprint = None
        self.weights = {
            "global": float(cfg.solver.get("global_l1_weight", 1.0)),
            "residual": float(cfg.solver.get("grid_residual_l1_weight", 0.01)),
            "tv": float(cfg.solver.get("grid_tv_weight", 0.0001)),
        }
        if any(not math.isfinite(x) or x < 0 for x in self.weights.values()):
            raise ValueError("Loss weights must be finite and nonnegative.")

    def _shared_step(self, batch, stage):
        source, target = batch["source"], batch["target"]
        parameters = self.model.predict_single_kernels(batch["context"])
        has_grid = hasattr(self.model, "forward_single_stages")
        if has_grid:
            global_image, residual, final = self.model.forward_single_stages(source, parameters)
            residual_l1 = residual.abs().mean()
            grid_tv = self.model.spatial_grid_total_variation()
        else:
            global_image = final = self.model.forward_single(source, parameters)
            residual_l1 = grid_tv = source.new_zeros(())
        global_l1 = F.l1_loss(global_image, target)
        final_l1 = F.l1_loss(final, target)
        objective = final_l1
        if has_grid:
            objective = (objective + self.weights["global"] * global_l1
                         + self.weights["residual"] * residual_l1 + self.weights["tv"] * grid_tv)
        loss = objective if stage == "train" else final_l1
        values = {
            "loss": loss, "final_l1": final_l1, "global_l1": global_l1,
            "grid_residual_l1": residual_l1, "grid_tv": grid_tv,
        }
        with torch.no_grad():
            clipped_global, clipped_final = global_image.clamp(0, 1), final.clamp(0, 1)
            values.update(
                source_psnr=mean_frame_psnr(source, target),
                global_psnr=mean_frame_psnr(clipped_global, target),
                psnr=mean_frame_psnr(clipped_final, target),
            )
            values["grid_gain"] = values["psnr"] - values["global_psnr"]
            if stage != "train":
                values["ssim"] = ssim(clipped_final.float(), target.float(), data_range=1.0, size_average=True)
        for name, value in values.items():
            self.log(
                f"{stage}_{name}", value, on_step=stage == "train" and name == "loss",
                on_epoch=True, batch_size=source.shape[0], sync_dist=True,
                prog_bar=name in ("loss", "psnr", "global_psnr"),
            )
        return loss

    def configure_optimizers(self):
        # Decay learned weight matrices, preserving normalization and bias terms.
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                (no_decay if parameter.ndim < 2 or "pos_embed" in name else decay).append(parameter)
        train = self.cfg.train
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": float(train.get("weight_decay", 0.01))},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=float(train.lr), betas=tuple(float(x) for x in train.get("betas", [0.9, 0.999])),
        )
        epochs = int(train.max_epochs)
        warmup = int(train.get("warmup_epochs", 2))
        floor = float(train.get("eta_min", 1e-6)) / float(train.lr)
        if not (0 <= warmup < epochs and 0 <= floor <= 1):
            raise ValueError("Need 0 <= warmup_epochs < max_epochs and 0 <= eta_min <= lr.")

        def multiplier(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            progress = min(1.0, (epoch - warmup) / max(1, epochs - warmup - 1))
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["sdr_split_fingerprint"] = self.split_fingerprint

    def on_load_checkpoint(self, checkpoint):
        if self.split_fingerprint and checkpoint.get("sdr_split_fingerprint") != self.split_fingerprint:
            raise ValueError("Checkpoint temporal split differs from current data/config. Use a new experiment.")
