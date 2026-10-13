from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


class ColorFM_L(L.LightningModule):
    """Supervised solver for one-pass, source-only SDR enhancement.

    The target frame is used only by the losses.  The model receives the source
    frame once, predicts one velocity field, and applies one Euler transport
    step via ``model.forward_single``.
    """

    def __init__(self, cfg, model, *_unused_paths):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.save_hyperparameters(ignore=["model"])

    def forward(
        self,
        source: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Parameter prediction sees the complete frame resized to context_size,
        # while the generated pixel MLP is applied to the aligned training crop
        # (or to the complete frame during validation/test).
        context = source if context is None else context
        kernels = self.model.predict_single_kernels(context)
        return self.model.forward_single(source, kernels=kernels)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        source = batch["source"]
        target = batch["target"]
        context = batch["context"]
        prediction = self(source, context)

        # Paired SDR frames are spatially aligned, so direct L1 supervision is
        # used without a perceptual loss.
        loss = F.l1_loss(prediction, target)
        mse_loss = F.mse_loss(prediction, target)

        batch_size = source.shape[0]
        on_step = stage == "train"
        self.log(
            f"{stage}_loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_l1",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_mse",
            mse_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        with torch.no_grad():
            clipped_mse = F.mse_loss(prediction.clamp(0.0, 1.0), target)
            psnr = -10.0 * torch.log10(clipped_mse.clamp_min(1e-10))
        self.log(
            f"{stage}_psnr",
            psnr,
            on_step=False,
            on_epoch=True,
            prog_bar=stage != "train",
            sync_dist=True,
            batch_size=batch_size,
        )
        if stage != "train":
            with torch.no_grad():
                ssim_value = ssim(
                    prediction.clamp(0.0, 1.0),
                    target,
                    data_range=1.0,
                    size_average=True,
                )
            self.log(
                f"{stage}_ssim",
                ssim_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return loss

    def training_step(self, batch, batch_idx):
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        del batch_idx
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        del batch_idx
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        optimizer_name = str(self.cfg.train.optimizer)
        learning_rate = float(self.cfg.train.lr)
        betas = tuple(float(value) for value in self.cfg.train.betas)

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=learning_rate,
                betas=betas,
            )
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                trainable_parameters,
                lr=learning_rate,
                momentum=float(self.cfg.train.sgd.momentum),
            )
        elif optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=learning_rate,
                betas=betas,
                weight_decay=float(self.cfg.train.weight_decay),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        warmup_epochs = int(self.cfg.train.get("warmup_epochs", 0))
        constant_epochs = int(self.cfg.train.get("constant_epochs", 0))
        cosine_epochs = int(self.cfg.train.get("cosine_epochs", 0))
        max_epochs = int(self.cfg.train.max_epochs)
        scheduled_epochs = warmup_epochs + constant_epochs + cosine_epochs
        if scheduled_epochs not in (0, max_epochs):
            raise ValueError(
                "max_epochs must equal warmup_epochs + constant_epochs + "
                f"cosine_epochs, got {max_epochs} != {scheduled_epochs}."
            )
        if scheduled_epochs == 0:
            return optimizer

        schedulers = []
        durations = []
        if warmup_epochs > 0:
            schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.001,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
            )
            durations.append(warmup_epochs)
        if constant_epochs > 0:
            schedulers.append(
                torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=1.0,
                    total_iters=constant_epochs,
                )
            )
            durations.append(constant_epochs)
        if cosine_epochs > 0:
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_epochs,
                    eta_min=float(self.cfg.train.eta_min),
                )
            )
            durations.append(cosine_epochs)

        if len(schedulers) == 1:
            scheduler = schedulers[0]
        else:
            milestones = []
            elapsed = 0
            for duration in durations[:-1]:
                elapsed += duration
                milestones.append(elapsed)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=schedulers,
                milestones=milestones,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
