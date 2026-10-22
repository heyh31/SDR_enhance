"""Shared ViT conditioned on the previous enhanced frame."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.colorfm_l_spatial_grid_shared_vit import ColorFM_L as SharedViT


class ColorFM_L(SharedViT):
    """Autoregressive input conditioning for Shared ViT Global/Grid prediction.

    At time t, the ViT receives ``concat(source_t, enhanced_(t-1))`` as a
    six-channel image. The first frame uses its own source image as the
    previous-frame condition. The predicted Global/Grid are applied to the
    current source, and the clamped enhanced result conditions the next frame.
    """

    supports_temporal_recurrence = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.clip_length = int(cfg.data.get("train_clip_length", 10))
        if self.clip_length < 2:
            raise ValueError("Input-recurrent training requires train_clip_length >= 2.")
        self._replace_patch_embed_with_conditioned_input()
        self.reset_temporal_state()

    def _replace_patch_embed_with_conditioned_input(self):
        projection = self.encoder.patch_embed.proj
        conditioned = torch.nn.Conv2d(
            6, projection.out_channels,
            kernel_size=projection.kernel_size,
            stride=projection.stride,
            padding=projection.padding,
            dilation=projection.dilation,
            groups=1,
            bias=projection.bias is not None,
            padding_mode=projection.padding_mode,
        )
        # Preserve the original Shared-ViT response to the current source at
        # initialization; the previous-enhanced condition learns from zero.
        with torch.no_grad():
            conditioned.weight.zero_()
            conditioned.weight[:, :3].copy_(projection.weight)
            if projection.bias is not None:
                conditioned.bias.copy_(projection.bias)
        self.encoder.patch_embed.proj = conditioned

    def reset_temporal_state(self):
        """Start a new clip/video with no previous enhanced frame."""
        self._previous_enhanced = None
        self._temporal_frame_index = None
        self._pending_frame_index = None

    def _conditioned_tokens(self, context, previous_enhanced):
        if context.ndim != 4 or context.shape[1] != 3:
            raise ValueError(f"Expected current context [B,3,H,W], got {tuple(context.shape)}.")
        if previous_enhanced is None:
            previous_enhanced = context
        if previous_enhanced.ndim != 4 or previous_enhanced.shape[:2] != context.shape[:2]:
            raise ValueError("Previous enhanced condition must have shape [B,3,H,W].")
        size = (self.context_size, self.context_size)
        current = F.interpolate(context, size=size, mode="bilinear", align_corners=False)
        previous = F.interpolate(
            previous_enhanced.to(device=context.device, dtype=context.dtype),
            size=size, mode="bilinear", align_corners=False,
        ).clamp(0.0, 1.0)
        conditioned = torch.cat((current, previous), dim=1)
        return self.encoder.forward_features(conditioned)["x_norm_patchtokens"]

    def predict_single_parameters(
        self, context, previous_enhanced=None, frame_index=None,
    ):
        """Predict current parameters, using prior output unless explicitly given."""
        if frame_index is not None and self._temporal_frame_index is not None:
            if int(frame_index) != self._temporal_frame_index + 1:
                self.reset_temporal_state()
        if previous_enhanced is None:
            previous_enhanced = self._previous_enhanced
        tokens = self._conditioned_tokens(context, previous_enhanced)
        kernels = self.param_gen(tokens)
        side = self.context_size // self.patch_size
        features = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, side, side)
        grid = self.spatial_grid_head(features)
        self.params = {f"kernel{i}_single": kernel for i, kernel in enumerate(kernels, 1)}
        self.params["spatial_grid"] = grid
        self._pending_frame_index = None if frame_index is None else int(frame_index)
        return kernels, grid

    def forward_single_stages(self, image, parameters=None):
        if parameters is None:
            parameters = self.predict_single_parameters(image)
        result = super().forward_single_stages(image, parameters)
        self._previous_enhanced = result[-1].detach().clamp(0.0, 1.0)
        self._temporal_frame_index = self._pending_frame_index
        self._pending_frame_index = None
        return result

    def forward_clip_stages(self, source, context=None, parameters=None):
        """Sequentially process a [B,N,C,H,W] clip with BPTT through outputs."""
        if source.ndim != 5:
            raise ValueError(f"Expected source [B,N,C,H,W], got {tuple(source.shape)}.")
        batch, frames, channels, height, width = source.shape
        if channels != 3 or frames != self.clip_length:
            raise ValueError(
                f"Expected [B,{self.clip_length},3,H,W], got {tuple(source.shape)}."
            )
        if context is None:
            context = source
        if context.ndim != 5 or context.shape[:3] != source.shape[:3]:
            raise ValueError("Context must have the same [B,N,C] dimensions as source.")

        previous_enhanced = None
        global_frames, residual_frames, final_frames = [], [], []
        kernel_frames = [[] for _ in range(4)]
        grid_frames = []
        for index in range(frames):
            current_context = context[:, index]
            condition = current_context if previous_enhanced is None else previous_enhanced
            parameters = self.predict_single_parameters(
                current_context, previous_enhanced=condition,
            )
            kernels, grid = parameters
            for layer, kernel in enumerate(kernels):
                kernel_frames[layer].append(kernel)
            grid_frames.append(grid)
            global_image, residual, final_image = super().forward_single_stages(
                source[:, index], parameters,
            )
            global_frames.append(global_image)
            residual_frames.append(residual)
            final_frames.append(final_image)
            # Match video output range before feeding this result to the next
            # frame, while retaining gradients through the un-detached tensor.
            previous_enhanced = final_image.clamp(0.0, 1.0)

        self._previous_enhanced = previous_enhanced.detach()
        self.params = {
            f"kernel{layer + 1}_single": torch.cat(values, dim=0)
            for layer, values in enumerate(kernel_frames)
        }
        self.params["spatial_grid"] = torch.cat(grid_frames, dim=0)
        self._temporal_frame_index = None
        self._pending_frame_index = None
        return tuple(torch.stack(values, dim=1) for values in
                     (global_frames, residual_frames, final_frames))
