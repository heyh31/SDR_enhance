#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from datasets.sdr_pairing import discover_video_pairs, split_video_pairs


SUPPORTED_ARCHITECTURES = {
    "colorfm_l_spatial_grid_shared_vit",
    "colorfm_l_spatial_grid_shared_vit_recurrent",
}
GRID_ARCHITECTURES = {
    "colorfm_l_spatial_grid_shared_vit",
    "colorfm_l_spatial_grid_shared_vit_recurrent",
}
GRID_AFTER_GLOBAL_ARCHITECTURE = None


def add_video_writer_arguments(parser):
    parser.add_argument(
        "--writer-backend",
        choices=("ffmpeg", "opencv"),
        default="ffmpeg",
        help=(
            "Video encoding backend. FFmpeg explicitly applies one color "
            "conversion to all four outputs; OpenCV is retained as a fallback. "
            "Default: ffmpeg."
        ),
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help=(
            "FourCC used only with --writer-backend opencv. Default: mp4v."
        ),
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="FFmpeg executable name or path. Default: ffmpeg.",
    )
    parser.add_argument(
        "--ffmpeg-codec",
        default="libx264",
        help="FFmpeg video encoder. Default: libx264.",
    )
    parser.add_argument(
        "--ffmpeg-crf",
        type=int,
        default=10,
        help="FFmpeg constant-rate factor (0-51). Default: 10.",
    )
    parser.add_argument(
        "--ffmpeg-preset",
        default="medium",
        help="FFmpeg encoder preset. Default: medium.",
    )
    parser.add_argument(
        "--ffmpeg-pix-fmt",
        default="yuv420p",
        help=(
            "FFmpeg output pixel format. yuv420p is browser compatible; "
            "yuv444p avoids chroma subsampling for diagnostics. Default: yuv420p."
        ),
    )
    parser.add_argument(
        "--ffmpeg-color-space",
        choices=("bt709", "bt601"),
        default="bt709",
        help="RGB-to-YUV matrix and output color tags. Default: bt709.",
    )
    parser.add_argument(
        "--ffmpeg-color-range",
        choices=("tv", "pc"),
        default="tv",
        help="Output YUV range: tv=limited, pc=full. Default: tv.",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained SDR model on one aligned source/GT video interval "
            "and save source, GT, Global, and Global+Local clips."
        )
    )
    parser.add_argument("--config", required=True, help="Training YAML path.")
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help=(
            "Checkpoint path, or 'auto' to select the configured best-metric "
            "checkpoint under output_dir/exp_name/checkpoints."
        ),
    )
    parser.add_argument(
        "--video",
        default=None,
        help=(
            "Unique full or partial source video name under data.root. If "
            "omitted, the configured test video is used."
        ),
    )
    parser.add_argument(
        "--source-video",
        default=None,
        help="Direct source video path; must be used with --gt-video.",
    )
    parser.add_argument(
        "--gt-video",
        default=None,
        help="Direct aligned GT video path; must be used with --source-video.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional override for data.root.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        required=True,
        help="Inclusive start time in seconds.",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        required=True,
        help="Exclusive end time in seconds.",
    )
    parser.add_argument(
        "--kernel-average",
        choices=("on", "off"),
        default="on",
        help=(
            "Average Global kernels over neighboring source frames. This works "
            "for every supported architecture. Default: on."
        ),
    )
    parser.add_argument(
        "--kernel-offsets",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Relative frame offsets used for kernel averaging, for example "
            "--kernel-offsets -2 -1 0 1 2. Defaults to config values or a "
            "five-frame window."
        ),
    )
    parser.add_argument(
        "--kernel-weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional averaging weights corresponding to --kernel-offsets. "
            "Defaults to config values when compatible, otherwise equal weights."
        ),
    )
    parser.add_argument(
        "--kernel-average-alpha",
        type=float,
        default=None,
        help=(
            "Blend between the center kernel and temporal mean: 0=center only, "
            "1=full mean. Defaults to model.temporal_average_alpha or 1."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device, for example cuda:0 or cpu. Default: auto.",
    )
    parser.add_argument(
        "--output-dir",
        default="./video_inference_outputs",
        help="Parent output directory.",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=16,
        help="Number of output frames retained in one CPU-side chunk.",
    )
    parser.add_argument(
        "--parameter-batch-size",
        type=int,
        default=8,
        help="Low-resolution parameter-prediction batch size.",
    )
    parser.add_argument(
        "--render-batch-size",
        type=int,
        default=1,
        help="Full-resolution Global/Grid rendering batch size.",
    )
    add_video_writer_arguments(parser)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files for the same interval.",
    )
    return parser.parse_args()


def build_model(cfg):
    architecture = str(cfg.model.get("architecture", "colorfm_l")).casefold()
    if architecture == "colorfm_l_spatial_grid_shared_vit":
        from models.colorfm_l_spatial_grid_shared_vit import ColorFM_L as ModelClass
    elif architecture == "colorfm_l_spatial_grid_shared_vit_recurrent":
        from models.colorfm_l_spatial_grid_shared_vit_recurrent import ColorFM_L as ModelClass
    else:
        raise ValueError(
            f"Unsupported model.architecture={architecture!r}; supported="
            f"{sorted(SUPPORTED_ARCHITECTURES)}"
        )

    return architecture, ModelClass(cfg)


def resolve_device(requested):
    if str(requested).casefold() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({requested}) but is unavailable.")
    return device


def resolve_checkpoint(cfg, requested):
    if requested and str(requested).casefold() != "auto":
        path = Path(requested).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    checkpoint_dir = (
        Path(str(cfg.output_dir)).expanduser()
        / str(cfg.exp_name)
        / "checkpoints"
    )
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )

    solver = cfg.get("solver", {})
    metric = str(solver.get("best_checkpoint_metric", "val_loss"))
    mode = str(solver.get("best_checkpoint_mode", "min"))
    for metric_name, metric_mode in [(metric, mode), ("val_loss", "min")]:
        scored = []
        pattern = re.compile(rf"{re.escape(metric_name)}=([0-9]+(?:\.[0-9]+)?)")
        for path in checkpoint_dir.glob("*.ckpt"):
            match = pattern.search(path.name)
            if match:
                scored.append((float(match.group(1)), path.stat().st_mtime, path))
        if scored:
            if metric_mode == "max":
                return max(scored, key=lambda item: (item[0], item[1]))[2]
            return min(scored, key=lambda item: (item[0], -item[1]))[2]

    last = checkpoint_dir / "last.ckpt"
    if last.is_file():
        return last
    candidates = list(checkpoint_dir.glob("*.ckpt"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(f"No .ckpt files found under {checkpoint_dir}")


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(model, checkpoint_path):
    payload = _torch_load(checkpoint_path)
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint has no state_dict: {checkpoint_path}")

    model_state = {}
    for original_key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        key = original_key
        prefix_removed = True
        while prefix_removed:
            prefix_removed = False
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    prefix_removed = True
        model_state[key] = value

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the configured model. "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    return {
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "global_step": (
            payload.get("global_step") if isinstance(payload, dict) else None
        ),
    }


def video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    metadata = {
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if metadata["frame_count"] <= 0 or metadata["fps"] <= 0.0:
        raise ValueError(f"Invalid video metadata for {path}: {metadata}")
    if metadata["width"] <= 0 or metadata["height"] <= 0:
        raise ValueError(f"Invalid video geometry for {path}: {metadata}")
    return metadata


def select_video_pair(cfg, args):
    direct_source = args.source_video is not None
    direct_target = args.gt_video is not None
    if direct_source != direct_target:
        raise ValueError("--source-video and --gt-video must be provided together.")
    if direct_source:
        if args.video is not None:
            raise ValueError("Do not combine --video with direct video paths.")
        source = Path(args.source_video).expanduser()
        target = Path(args.gt_video).expanduser()
        if not source.is_file() or not target.is_file():
            raise FileNotFoundError(
                f"Direct source/GT paths do not both exist: {source}, {target}"
            )
        return source.stem, source, target

    temporal_tail = str(cfg.data.get("split_mode", "")) == "temporal_tail"
    pairs = discover_video_pairs(
        root=cfg.data.root,
        source_dir=str(cfg.data.source_dir),
        target_dir=str(cfg.data.target_dir),
        minimum_pairs=1 if temporal_tail or args.video is not None else 3,
    )
    if args.video is None:
        if temporal_tail:
            # All videos participate in tail-split training. The inference
            # interval is explicit; no held-out-video split_seed is needed.
            pair = pairs[0]
            return pair.name, pair.source, pair.target
        splits = split_video_pairs(
            pairs,
            seed=int(cfg.data.split_seed),
            val_video=cfg.data.get("val_video"),
            test_video=cfg.data.get("test_video"),
        )
        pair = splits["test"][0]
        return pair.name, pair.source, pair.target

    requested = str(args.video).casefold().strip()
    exact = [
        pair
        for pair in pairs
        if requested in {pair.name.casefold(), pair.source.name.casefold()}
    ]
    candidates = exact or [
        pair
        for pair in pairs
        if requested in pair.name.casefold()
        or requested in pair.source.name.casefold()
        or requested in pair.target.name.casefold()
    ]
    if len(candidates) != 1:
        names = [pair.source.name for pair in candidates[:10]]
        raise ValueError(
            f"--video={args.video!r} matched {len(candidates)} pairs: {names}. "
            "Use a unique full or partial source filename."
        )
    pair = candidates[0]
    return pair.name, pair.source, pair.target


def resolve_kernel_spec(cfg, args):
    enabled = args.kernel_average == "on"
    if not enabled:
        return False, (0,), (1.0,), 0.0

    if args.kernel_offsets is not None:
        offsets = tuple(int(value) for value in args.kernel_offsets)
    else:
        configured_offsets = cfg.data.get("temporal_frame_offsets")
        if configured_offsets is not None:
            offsets = tuple(int(value) for value in configured_offsets)
        else:
            frame_count = int(cfg.data.get("temporal_num_frames", 5))
            stride = int(cfg.data.get("temporal_frame_stride", 1))
            if frame_count <= 0 or frame_count % 2 == 0:
                raise ValueError(
                    "A generated averaging window requires an odd positive "
                    f"temporal_num_frames, got {frame_count}."
                )
            if stride <= 0:
                raise ValueError("temporal_frame_stride must be positive.")
            radius = frame_count // 2
            offsets = tuple(
                offset * stride for offset in range(-radius, radius + 1)
            )

    if not offsets or 0 not in offsets:
        raise ValueError("Kernel averaging offsets must be non-empty and contain zero.")
    if len(set(offsets)) != len(offsets):
        raise ValueError("Kernel averaging offsets cannot contain duplicates.")

    if args.kernel_weights is not None:
        weights = tuple(float(value) for value in args.kernel_weights)
    else:
        configured_weights = cfg.model.get("temporal_average_weights")
        if configured_weights is not None and len(configured_weights) == len(offsets):
            weights = tuple(float(value) for value in configured_weights)
        else:
            weights = (1.0,) * len(offsets)
    if len(weights) != len(offsets):
        raise ValueError(
            f"Kernel weights/offsets length mismatch: {len(weights)} vs "
            f"{len(offsets)}."
        )
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("Kernel weights must be finite and non-negative.")
    weight_sum = math.fsum(weights)
    if weight_sum <= 0.0:
        raise ValueError("Kernel weights cannot sum to zero.")
    weights = tuple(value / weight_sum for value in weights)

    alpha = (
        float(args.kernel_average_alpha)
        if args.kernel_average_alpha is not None
        else float(cfg.model.get("temporal_average_alpha", 1.0))
    )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Kernel average alpha must be between zero and one.")
    return True, offsets, weights, alpha


class IndexedVideoReader:
    def __init__(self, path):
        self.path = Path(path)
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video: {self.path}")
        self.next_index = None

    def read(self, frame_index):
        frame_index = int(frame_index)
        if self.next_index != frame_index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(
                f"Unable to decode frame {frame_index} from {self.path}"
            )
        self.next_index = frame_index + 1
        return frame

    def close(self):
        self.capture.release()


def bgr_to_context_tensor(frame, context_size):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dtype = rgb.dtype
    array = np.ascontiguousarray(rgb, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    if np.issubdtype(dtype, np.integer):
        tensor.div_(float(np.iinfo(dtype).max))
    tensor = F.interpolate(
        tensor,
        size=(context_size, context_size),
        mode="bilinear",
        align_corners=False,
    )
    return tensor[0]


def bgr_batch_to_tensor(frames, device):
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    dtype = rgb_frames[0].dtype
    array = np.ascontiguousarray(np.stack(rgb_frames), dtype=np.float32)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
    if np.issubdtype(dtype, np.integer):
        tensor.div_(float(np.iinfo(dtype).max))
    return tensor.to(device, non_blocking=True)


def tensor_batch_to_bgr(tensor):
    rgb = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    rgb = np.rint(rgb * 255.0).astype(np.uint8)
    return [np.ascontiguousarray(frame[:, :, ::-1]) for frame in rgb]


def clamp_frame_index(frame_index, frame_count):
    return min(max(int(frame_index), 0), frame_count - 1)


def required_indices(center_indices, offsets, frame_count):
    return {
        clamp_frame_index(center + offset, frame_count)
        for center in center_indices
        for offset in offsets
    }


def predict_missing_parameters(
    model,
    has_grid,
    grid_after_global,
    missing_indices,
    frame_cache,
    kernel_cache,
    grid_cache,
    context_size,
    device,
    batch_size,
):
    ordered = sorted(missing_indices)
    if getattr(model, "supports_temporal_recurrence", False):
        for frame_index in ordered:
            context = bgr_to_context_tensor(frame_cache[frame_index], context_size).to(
                device, non_blocking=True
            )
            kernels, grids = model.predict_single_parameters(
                context, frame_index=frame_index
            )
            kernel_cache[frame_index] = tuple(
                kernel[0].detach().cpu() for kernel in kernels
            )
            grid_cache[frame_index] = grids[0].detach().cpu()
        return

    for start in range(0, len(ordered), batch_size):
        batch_indices = ordered[start : start + batch_size]
        contexts = torch.stack(
            [
                bgr_to_context_tensor(frame_cache[index], context_size)
                for index in batch_indices
            ]
        ).to(device, non_blocking=True)
        if has_grid and not grid_after_global:
            kernels, grids = model.predict_single_parameters(contexts)
        else:
            kernels = model.predict_single_kernels(contexts)
            grids = None

        for position, frame_index in enumerate(batch_indices):
            kernel_cache[frame_index] = tuple(
                kernel[position].detach().cpu() for kernel in kernels
            )
            if grids is not None:
                grid_cache[frame_index] = grids[position].detach().cpu()


def average_kernels_for_frame(
    center_index,
    kernel_cache,
    offsets,
    weights,
    alpha,
    frame_count,
):
    center_kernels = kernel_cache[center_index]
    frame_indices = [
        clamp_frame_index(center_index + offset, frame_count)
        for offset in offsets
    ]
    fused = []
    for layer, center_kernel in enumerate(center_kernels):
        stacked = torch.stack(
            [kernel_cache[index][layer] for index in frame_indices],
            dim=0,
        )
        layer_weights = torch.tensor(weights, dtype=stacked.dtype).view(
            len(weights), *([1] * (stacked.ndim - 1))
        )
        temporal_mean = (stacked * layer_weights).sum(dim=0)
        fused.append(center_kernel + alpha * (temporal_mean - center_kernel))
    return tuple(fused)


class FFmpegVideoWriter:
    """Write uint8 BGR frames to FFmpeg through a raw-video stdin pipe."""

    def __init__(
        self,
        path,
        fps,
        width,
        height,
        overwrite,
        ffmpeg_bin,
        codec,
        crf,
        preset,
        pixel_format,
        color_space,
        color_range,
    ):
        executable = shutil.which(str(ffmpeg_bin))
        if executable is None:
            explicit = Path(str(ffmpeg_bin)).expanduser()
            if explicit.is_file():
                executable = str(explicit)
            else:
                raise FileNotFoundError(
                    f"FFmpeg executable was not found: {ffmpeg_bin!r}. Install "
                    "FFmpeg, pass --ffmpeg-bin, or use --writer-backend opencv."
                )

        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self._closed = False
        if str(pixel_format).casefold() == "yuv420p" and (
            self.width % 2 or self.height % 2
        ):
            raise ValueError(
                "yuv420p requires an even frame width and height, got "
                f"{self.width}x{self.height}. Use --ffmpeg-pix-fmt yuv444p."
            )

        matrix = "bt709" if color_space == "bt709" else "bt601"
        color_tag = "bt709" if color_space == "bt709" else "smpte170m"
        output_range = "tv" if color_range == "tv" else "pc"
        scale_filter = (
            "scale=in_range=pc:"
            f"out_range={output_range}:out_color_matrix={matrix}"
        )
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y" if overwrite else "-n",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            f"{float(fps):.12g}",
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            scale_filter,
            "-c:v",
            str(codec),
            "-preset",
            str(preset),
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            str(pixel_format),
            "-colorspace",
            color_tag,
            "-color_primaries",
            color_tag,
            "-color_trc",
            color_tag,
            "-color_range",
            output_range,
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _stderr_text(self):
        if self._process.stderr is None:
            return ""
        data = self._process.stderr.read()
        return data.decode("utf-8", errors="replace").strip()

    def _raise_stopped(self):
        return_code = self._process.poll()
        stderr = self._stderr_text()
        raise RuntimeError(
            f"FFmpeg stopped while writing {self.path} (exit={return_code})."
            + (f"\n{stderr}" if stderr else "")
        )

    def write(self, frame):
        if self._closed:
            raise RuntimeError(f"Cannot write to closed FFmpeg output: {self.path}")
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Frame shape for {self.path} is {frame.shape}; expected "
                f"({self.height}, {self.width}, 3)."
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        if self._process.poll() is not None:
            self._raise_stopped()
        try:
            self._process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            self._raise_stopped()

    def release(self):
        if self._closed:
            return
        self._closed = True
        close_error = None
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except (BrokenPipeError, OSError) as exc:
                close_error = exc
        stderr = self._stderr_text()
        return_code = self._process.wait()
        if return_code != 0 or close_error is not None:
            detail = stderr or (str(close_error) if close_error else "")
            raise RuntimeError(
                f"FFmpeg failed while finalizing {self.path} "
                f"(exit={return_code})." + (f"\n{detail}" if detail else "")
            )


def release_video_writers(writers):
    errors = []
    for name, writer in writers.items():
        try:
            writer.release()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError("Failed to finalize video outputs:\n" + "\n".join(errors))


def open_video_writers(
    paths,
    codec,
    fps,
    width,
    height,
    overwrite,
    writer_backend="ffmpeg",
    ffmpeg_bin="ffmpeg",
    ffmpeg_codec="libx264",
    ffmpeg_crf=10,
    ffmpeg_preset="medium",
    ffmpeg_pix_fmt="yuv420p",
    ffmpeg_color_space="bt709",
    ffmpeg_color_range="tv",
):
    writer_backend = str(writer_backend).casefold()
    if writer_backend not in {"ffmpeg", "opencv"}:
        raise ValueError(f"Unsupported writer backend: {writer_backend!r}.")
    if writer_backend == "opencv" and len(codec) != 4:
        raise ValueError(f"--codec must contain four characters, got {codec!r}.")
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist: {existing}. Use --overwrite to replace."
        )
    writers = {}
    try:
        for name, path in paths.items():
            if writer_backend == "ffmpeg":
                writer = FFmpegVideoWriter(
                    path=path,
                    fps=fps,
                    width=width,
                    height=height,
                    overwrite=overwrite,
                    ffmpeg_bin=ffmpeg_bin,
                    codec=ffmpeg_codec,
                    crf=ffmpeg_crf,
                    preset=ffmpeg_preset,
                    pixel_format=ffmpeg_pix_fmt,
                    color_space=ffmpeg_color_space,
                    color_range=ffmpeg_color_range,
                )
            else:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    str(path),
                    fourcc,
                    float(fps),
                    (int(width), int(height)),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Unable to open VideoWriter for {path} "
                        f"with codec={codec!r}."
                    )
            writers[name] = writer
    except Exception:
        release_video_writers(writers)
        raise
    return writers


def safe_name(value):
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(value))
    return value.strip("._-") or "video"


def main():
    args = parse_args()
    if not math.isfinite(args.start_sec) or args.start_sec < 0.0:
        raise ValueError("--start-sec must be finite and non-negative.")
    if not math.isfinite(args.end_sec) or args.end_sec <= args.start_sec:
        raise ValueError("--end-sec must be finite and greater than --start-sec.")
    for name in ("chunk_frames", "parameter_batch_size", "render_batch_size"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.writer_backend == "ffmpeg" and not 0 <= args.ffmpeg_crf <= 51:
        raise ValueError("--ffmpeg-crf must be between 0 and 51.")

    cfg = OmegaConf.load(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    torch.set_float32_matmul_precision(
        str(cfg.trainer.get("matmul_precision", "high"))
    )
    pair_name, source_path, target_path = select_video_pair(cfg, args)
    source_meta = video_metadata(source_path)
    target_meta = video_metadata(target_path)
    if (source_meta["width"], source_meta["height"]) != (
        target_meta["width"],
        target_meta["height"],
    ):
        raise ValueError(
            "Source/GT geometry mismatch: "
            f"source={source_meta['width']}x{source_meta['height']}, "
            f"GT={target_meta['width']}x{target_meta['height']}"
        )
    if abs(source_meta["fps"] - target_meta["fps"]) > 0.05:
        raise ValueError(
            f"Source/GT FPS mismatch: {source_meta['fps']} vs {target_meta['fps']}"
        )

    frame_count = min(source_meta["frame_count"], target_meta["frame_count"])
    fps = source_meta["fps"]
    start_frame = int(round(args.start_sec * fps))
    requested_end_frame = int(round(args.end_sec * fps))
    if start_frame >= frame_count:
        raise ValueError(
            f"Start time maps to frame {start_frame}, outside {frame_count} frames."
        )
    end_frame = min(requested_end_frame, frame_count)
    if end_frame <= start_frame:
        raise ValueError("The requested interval contains no complete video frames.")
    if end_frame != requested_end_frame:
        print(
            f"Requested end was clamped to paired duration: "
            f"{end_frame / fps:.6f}s."
        )

    kernel_average, offsets, weights, average_alpha = resolve_kernel_spec(
        cfg, args
    )
    device = resolve_device(args.device)
    architecture, model = build_model(cfg)
    has_grid = architecture in GRID_ARCHITECTURES
    grid_after_global = architecture == GRID_AFTER_GLOBAL_ARCHITECTURE
    checkpoint_path = resolve_checkpoint(cfg, args.checkpoint)
    checkpoint_meta = load_checkpoint(model, checkpoint_path)
    model.to(device).eval()
    if getattr(model, "supports_temporal_recurrence", False):
        model.reset_temporal_state()

    mode_name = (
        f"kernel_avg{len(offsets)}_alpha{average_alpha:g}"
        if kernel_average
        else "kernel_single"
    )
    interval_name = f"f{start_frame:08d}-{end_frame - 1:08d}"
    output_dir = (
        Path(args.output_dir).expanduser()
        / f"{safe_name(pair_name)}_{interval_name}_{mode_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "source": output_dir / "source.mp4",
        "gt": output_dir / "gt.mp4",
        "global": output_dir / "global.mp4",
        "final": output_dir / "global_local.mp4",
    }
    writers = open_video_writers(
        output_paths,
        args.codec,
        fps,
        source_meta["width"],
        source_meta["height"],
        args.overwrite,
        writer_backend=args.writer_backend,
        ffmpeg_bin=args.ffmpeg_bin,
        ffmpeg_codec=args.ffmpeg_codec,
        ffmpeg_crf=args.ffmpeg_crf,
        ffmpeg_preset=args.ffmpeg_preset,
        ffmpeg_pix_fmt=args.ffmpeg_pix_fmt,
        ffmpeg_color_space=args.ffmpeg_color_space,
        ffmpeg_color_range=args.ffmpeg_color_range,
    )

    print(f"Architecture: {architecture}")
    print(f"Checkpoint: {checkpoint_path}")
    print(
        f"Epoch / step: {checkpoint_meta['epoch']} / "
        f"{checkpoint_meta['global_step']}"
    )
    print(f"Device: {device}")
    print(f"Source: {source_path}")
    print(f"GT: {target_path}")
    if args.writer_backend == "ffmpeg":
        print(
            "Video writer: FFmpeg | "
            f"codec={args.ffmpeg_codec} | crf={args.ffmpeg_crf} | "
            f"preset={args.ffmpeg_preset} | pix_fmt={args.ffmpeg_pix_fmt} | "
            f"colors={args.ffmpeg_color_space}/{args.ffmpeg_color_range}"
        )
    else:
        print(f"Video writer: OpenCV | FourCC={args.codec}")
    print(
        f"Interval: [{start_frame / fps:.6f}s, {end_frame / fps:.6f}s) | "
        f"frames={start_frame}..{end_frame - 1} ({end_frame - start_frame}) | "
        f"fps={fps:.6f}"
    )
    print(
        "Kernel averaging: "
        f"{'on' if kernel_average else 'off'} | offsets={list(offsets)} | "
        f"weights={[round(value, 8) for value in weights]} | "
        f"alpha={average_alpha:g}"
    )
    if not has_grid:
        print(
            "This architecture has no Local/Grid branch; global_local.mp4 will "
            "be identical to global.mp4."
        )

    source_reader = IndexedVideoReader(source_path)
    target_reader = IndexedVideoReader(target_path)
    frame_cache = {}
    kernel_cache = {}
    grid_cache = {}
    context_size = int(cfg.data.context_size)
    total_frames = end_frame - start_frame
    completed = 0
    started = time.perf_counter()

    try:
        with torch.inference_mode():
            for chunk_start in range(start_frame, end_frame, args.chunk_frames):
                chunk_end = min(chunk_start + args.chunk_frames, end_frame)
                centers = list(range(chunk_start, chunk_end))
                needed = required_indices(centers, offsets, frame_count)
                missing_frames = sorted(needed.difference(frame_cache))
                for frame_index in missing_frames:
                    frame_cache[frame_index] = source_reader.read(frame_index)

                missing_parameters = needed.difference(kernel_cache)
                predict_missing_parameters(
                    model=model,
                    has_grid=has_grid,
                    grid_after_global=grid_after_global,
                    missing_indices=missing_parameters,
                    frame_cache=frame_cache,
                    kernel_cache=kernel_cache,
                    grid_cache=grid_cache,
                    context_size=context_size,
                    device=device,
                    batch_size=args.parameter_batch_size,
                )

                for render_start in range(0, len(centers), args.render_batch_size):
                    render_indices = centers[
                        render_start : render_start + args.render_batch_size
                    ]
                    source_frames = [frame_cache[index] for index in render_indices]
                    source_tensor = bgr_batch_to_tensor(source_frames, device)
                    frame_kernels = [
                        average_kernels_for_frame(
                            center_index=index,
                            kernel_cache=kernel_cache,
                            offsets=offsets,
                            weights=weights,
                            alpha=average_alpha,
                            frame_count=frame_count,
                        )
                        for index in render_indices
                    ]
                    batched_kernels = tuple(
                        torch.stack(
                            [kernels[layer] for kernels in frame_kernels]
                        ).to(device, non_blocking=True)
                        for layer in range(4)
                    )

                    if grid_after_global:
                        global_tensor, _, final_tensor = model.forward_single_stages(
                            source_tensor,
                            parameters=batched_kernels,
                        )
                    elif has_grid:
                        batched_grid = torch.stack(
                            [grid_cache[index] for index in render_indices]
                        ).to(device, non_blocking=True)
                        global_tensor, _, final_tensor = model.forward_single_stages(
                            source_tensor,
                            parameters=(batched_kernels, batched_grid),
                        )
                    else:
                        global_tensor = model.forward_single(
                            source_tensor,
                            kernels=batched_kernels,
                        )
                        final_tensor = global_tensor

                    global_frames = tensor_batch_to_bgr(global_tensor)
                    final_frames = tensor_batch_to_bgr(final_tensor)
                    for position, frame_index in enumerate(render_indices):
                        gt_frame = target_reader.read(frame_index)
                        if gt_frame.shape != source_frames[position].shape:
                            raise ValueError(
                                f"Decoded geometry mismatch at frame {frame_index}: "
                                f"source={source_frames[position].shape}, "
                                f"GT={gt_frame.shape}"
                            )
                        writers["source"].write(source_frames[position])
                        writers["gt"].write(gt_frame)
                        writers["global"].write(global_frames[position])
                        writers["final"].write(final_frames[position])
                        completed += 1

                if chunk_end < end_frame:
                    next_end = min(chunk_end + args.chunk_frames, end_frame)
                    retained = required_indices(
                        range(chunk_end, next_end),
                        offsets,
                        frame_count,
                    )
                else:
                    retained = set()
                frame_cache = {
                    index: value
                    for index, value in frame_cache.items()
                    if index in retained
                }
                kernel_cache = {
                    index: value
                    for index, value in kernel_cache.items()
                    if index in retained
                }
                grid_cache = {
                    index: value
                    for index, value in grid_cache.items()
                    if index in retained
                }

                elapsed = time.perf_counter() - started
                speed = completed / max(elapsed, 1e-9)
                print(
                    f"\rProcessed: {completed}/{total_frames} frames | "
                    f"{speed:.3f} fps",
                    end="",
                    flush=True,
                )
    finally:
        source_reader.close()
        target_reader.close()
        release_video_writers(writers)
    print()

    elapsed = time.perf_counter() - started
    for name, path in output_paths.items():
        output_meta = video_metadata(path)
        if output_meta["frame_count"] != completed:
            raise RuntimeError(
                f"Output frame-count mismatch for {name}: wrote {completed}, "
                f"decoded metadata reports {output_meta['frame_count']} ({path})."
            )
        if (output_meta["width"], output_meta["height"]) != (
            source_meta["width"],
            source_meta["height"],
        ):
            raise RuntimeError(
                f"Output geometry mismatch for {name}: {output_meta} ({path})."
            )

    metadata = {
        "architecture": architecture,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_meta["epoch"],
        "checkpoint_global_step": checkpoint_meta["global_step"],
        "source": str(source_path),
        "gt": str(target_path),
        "fps": fps,
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame,
        "start_sec_actual": start_frame / fps,
        "end_sec_actual": end_frame / fps,
        "kernel_average": kernel_average,
        "kernel_offsets": list(offsets),
        "kernel_weights": list(weights),
        "kernel_average_alpha": average_alpha,
        "has_grid": has_grid,
        "video_writer": {
            "backend": args.writer_backend,
            "codec": (
                args.ffmpeg_codec
                if args.writer_backend == "ffmpeg"
                else args.codec
            ),
            "crf": args.ffmpeg_crf if args.writer_backend == "ffmpeg" else None,
            "preset": (
                args.ffmpeg_preset if args.writer_backend == "ffmpeg" else None
            ),
            "pixel_format": (
                args.ffmpeg_pix_fmt if args.writer_backend == "ffmpeg" else None
            ),
            "color_space": (
                args.ffmpeg_color_space
                if args.writer_backend == "ffmpeg"
                else None
            ),
            "color_range": (
                args.ffmpeg_color_range
                if args.writer_backend == "ffmpeg"
                else None
            ),
        },
        "elapsed_sec": elapsed,
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "audio": "not copied; all four output clips are video-only",
    }
    metadata_path = output_dir / "inference.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Completed {completed} frames in {elapsed:.3f}s.")
    for name, path in output_paths.items():
        print(f"  {name:>6}: {path}")
    print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()
