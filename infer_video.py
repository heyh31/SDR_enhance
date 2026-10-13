#!/usr/bin/env python3
"""Run a shared-ViT SDR model on a single input video interval (no GT required)."""
from __future__ import annotations

import argparse
from datetime import datetime
from fractions import Fraction
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from infer_sdr_video import build_model, load_checkpoint, resolve_checkpoint


def parse_time(value):
    """Parse seconds, MM:SS, or HH:MM:SS (end time is exclusive)."""
    try:
        fields = str(value).strip().split(":")
        if not 1 <= len(fields) <= 3:
            raise ValueError
        parts = [Fraction(field) for field in fields]
        if any(part < 0 for part in parts):
            raise ValueError
        if len(parts) > 1 and (
            any(part.denominator != 1 for part in parts[:-1])
            or any(part >= 60 for part in parts[1:])
        ):
            raise ValueError
        result = Fraction(0)
        for part in parts:
            result = result * 60 + part
        return result
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid time {value!r}; use seconds or HH:MM:SS[.sss]."
        ) from exc


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Infer a video from one SDR input without requiring a paired GT video. "
            "Writes source (optional), Global, and Global+Local clips."
        )
    )
    parser.add_argument("--config", required=True, help="Project model/config YAML.")
    parser.add_argument("--checkpoint", default="auto", help="Checkpoint path, or auto from config output_dir/exp_name.")
    parser.add_argument("--input", "--source", required=True, help="Input SDR video.")
    parser.add_argument("--start", "--start-sec", type=parse_time, default=Fraction(0), help="Inclusive start time (seconds, MM:SS, or HH:MM:SS).")
    parser.add_argument("--end", "--end-sec", type=parse_time, required=True, help="Exclusive end time (seconds, MM:SS, or HH:MM:SS).")
    parser.add_argument("--output", help="Output directory; defaults to ./video_inference_outputs/<run>.")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--save-source", dest="save_source", action="store_true", default=True, help="Save original clip as source.mp4 (default).")
    source_group.add_argument("--no-save-source", dest="save_source", action="store_false", help="Do not save the original clip.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="FFmpeg executable name or path.")
    parser.add_argument("--crf", type=int, default=12, help="H.264 quality, 0..51; lower is better.")
    parser.add_argument("--preset", default="fast", choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"))
    parser.add_argument("--pixel-format", choices=("yuv420p", "yuv444p"), default="yuv420p")
    return parser


def resolve_ffmpeg(value):
    resolved = shutil.which(str(value))
    if resolved is None and Path(value).expanduser().is_file():
        resolved = str(Path(value).expanduser().resolve())
    if resolved is None:
        raise FileNotFoundError("FFmpeg was not found. Install it/add it to PATH, or pass --ffmpeg-bin.")
    return resolved


def video_metadata(capture, path):
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    count, fps, width, height = [capture.get(prop) for prop in (
        cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FPS,
        cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
    )]
    if not all(math.isfinite(value) and value > 0 for value in (count, fps, width, height)):
        raise ValueError(f"Invalid video metadata for {path}")
    return {
        "frame_count": int(count),
        "fps": Fraction(fps).limit_denominator(100000),
        "width": int(width),
        "height": int(height),
    }


class FFmpegWriter:
    """Write RGB frames with a consistent explicit RGB-to-YUV conversion."""

    def __init__(self, path, meta, executable, crf, preset, pixel_format):
        self.path = Path(path)
        self.closed = False
        self.log = tempfile.TemporaryFile()
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size",
            f"{meta['width']}x{meta['height']}", "-framerate", str(meta["fps"]),
            "-i", "pipe:0", "-an", "-vf",
            "scale=in_range=pc:out_range=tv:out_color_matrix=bt709",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-threads", "2", "-pix_fmt", pixel_format,
            "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv", "-movflags",
            "+faststart", str(self.path),
        ]
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self.log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except BaseException:
            self.log.close()
            raise

    def _error(self):
        self.log.seek(0)
        return self.log.read().decode("utf-8", errors="replace")[-8000:]

    def write(self, rgb):
        try:
            self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"FFmpeg failed writing {self.path.name}:\n{self._error()}") from exc

    def close(self, abort=False):
        if self.closed:
            return
        self.closed = True
        try:
            if abort and self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.stdin.close()
            except (OSError, BrokenPipeError):
                pass
            code = self.process.wait(timeout=30)
            if not abort and code != 0:
                raise RuntimeError(f"FFmpeg exited with {code}: {self.path.name}\n{self._error()}")
        finally:
            self.log.close()


def frame_tensor(bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    array = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def to_rgb8(tensor):
    frame = tensor[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.rint(frame * 255.0).astype(np.uint8)


@torch.inference_mode()
def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.end <= args.start:
        raise ValueError("Need 0 <= --start < --end.")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51.")
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = OmegaConf.load(config_path)
    # Interpret relative run/checkpoint paths from the config directory.
    for key in ("output_dir",):
        value = Path(str(cfg.get(key, "./runs")).replace("\\", "/"))
        if not value.is_absolute():
            cfg[key] = str((config_path.parent / value).resolve())
    checkpoint = resolve_checkpoint(cfg, args.checkpoint)
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(args.output).expanduser().resolve() if args.output else (
        Path.cwd() / "video_inference_outputs" /
        f"{source.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if output == source or output in source.parents:
        raise ValueError("Output directory must not contain/overwrite the source video.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty or new: {output}")

    executable = resolve_ffmpeg(args.ffmpeg_bin)
    capture = cv2.VideoCapture(str(source))
    meta = video_metadata(capture, source)
    fps = meta["fps"]
    first = math.ceil(args.start * fps)
    stop = math.ceil(args.end * fps)
    duration = Fraction(meta["frame_count"], 1) / fps
    if args.start < 0 or args.end > duration or first >= stop or stop > meta["frame_count"]:
        capture.release()
        raise ValueError(f"Requested [{float(args.start):.3f}, {float(args.end):.3f}) interval is outside video duration {float(duration):.3f}s.")
    if args.pixel_format == "yuv420p" and (meta["width"] % 2 or meta["height"] % 2):
        capture.release()
        raise ValueError("Odd video dimensions require --pixel-format yuv444p; no resize/crop is applied.")
    if first and not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
        capture.release()
        raise RuntimeError(f"Could not seek to frame {first}.")

    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    architecture, model = build_model(cfg)
    if architecture not in {"colorfm_l_spatial_grid_shared_vit", "colorfm_l_spatial_grid_shared_vit_recurrent"}:
        capture.release()
        raise ValueError(f"This CLI supports shared-ViT architectures, got {architecture!r}.")
    checkpoint_meta = load_checkpoint(model, checkpoint)
    model.to(device).eval()
    recurrent = bool(getattr(model, "supports_temporal_recurrence", False))
    if recurrent:
        model.reset_temporal_state()

    output.mkdir(parents=True, exist_ok=True)
    paths = {"global": output / "global.mp4", "global_local": output / "global_local.mp4"}
    if args.save_source:
        paths["source"] = output / "source.mp4"
    writers = {}
    try:
        for stage, path in paths.items():
            writers[stage] = FFmpegWriter(path, meta, executable, args.crf, args.preset, args.pixel_format)
        started = time.perf_counter()
        context_size = int(cfg.data.get("context_size", 256))
        print(f"Architecture: {architecture} | device={device}")
        print(f"Checkpoint: {checkpoint} | epoch={checkpoint_meta['epoch']} step={checkpoint_meta['global_step']}")
        print(f"Input: {source}")
        print(f"Interval: [{first / float(fps):.3f}s, {stop / float(fps):.3f}s) | frames={stop-first} | fps={float(fps):.6f}")
        print(f"Temporal mode: {'recurrent feature state across selected frames' if recurrent else 'independent frame inference; no kernel averaging'}")
        print(f"Output: {output}", flush=True)
        for processed, index in enumerate(range(first, stop), start=1):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index}; stopping rather than silently skipping it.")
            if bgr.shape[:2] != (meta["height"], meta["width"]):
                raise RuntimeError(f"Frame geometry changed at frame {index}.")
            source_tensor = frame_tensor(bgr, device)
            context = F.interpolate(source_tensor, size=(context_size, context_size), mode="bilinear", align_corners=False)
            if recurrent:
                parameters = model.predict_single_parameters(context, frame_index=index)
            else:
                parameters = model.predict_single_parameters(context)
            global_tensor, _, final_tensor = model.forward_single_stages(source_tensor, parameters=parameters)
            writers["global"].write(to_rgb8(global_tensor))
            writers["global_local"].write(to_rgb8(final_tensor))
            if args.save_source:
                writers["source"].write(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if processed == 1 or processed == stop-first or processed % max(1, round(float(fps) * 5)) == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(f"Frames: {processed}/{stop-first} | {processed/elapsed:.2f} fps", flush=True)
        for writer in writers.values():
            writer.close()
        print("Saved:")
        for path in paths.values():
            print(f"  {path}")
    finally:
        capture.release()
        for writer in writers.values():
            if not writer.closed:
                writer.close(abort=True)


if __name__ == "__main__":
    main()
