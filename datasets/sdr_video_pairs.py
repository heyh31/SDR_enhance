from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.sdr_pairing import (
    discover_video_pairs,
    split_video_pairs,
    write_split_manifest,
)


def _parse_hw(value, name):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        size = int(value)
        result = (size, size)
    else:
        values = list(value)
        if len(values) != 2:
            raise ValueError(f"{name} must be an integer or [height, width].")
        result = (int(values[0]), int(values[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} values must be greater than zero, got {result}.")
    return result


def _video_metadata(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    metadata = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if metadata["frames"] <= 0:
        raise ValueError(f"Video reports no frames: {path}")
    return metadata


class PairedSDRVideoDataset(Dataset):
    def __init__(
        self,
        pairs,
        split,
        context_size,
        crop_size,
        frame_stride,
        max_frames_per_video,
        random_samples_per_video=0,
        horizontal_flip=True,
        spatial_mode="crop",
        resize_size=None,
        temporal_num_frames=1,
        temporal_frame_stride=1,
        temporal_frame_offsets=None,
        temporal_supervise_all_frames=False,
    ):
        self.pairs = list(pairs)
        self.split = str(split)
        self.context_size = int(context_size)
        self.crop_size = int(crop_size)
        self.spatial_mode = str(spatial_mode).casefold()
        if self.spatial_mode not in {"crop", "resize"}:
            raise ValueError(
                f"spatial_mode must be 'crop' or 'resize', got {spatial_mode!r}."
            )
        self.resize_size = (
            _parse_hw(resize_size, "train_resize_size")
            if self.spatial_mode == "resize"
            else None
        )
        self.horizontal_flip = bool(horizontal_flip) and self.split == "train"
        self.temporal_num_frames = int(temporal_num_frames)
        self.temporal_frame_stride = int(temporal_frame_stride)
        if self.temporal_num_frames <= 0 or self.temporal_num_frames % 2 == 0:
            raise ValueError(
                "temporal_num_frames must be a positive odd integer, got "
                f"{self.temporal_num_frames}."
            )
        if self.temporal_frame_stride <= 0:
            raise ValueError("temporal_frame_stride must be greater than zero.")
        if temporal_frame_offsets is None:
            radius = self.temporal_num_frames // 2
            self.temporal_frame_offsets = [
                offset * self.temporal_frame_stride
                for offset in range(-radius, radius + 1)
            ]
        else:
            self.temporal_frame_offsets = [
                int(offset) for offset in temporal_frame_offsets
            ]
            if len(self.temporal_frame_offsets) != self.temporal_num_frames:
                raise ValueError(
                    "temporal_frame_offsets must contain temporal_num_frames "
                    f"values, got {len(self.temporal_frame_offsets)} and "
                    f"{self.temporal_num_frames}."
                )
            if len(set(self.temporal_frame_offsets)) != len(
                self.temporal_frame_offsets
            ):
                raise ValueError("temporal_frame_offsets cannot contain duplicates.")
            if 0 not in self.temporal_frame_offsets:
                raise ValueError("temporal_frame_offsets must contain zero.")
            if self.temporal_frame_offsets[self.temporal_num_frames // 2] != 0:
                raise ValueError(
                    "The middle value of temporal_frame_offsets must be zero."
                )
        self.temporal_center_position = self.temporal_frame_offsets.index(0)
        self.temporal_supervise_all_frames = bool(
            temporal_supervise_all_frames
        )
        self._captures = {}
        self.frame_counts = []
        self.samples = []

        for pair_index, pair in enumerate(self.pairs):
            source_meta = _video_metadata(pair.source)
            target_meta = _video_metadata(pair.target)
            if (source_meta["width"], source_meta["height"]) != (
                target_meta["width"],
                target_meta["height"],
            ):
                raise ValueError(
                    f"Geometry mismatch for {pair.name}: source="
                    f"{source_meta['width']}x{source_meta['height']}, target="
                    f"{target_meta['width']}x{target_meta['height']}"
                )
            if source_meta["fps"] and target_meta["fps"]:
                if abs(source_meta["fps"] - target_meta["fps"]) > 0.05:
                    raise ValueError(
                        f"FPS mismatch for {pair.name}: source={source_meta['fps']}, "
                        f"target={target_meta['fps']}"
                    )

            frame_count = min(source_meta["frames"], target_meta["frames"])
            self.frame_counts.append(frame_count)
            random_count = int(random_samples_per_video)
            if self.split == "train" and random_count > 0:
                # -1 is resolved to a uniformly random frame in __getitem__.
                # Repeated slots keep each video equally represented per epoch,
                # while every frame remains eligible across epochs.
                self.samples.extend((pair_index, -1) for _ in range(random_count))
            else:
                indices = list(range(0, frame_count, max(int(frame_stride), 1)))
                maximum = int(max_frames_per_video)
                if maximum > 0 and len(indices) > maximum:
                    positions = np.linspace(
                        0,
                        len(indices) - 1,
                        maximum,
                        dtype=np.int64,
                    )
                    indices = [indices[position] for position in positions]
                self.samples.extend((pair_index, frame_index) for frame_index in indices)

        if not self.samples:
            raise ValueError(f"No samples were created for the {self.split} split.")

    def __len__(self):
        return len(self.samples)

    def _capture(self, pair_index, domain):
        key = (pair_index, domain)
        capture = self._captures.get(key)
        if capture is None:
            path = getattr(self.pairs[pair_index], domain)
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"Unable to open video: {path}")
            self._captures[key] = capture
        return capture

    def _read_frame(self, pair_index, domain, frame_index):
        capture = self._capture(pair_index, domain)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            path = getattr(self.pairs[pair_index], domain)
            raise RuntimeError(f"Unable to decode frame {frame_index} from {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _paired_crop(self, source, target):
        sources, targets, flipped = self._paired_crop_group(
            [source],
            [target],
        )
        return sources[0], targets[0], flipped

    def _paired_crop_group(self, sources, targets):
        if len(sources) != len(targets) or not sources:
            raise ValueError("Paired frame groups must be non-empty and equal-sized.")
        height, width = sources[0].shape[:2]
        expected_shape = sources[0].shape
        if any(frame.shape != expected_shape for frame in sources + targets):
            raise ValueError("All frames in one temporal group must share geometry.")
        crop = self.crop_size
        if crop <= 0:
            return sources, targets, False

        if height < crop or width < crop:
            scale = max(crop / height, crop / width)
            size = (int(round(width * scale)), int(round(height * scale)))
            sources = [
                cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
                for frame in sources
            ]
            targets = [
                cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
                for frame in targets
            ]
            height, width = sources[0].shape[:2]

        if self.split == "train":
            top = random.randint(0, height - crop)
            left = random.randint(0, width - crop)
        else:
            top = (height - crop) // 2
            left = (width - crop) // 2
        sources = [
            frame[top : top + crop, left : left + crop]
            for frame in sources
        ]
        targets = [
            frame[top : top + crop, left : left + crop]
            for frame in targets
        ]

        flipped = self.horizontal_flip and random.random() < 0.5
        if flipped:
            sources = [np.flip(frame, axis=1) for frame in sources]
            targets = [np.flip(frame, axis=1) for frame in targets]
        return sources, targets, flipped

    def _paired_transform(self, source, target):
        sources, targets, flipped = self._paired_transform_group(
            [source],
            [target],
        )
        return sources[0], targets[0], flipped

    def _paired_transform_group(self, sources, targets):
        if self.split != "train" or self.spatial_mode == "crop":
            return self._paired_crop_group(sources, targets)

        output_height, output_width = self.resize_size
        input_height, input_width = sources[0].shape[:2]
        interpolation = (
            cv2.INTER_AREA
            if input_height > output_height or input_width > output_width
            else cv2.INTER_LANCZOS4
        )
        output_size = (output_width, output_height)
        sources = [
            cv2.resize(frame, output_size, interpolation=interpolation)
            for frame in sources
        ]
        targets = [
            cv2.resize(frame, output_size, interpolation=interpolation)
            for frame in targets
        ]
        flipped = self.horizontal_flip and random.random() < 0.5
        if flipped:
            sources = [np.flip(frame, axis=1) for frame in sources]
            targets = [np.flip(frame, axis=1) for frame in targets]
        return sources, targets, flipped

    @staticmethod
    def _to_tensor(image):
        dtype = image.dtype
        image = np.ascontiguousarray(image, dtype=np.float32)
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        if np.issubdtype(dtype, np.integer):
            tensor.div_(float(np.iinfo(dtype).max))
        return tensor

    def _make_context(self, source):
        """Resize the complete source frame used by the parameter encoder."""
        size = self.context_size
        if size <= 0:
            return source
        height, width = source.shape[:2]
        interpolation = (
            cv2.INTER_AREA if max(height, width) > size else cv2.INTER_LANCZOS4
        )
        return cv2.resize(source, (size, size), interpolation=interpolation)

    def _temporal_frame_indices(self, pair_index, center_index):
        frame_count = self.frame_counts[pair_index]
        return [
            min(
                max(center_index + offset, 0),
                frame_count - 1,
            )
            for offset in self.temporal_frame_offsets
        ]

    def __getitem__(self, index):
        pair_index, frame_index = self.samples[index]
        if frame_index < 0:
            frame_index = random.randrange(self.frame_counts[pair_index])
        pair = self.pairs[pair_index]
        temporal_indices = self._temporal_frame_indices(pair_index, frame_index)
        temporal_sources = [
            self._read_frame(pair_index, "source", temporal_index)
            for temporal_index in temporal_indices
        ]
        temporal_targets = [
            self._read_frame(pair_index, "target", temporal_index)
            for temporal_index in temporal_indices
        ]
        contexts = [self._make_context(frame) for frame in temporal_sources]
        temporal_sources, temporal_targets, flipped = self._paired_transform_group(
            temporal_sources,
            temporal_targets,
        )
        if flipped:
            contexts = [np.flip(context, axis=1) for context in contexts]
        context_tensors = [self._to_tensor(context) for context in contexts]
        context = (
            context_tensors[0]
            if self.temporal_num_frames == 1
            else torch.stack(context_tensors, dim=0)
        )
        source_tensors = [self._to_tensor(frame) for frame in temporal_sources]
        target_tensors = [self._to_tensor(frame) for frame in temporal_targets]
        if self.temporal_supervise_all_frames:
            source = torch.stack(source_tensors, dim=0)
            target = torch.stack(target_tensors, dim=0)
        else:
            source = source_tensors[self.temporal_center_position]
            target = target_tensors[self.temporal_center_position]
        return {
            "context": context,
            "source": source,
            "target": target,
            "video_name": pair.name,
            "frame_index": frame_index,
            "temporal_frame_indices": torch.tensor(
                temporal_indices,
                dtype=torch.int64,
            ),
        }

    def __del__(self):
        for capture in self._captures.values():
            capture.release()


class CachedSDRFrameDataset(PairedSDRVideoDataset):
    """Read lossless, full-resolution pairs produced by prepare_sdr_frames.py."""

    def __init__(
        self,
        cache_root,
        records,
        split,
        context_size,
        crop_size,
        horizontal_flip=True,
        spatial_mode="crop",
        resize_size=None,
        temporal_num_frames=1,
        temporal_frame_stride=1,
        temporal_frame_offsets=None,
        temporal_supervise_all_frames=False,
    ):
        # Do not call the video dataset constructor: no VideoCapture is needed.
        self.cache_root = Path(cache_root).expanduser()
        self.split = str(split)
        self.context_size = int(context_size)
        self.crop_size = int(crop_size)
        self.spatial_mode = str(spatial_mode).casefold()
        if self.spatial_mode not in {"crop", "resize"}:
            raise ValueError(
                f"spatial_mode must be 'crop' or 'resize', got {spatial_mode!r}."
            )
        self.resize_size = (
            _parse_hw(resize_size, "train_resize_size")
            if self.spatial_mode == "resize"
            else None
        )
        self.horizontal_flip = bool(horizontal_flip) and self.split == "train"
        self.temporal_num_frames = int(temporal_num_frames)
        self.temporal_frame_stride = int(temporal_frame_stride)
        if self.temporal_num_frames != 1:
            raise ValueError(
                "Temporal context training requires direct video input because "
                "the current sparse frame cache does not guarantee adjacent "
                "frames. Set data.frame_cache_root='' or use "
                "temporal_num_frames=1."
            )
        if temporal_supervise_all_frames:
            raise ValueError(
                "temporal_supervise_all_frames is unavailable for sparse cache."
            )
        self._captures = {}
        self.samples = list(records)
        self.video_names = sorted({record["video"] for record in self.samples})
        if not self.samples:
            raise ValueError(
                f"No {self.split!r} records were found in "
                f"{self.cache_root / 'manifest.jsonl'}"
            )

    @staticmethod
    def _decode_image(path):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Unable to decode cached frame: {path}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected a three-channel RGB PNG, got shape={image.shape}: {path}"
            )
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        record = self.samples[index]
        source_path = self.cache_root / record["source"]
        target_path = self.cache_root / record["target"]
        source = self._decode_image(source_path)
        target = self._decode_image(target_path)
        if source.shape != target.shape:
            raise ValueError(
                f"Cached pair geometry mismatch: source={source.shape}, "
                f"target={target.shape}, frame={record['frame_index']}"
            )

        # Context always sees the complete source frame. Supervision applies the
        # selected aligned crop/resize transform to source and target together.
        context = self._make_context(source)
        source, target, flipped = self._paired_transform(source, target)
        if flipped:
            context = np.flip(context, axis=1)
        return {
            "context": self._to_tensor(context),
            "source": self._to_tensor(source),
            "target": self._to_tensor(target),
            "video_name": record["video"],
            "frame_index": int(record["frame_index"]),
        }


def _read_cached_frame_manifest(cache_root):
    cache_root = Path(cache_root).expanduser()
    manifest_path = cache_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Frame-cache manifest does not exist: {manifest_path}. "
            "Run prepare_sdr_frames.py first."
        )
    records = {"train": [], "val": [], "test": []}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            split = record.get("split")
            if split not in records:
                raise ValueError(
                    f"Invalid split {split!r} on line {line_number} of {manifest_path}"
                )
            for key in ("video", "frame_index", "source", "target"):
                if key not in record:
                    raise ValueError(
                        f"Missing {key!r} on line {line_number} of {manifest_path}"
                    )
            records[split].append(record)
    return cache_root, records


def build_sdr_dataloaders(cfg):
    train_spatial_mode = str(cfg.get("train_spatial_mode", "crop")).casefold()
    train_resize_size = cfg.get("train_resize_size", [1024, 1024])
    temporal_num_frames = int(cfg.get("temporal_num_frames", 1))
    temporal_frame_stride = int(cfg.get("temporal_frame_stride", 1))
    temporal_frame_offsets = cfg.get("temporal_frame_offsets")
    temporal_supervise_all_frames = bool(
        cfg.get("temporal_supervise_all_frames", False)
    )
    cache_value = cfg.get("frame_cache_root", "")
    cache_root = "" if cache_value is None else str(cache_value).strip()
    if cache_root:
        cache_root, splits = _read_cached_frame_manifest(cache_root)
        datasets = {
            split: CachedSDRFrameDataset(
                cache_root=cache_root,
                records=splits[split],
                split=split,
                context_size=int(cfg.context_size),
                crop_size=(
                    int(cfg.crop_size)
                    if split == "train"
                    else int(cfg.get("eval_crop_size", 0))
                ),
                horizontal_flip=bool(cfg.get("horizontal_flip", True)),
                spatial_mode=(train_spatial_mode if split == "train" else "crop"),
                resize_size=(train_resize_size if split == "train" else None),
                temporal_num_frames=temporal_num_frames,
                temporal_frame_stride=temporal_frame_stride,
                temporal_frame_offsets=temporal_frame_offsets,
                temporal_supervise_all_frames=temporal_supervise_all_frames,
            )
            for split in ("train", "val", "test")
        }
    else:
        pairs = discover_video_pairs(
            root=cfg.root,
            source_dir=str(cfg.source_dir),
            target_dir=str(cfg.target_dir),
        )
        splits = split_video_pairs(
            pairs,
            seed=int(cfg.split_seed),
            val_video=cfg.get("val_video"),
            test_video=cfg.get("test_video"),
        )

        datasets = {
            "train": PairedSDRVideoDataset(
                splits["train"],
                split="train",
                context_size=int(cfg.context_size),
                crop_size=int(cfg.crop_size),
                frame_stride=1,
                max_frames_per_video=0,
                random_samples_per_video=int(cfg.samples_per_video_per_epoch),
                horizontal_flip=bool(cfg.get("horizontal_flip", True)),
                spatial_mode=train_spatial_mode,
                resize_size=train_resize_size,
                temporal_num_frames=temporal_num_frames,
                temporal_frame_stride=temporal_frame_stride,
                temporal_frame_offsets=temporal_frame_offsets,
                temporal_supervise_all_frames=temporal_supervise_all_frames,
            ),
            "val": PairedSDRVideoDataset(
                splits["val"],
                split="val",
                context_size=int(cfg.context_size),
                crop_size=int(cfg.get("eval_crop_size", 0)),
                frame_stride=int(cfg.eval_frame_stride),
                max_frames_per_video=int(cfg.max_eval_frames_per_video),
                horizontal_flip=False,
                spatial_mode="crop",
                temporal_num_frames=temporal_num_frames,
                temporal_frame_stride=temporal_frame_stride,
                temporal_frame_offsets=temporal_frame_offsets,
                temporal_supervise_all_frames=temporal_supervise_all_frames,
            ),
            "test": PairedSDRVideoDataset(
                splits["test"],
                split="test",
                context_size=int(cfg.context_size),
                crop_size=int(cfg.get("eval_crop_size", 0)),
                frame_stride=int(cfg.eval_frame_stride),
                max_frames_per_video=int(cfg.max_eval_frames_per_video),
                horizontal_flip=False,
                spatial_mode="crop",
                temporal_num_frames=temporal_num_frames,
                temporal_frame_stride=temporal_frame_stride,
                temporal_frame_offsets=temporal_frame_offsets,
                temporal_supervise_all_frames=temporal_supervise_all_frames,
            ),
        }

    num_workers = int(cfg.num_workers)
    loader_common = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg.get("pin_memory", True)),
        "persistent_workers": bool(cfg.get("persistent_workers", True))
        and num_workers > 0,
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(cfg.batch_size),
            shuffle=True,
            drop_last=True,
            **loader_common,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=int(cfg.get("eval_batch_size", 1)),
            shuffle=False,
            drop_last=False,
            **loader_common,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=int(cfg.get("eval_batch_size", 1)),
            shuffle=False,
            drop_last=False,
            **loader_common,
        ),
    }
    return loaders, splits
