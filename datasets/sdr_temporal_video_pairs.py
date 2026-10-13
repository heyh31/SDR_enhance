from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from datasets.sdr_pairing import discover_video_pairs, VideoPair
from datasets.sdr_video_pairs import PairedSDRVideoDataset, _parse_hw, _video_metadata


def temporal_ranges(frame_count, fps, test_seconds=300, val_seconds=0,
                    gap_seconds=30, min_train_seconds=60, validation_uses_test_tail=True):
    """Half-open ranges; by default validation and test share the final tail."""
    values = (fps, test_seconds, val_seconds, gap_seconds, min_train_seconds)
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError("Split durations and FPS must be finite.")
    if min(fps, test_seconds, min_train_seconds) <= 0 or gap_seconds < 0 or val_seconds < 0:
        raise ValueError("FPS/train/test must be positive; gap/val durations may be zero.")
    if not validation_uses_test_tail and val_seconds <= 0:
        raise ValueError("Separate validation requires val_seconds > 0.")
    test_frames = math.ceil(test_seconds * fps)
    val_frames = math.ceil(val_seconds * fps)
    gap_frames = math.ceil(gap_seconds * fps)
    test_start = int(frame_count) - test_frames
    if validation_uses_test_tail:
        train_end = test_start - gap_frames
        if train_end < math.ceil(min_train_seconds * fps):
            raise ValueError(
                f"Video is too short ({frame_count / fps:.3f}s). Need at least "
                f"{(test_frames + gap_frames) / fps + min_train_seconds:.3f}s "
                "for train | gap | shared val/test tail. Adjust explicit durations."
            )
        return {
            "train": [0, train_end],
            "gap_train_test": [train_end, test_start],
            "val": [test_start, int(frame_count)],
            "test": [test_start, int(frame_count)],
        }
    val_end = test_start - gap_frames
    val_start = val_end - val_frames
    train_end = val_start - gap_frames
    if train_end < math.ceil(min_train_seconds * fps):
        raise ValueError(
            f"Video is too short ({frame_count / fps:.3f}s). Need at least "
            f"{(test_frames + val_frames + 2 * gap_frames) / fps + min_train_seconds:.3f}s "
            "for the requested train/val/test ranges. Adjust explicit durations; "
            "no video or test tail will be silently removed."
        )
    return {
        "train": [0, train_end],
        "gap_train_val": [train_end, val_start],
        "val": [val_start, val_end],
        "gap_val_test": [val_end, test_start],
        "test": [test_start, int(frame_count)],
    }


def evaluation_indices(start, end, fps, interval_seconds=10, maximum=180):
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("eval_interval_seconds must be positive and finite.")
    stride = max(1, round(fps * interval_seconds))
    # A midpoint in each time bucket, including a final partial bucket.
    indices = [min(top + stride // 2, end - 1) for top in range(start, end, stride)]
    if maximum > 0 and len(indices) > maximum:
        positions = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
        indices = [indices[int(i)] for i in positions]
    return indices


def build_temporal_manifest(cfg):
    if str(cfg.get("split_mode", "")) != "temporal_tail":
        raise ValueError("This dataset requires data.split_mode: temporal_tail.")
    pairs = discover_video_pairs(cfg.root, str(cfg.source_dir), str(cfg.target_dir),
                                 minimum_pairs=1)
    settings = {
        "test_seconds": float(cfg.get("test_tail_seconds", 300)),
        "val_seconds": float(cfg.get("val_tail_seconds", 0)),
        "gap_seconds": float(cfg.get("split_gap_seconds", 30)),
        "min_train_seconds": float(cfg.get("min_train_seconds", 60)),
        "validation_uses_test_tail": bool(cfg.get("validation_uses_test_tail", True)),
    }
    records = []
    for pair in pairs:
        source_meta = _video_metadata(pair.source)
        target_meta = _video_metadata(pair.target)
        for meta in (source_meta, target_meta):
            if not math.isfinite(meta["fps"]) or meta["fps"] <= 0:
                raise ValueError(f"Invalid FPS for {pair.name}: {meta}")
        if abs(source_meta["fps"] - target_meta["fps"]) > 1e-4:
            raise ValueError(f"FPS mismatch for {pair.name}; aligned constant-FPS pairs required.")
        if (source_meta["width"], source_meta["height"]) != (target_meta["width"], target_meta["height"]):
            raise ValueError(f"Geometry mismatch for {pair.name}.")
        if source_meta["frames"] != target_meta["frames"]:
            raise ValueError(
                f"Frame count mismatch for {pair.name}: source={source_meta['frames']}, "
                f"target={target_meta['frames']}. Align pairs before using a temporal split."
            )
        fps, count = source_meta["fps"], source_meta["frames"]
        try:
            ranges = temporal_ranges(count, fps, **settings)
        except ValueError as exc:
            raise ValueError(f"{pair.source.name}: {exc}") from exc
        eval_indices = {
            split: evaluation_indices(
                *ranges[split], fps, float(cfg.get("eval_interval_seconds", 10)),
                int(cfg.get("max_eval_frames_per_video", 180)),
            )
            for split in ("val", "test")
        }
        records.append({
            "name": pair.name,
            "source": str(pair.source.resolve()),
            "target": str(pair.target.resolve()),
            "fps": fps, "frame_count": count,
            "width": source_meta["width"], "height": source_meta["height"],
            "ranges": ranges,
            "ranges_seconds": {k: [a / fps, b / fps] for k, (a, b) in ranges.items()},
            "eval_indices": eval_indices,
        })
    payload = {
        "version": 1, "split_mode": "temporal_tail", "intervals": "[start,end)",
        "settings": settings, "videos": records,
        "evaluation_scope": (
            "shared validation/test tail of seen videos; test is reused for checkpoint selection"
            if settings["validation_uses_test_tail"] else
            "held-out time segments of seen videos; not unseen-video generalization"
        ),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return payload


def save_temporal_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_worker(worker_id):
    del worker_id
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)


class TemporalSDRVideoDataset(PairedSDRVideoDataset):
    """All videos participate; train is disjoint from both evaluation ranges.

    Training uses one uniform random frame per temporal bucket (no replacement
    within one traversal). Validation/test use fixed frame IDs. Only complete
    frames are resized, preserving the grid's full-frame coordinate system.
    """

    def __init__(self, manifest, cfg, split, cache_records=None, resized_train_cache=False):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        self._captures = OrderedDict()
        self.max_open_captures = int(cfg.get("max_open_captures", 4))
        if self.max_open_captures < 2:
            raise ValueError("max_open_captures must be >=2.")
        self.records = manifest["videos"]
        self.pairs = [VideoPair(r["name"], Path(r["source"]), Path(r["target"])) for r in self.records]
        self.frame_counts = [r["frame_count"] for r in self.records]
        self.split = split
        self.context_size = int(cfg.context_size)
        self.crop_size = 0
        self.spatial_mode = "resize"
        self.resize_size = _parse_hw(cfg.get("train_resize_size", [540, 960]), "train_resize_size")
        self.horizontal_flip = split == "train" and bool(cfg.get("horizontal_flip", True))
        if str(cfg.get("train_spatial_mode", "resize")) != "resize":
            raise ValueError("Temporal grid dataset uses full-frame resize; crops need coordinate remapping.")
        self.cached_train_resized = bool(resized_train_cache)
        if self.cached_train_resized and split != "train":
            raise ValueError("Resized train cache cannot be used for validation or test.")
        cache_option = "train_frame_cache_root" if self.cached_train_resized else "frame_cache_root"
        self.cache_root = Path(str(cfg.get(cache_option, ""))).expanduser()
        self.cached = cache_records
        self.samples = []
        self.buckets = []
        self.effective_eval_indices = {}
        slots = int(cfg.get("samples_per_video_per_epoch", 1000))
        if slots <= 0:
            raise ValueError("samples_per_video_per_epoch must be positive.")
        for pair_index, record in enumerate(self.records):
            start, end = record["ranges"][split]
            if self.cached is not None:
                available = sorted(
                    index for name, index in self.cached
                    if name == record["name"] and start <= index < end
                )
                if not available:
                    raise ValueError(f"Cache has no {split} frames for {record['name']}. Rebuild the temporal cache.")
            else:
                available = None
            if split == "train":
                size = len(available) if available is not None else end - start
                count = min(slots, size)
                for slot in range(count):
                    self.samples.append((pair_index, -1))
                    self.buckets.append((slot * size // count, (slot + 1) * size // count, available))
            else:
                if available is None:
                    indices = record["eval_indices"][split]
                else:
                    # Only exact predeclared eval IDs are accepted. Falling back
                    # to another cache frame would change the reported benchmark.
                    indices = record["eval_indices"][split]
                    missing = sorted(set(indices) - set(available))
                    if missing:
                        raise ValueError(f"Cache misses {split} frame IDs for {record['name']}: {missing[:8]}")
                self.effective_eval_indices[record["name"]] = indices
                self.samples.extend((pair_index, index) for index in indices)
                self.buckets.extend([None] * len(indices))
        if not self.samples:
            raise ValueError(f"No samples in {split}.")
        self.frame_sample_count = len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_captures"] = OrderedDict()
        return state

    def _capture(self, pair_index, domain):
        key = (pair_index, domain)
        if key in self._captures:
            capture = self._captures.pop(key)
            self._captures[key] = capture
            return capture
        while len(self._captures) >= self.max_open_captures:
            _, capture = self._captures.popitem(last=False)
            capture.release()
        capture = cv2.VideoCapture(str(getattr(self.pairs[pair_index], domain)))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open {domain} for {self.pairs[pair_index].name}")
        self._captures[key] = capture
        return capture

    def _read_frame(self, pair_index, domain, frame_index):
        if self.cached is not None:
            record = self.cached[(self.pairs[pair_index].name, frame_index)]
            path = self.cache_root / record[domain]
            if not self.cached_train_resized:
                path = path.resolve()
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3 or image.shape[2] != 3:
                raise RuntimeError(f"Cannot decode RGB cache frame: {path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        capture = self._capture(pair_index, domain)
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise RuntimeError(f"Frame seek failed: {self.pairs[pair_index].name}, {frame_index}")
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"Frame decode failed: {self.pairs[pair_index].name}, {frame_index}")
        position = capture.get(cv2.CAP_PROP_POS_FRAMES)
        if abs(position - (frame_index + 1)) > 0.5:
            raise RuntimeError(f"Decoder returned unexpected frame position {position}, requested {frame_index}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _transform_training_frames(self, sources, targets):
        if not self.cached_train_resized:
            return self._paired_transform_group(sources, targets)
        # The PNGs already match train_resize_size. Keep only the paired,
        # clip-consistent random flip; resizing them again wastes CPU time.
        flipped = self.horizontal_flip and random.random() < 0.5
        if flipped:
            sources = [np.flip(frame, axis=1) for frame in sources]
            targets = [np.flip(frame, axis=1) for frame in targets]
        return sources, targets, flipped

    def __getitem__(self, index):
        pair_index, frame_index = self.samples[index]
        record = self.records[pair_index]
        start, end = record["ranges"][self.split]
        if frame_index < 0:
            low, high, available = self.buckets[index]
            position = random.randrange(low, high)
            frame_index = available[position] if available is not None else start + position
        if not start <= frame_index < end:
            raise RuntimeError(f"Frame {frame_index} escapes {self.split} range [{start},{end}).")
        source = self._read_frame(pair_index, "source", frame_index)
        target = self._read_frame(pair_index, "target", frame_index)
        if source.shape != target.shape:
            raise ValueError(f"Decoded geometry mismatch for {record['name']} frame {frame_index}")
        expected_size = self.resize_size if self.cached_train_resized else (record["height"], record["width"])
        if source.shape[:2] != expected_size:
            raise ValueError(f"Frame geometry {source.shape[:2]} differs from expected {expected_size}.")
        context = self._make_context(source)
        sources, targets, flipped = self._transform_training_frames([source], [target])
        source, target = sources[0], targets[0]
        if flipped:
            context = np.flip(context, axis=1)
        return {
            "context": self._to_tensor(context), "source": self._to_tensor(source),
            "target": self._to_tensor(target), "video_name": record["name"],
            "frame_index": frame_index, "time_seconds": frame_index / record["fps"],
        }

    def __del__(self):
        for capture in getattr(self, "_captures", {}).values():
            capture.release()


class TemporalClipSDRVideoDataset(TemporalSDRVideoDataset):
    """Sample short clips, seeking once per paired stream and reading sequentially.

    A clip is selected uniformly from a temporal bucket of valid clip starts.
    Its frames stay inside the same train interval, and all frames in a clip
    receive the same horizontal flip. The frame count per epoch is unchanged
    when samples_per_video_per_epoch is divisible by train_clip_length.
    """

    def __init__(self, manifest, cfg, cache_records=None):
        clip_length = int(cfg.get("train_clip_length", 1))
        slots = int(cfg.get("samples_per_video_per_epoch", 1000))
        if clip_length <= 1:
            raise ValueError("TemporalClipSDRVideoDataset requires train_clip_length > 1.")
        if slots % clip_length:
            raise ValueError("samples_per_video_per_epoch must be divisible by train_clip_length.")
        if cache_records is not None:
            raise ValueError("Short-clip training requires video input; the current sparse frame cache has no guaranteed consecutive frames.")
        super().__init__(manifest, cfg, "train", cache_records=None)
        self.clip_length = clip_length
        self.samples = []
        self.buckets = []
        for pair_index, record in enumerate(self.records):
            start, end = record["ranges"]["train"]
            valid_starts = end - start - clip_length + 1
            count = min(slots // clip_length, (end - start) // clip_length)
            if count <= 0:
                raise ValueError(f"No {clip_length}-frame train clips available for {record['name']}.")
            for slot in range(count):
                self.samples.append((pair_index, -1))
                self.buckets.append((slot * valid_starts // count,
                                     (slot + 1) * valid_starts // count))
        self.frame_sample_count = len(self.samples) * clip_length

    def _read_clip(self, pair_index, domain, first_index):
        capture = self._capture(pair_index, domain)
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, first_index):
            raise RuntimeError(f"Frame seek failed: {self.pairs[pair_index].name}, {first_index}")
        frames = []
        for frame_index in range(first_index, first_index + self.clip_length):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Frame decode failed: {self.pairs[pair_index].name}, {frame_index}")
            position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if abs(position - (frame_index + 1)) > 0.5:
                raise RuntimeError(
                    f"Decoder returned unexpected frame position {position}, requested {frame_index}"
                )
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return frames

    def __getitem__(self, index):
        pair_index, _ = self.samples[index]
        record = self.records[pair_index]
        start, end = record["ranges"]["train"]
        low, high = self.buckets[index]
        first_index = start + random.randrange(low, high)
        if first_index + self.clip_length > end:
            raise RuntimeError(f"Clip escapes train range for {record['name']}.")
        sources = self._read_clip(pair_index, "source", first_index)
        targets = self._read_clip(pair_index, "target", first_index)
        expected = (record["height"], record["width"], 3)
        if any(source.shape != expected or target.shape != expected
               for source, target in zip(sources, targets)):
            raise ValueError(f"Decoded geometry mismatch for {record['name']} clip {first_index}.")
        contexts = [self._make_context(source) for source in sources]
        sources, targets, flipped = self._paired_transform_group(sources, targets)
        if flipped:
            contexts = [np.flip(context, axis=1) for context in contexts]
        return [
            {
                "context": self._to_tensor(contexts[offset]),
                "source": self._to_tensor(sources[offset]),
                "target": self._to_tensor(targets[offset]),
                "video_name": record["name"],
                "frame_index": first_index + offset,
                "time_seconds": (first_index + offset) / record["fps"],
            }
            for offset in range(self.clip_length)
        ]


class TemporalEvalClipSDRVideoDataset(TemporalSDRVideoDataset):
    """Read fixed, consecutive clips from a validation or test interval."""

    def __init__(self, manifest, cfg, split, clip_length):
        if split not in ("val", "test"):
            raise ValueError("Temporal evaluation clips are only for val/test.")
        if int(clip_length) <= 1:
            raise ValueError("Evaluation clip length must be greater than one.")
        super().__init__(manifest, cfg, split, cache_records=None)
        self.clip_length = int(clip_length)
        self.samples = []
        for pair_index, record in enumerate(self.records):
            start, end = record["ranges"][split]
            last_start = end - self.clip_length + 1
            if last_start <= start:
                raise ValueError(
                    f"{record['name']} {split} interval is shorter than "
                    f"the {self.clip_length}-frame evaluation clip."
                )
            starts = evaluation_indices(
                start, last_start, record["fps"],
                float(cfg.get("eval_interval_seconds", 10)),
                int(cfg.get("max_eval_frames_per_video", 180)),
            )
            self.samples.extend((pair_index, frame_index) for frame_index in starts)
        self.frame_sample_count = len(self.samples) * self.clip_length

    def _read_clip(self, pair_index, domain, first_index):
        capture = self._capture(pair_index, domain)
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, first_index):
            raise RuntimeError(f"Frame seek failed: {self.pairs[pair_index].name}, {first_index}")
        frames = []
        for frame_index in range(first_index, first_index + self.clip_length):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Frame decode failed: {self.pairs[pair_index].name}, {frame_index}")
            position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if abs(position - (frame_index + 1)) > 0.5:
                raise RuntimeError(
                    f"Decoder returned unexpected frame position {position}, requested {frame_index}"
                )
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return frames

    def __getitem__(self, index):
        pair_index, first_index = self.samples[index]
        record = self.records[pair_index]
        sources = self._read_clip(pair_index, "source", first_index)
        targets = self._read_clip(pair_index, "target", first_index)
        expected = (record["height"], record["width"], 3)
        if any(source.shape != expected or target.shape != expected
               for source, target in zip(sources, targets)):
            raise ValueError(f"Decoded geometry mismatch for {record['name']} clip {first_index}.")
        contexts = [self._make_context(source) for source in sources]
        return [
            {
                "context": self._to_tensor(contexts[offset]),
                "source": self._to_tensor(sources[offset]),
                "target": self._to_tensor(targets[offset]),
                "video_name": record["name"],
                "frame_index": first_index + offset,
                "time_seconds": (first_index + offset) / record["fps"],
            }
            for offset in range(self.clip_length)
        ]


def collate_temporal_clips(clips):
    """Flatten groups of frames back to the solver's usual [B,C,H,W] batch."""
    return default_collate([frame for clip in clips for frame in clip])


def collate_temporal_sequences(clips):
    """Keep frame order and stack clips as [B,N,...] tensors."""
    if not clips or not clips[0]:
        raise ValueError("Cannot collate an empty clip batch.")
    clip_length = len(clips[0])
    if any(len(clip) != clip_length for clip in clips):
        raise ValueError("All clips in a batch must have the same length.")
    time_batches = [default_collate([clip[t] for clip in clips]) for t in range(clip_length)]
    result = {}
    for key in time_batches[0]:
        values = [batch[key] for batch in time_batches]
        result[key] = torch.stack(values, dim=1) if torch.is_tensor(values[0]) else values
    return result


def read_temporal_cache(cfg, manifest):
    root = Path(str(cfg.frame_cache_root)).expanduser().resolve()
    saved = root / "temporal_splits.json"
    if not saved.is_file():
        raise ValueError("Use prepare_sdr_temporal_frames.py: cache needs temporal_splits.json.")
    fingerprint = json.loads(saved.read_text(encoding="utf-8")).get("fingerprint")
    if fingerprint != manifest["fingerprint"]:
        raise ValueError("Cache split fingerprint differs from current videos/config. Rebuild cache.")
    records = {}
    valid = {r["name"]: r for r in manifest["videos"]}
    with (root / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            name, index = record["video"], record["frame_index"]
            if name not in valid or not isinstance(index, int):
                raise ValueError(f"Unknown video or invalid frame index in cache: {record}")
            split = record.get("split")
            if split not in ("train", "val", "test"):
                raise ValueError(f"Invalid cache split: {record}")
            a, b = valid[name]["ranges"][split]
            if not a <= index < b:
                raise ValueError(f"Cache frame lies outside its declared temporal split: {record}")
            for domain in ("source", "target"):
                path = (root / record[domain]).resolve()
                if root not in path.parents or not path.is_file():
                    raise ValueError(f"Missing or unsafe cache image path: {path}")
            key = (name, index)
            if key in records:
                previous = records[key]
                shared_eval = (
                    manifest["settings"]["validation_uses_test_tail"]
                    and {previous["split"], split} == {"val", "test"}
                    and all(previous[d] == record[d] for d in ("source", "target"))
                )
                if not shared_eval:
                    raise ValueError(f"Duplicate cached frame: {key}")
                continue
            records[key] = record
    return records


def read_resized_train_cache(cfg, manifest):
    """Load a complete, train-only index; validation and test stay on video."""
    root = Path(str(cfg.train_frame_cache_root)).expanduser().resolve()
    metadata_path = root / "train_cache.json"
    index_path = root / "train_frames.jsonl"
    if not metadata_path.is_file() or not index_path.is_file():
        raise ValueError(f"Incomplete resized train cache at {root}; run prepare_sdr_resized_train_cache.py.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") == "resized_train_png_v1":
        extension = "png"
    elif (metadata.get("format") == "resized_train_image_v2"
          and metadata.get("image_format") == "jpeg"
          and 1 <= int(metadata.get("jpeg_quality", 0)) <= 100):
        extension = "jpg"
    else:
        raise ValueError("Unsupported resized train cache format.")
    if metadata.get("fingerprint") != manifest["fingerprint"]:
        raise ValueError("Resized train cache split differs from current videos/config. Rebuild the cache.")
    if metadata.get("resize_size") != list(_parse_hw(cfg.get("train_resize_size", [540, 960]),
                                                        "train_resize_size")):
        raise ValueError("Resized train cache geometry differs from data.train_resize_size.")
    if metadata.get("sampling") != "uniform_bucket_midpoint":
        raise ValueError("Unsupported resized train cache sampling method.")
    videos = {record["name"]: record for record in manifest["videos"]}
    for name in videos:
        directory = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        for domain in ("source", "target"):
            path = (root / directory / domain).resolve()
            if root not in path.parents:
                raise ValueError(f"Unsafe resized cache directory: {path}")
    records = {}
    with index_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            name, index = record.get("video"), record.get("frame_index")
            if name not in videos or type(index) is not int:
                raise ValueError(f"Invalid resized cache record at line {line_number}.")
            start, end = videos[name]["ranges"]["train"]
            if not start <= index < end or record.get("split") != "train":
                raise ValueError(f"Resized cache frame outside train split: {name}, {index}.")
            directory = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
            for domain in ("source", "target"):
                expected = f"{directory}/{domain}/{index:08d}.{extension}"
                if record.get(domain) != expected:
                    raise ValueError(f"Unsafe resized cache image path at line {line_number}.")
            key = (name, index)
            if key in records:
                raise ValueError(f"Duplicate resized cache frame: {key}.")
            records[key] = record
    if len(records) != int(metadata.get("frame_pair_count", -1)):
        raise ValueError("Resized cache index count differs from train_cache.json.")
    return records


def build_temporal_dataloaders(cfg, manifest=None):
    manifest = build_temporal_manifest(cfg) if manifest is None else manifest
    if int(cfg.get("eval_batch_size", 1)) != 1:
        raise ValueError("Use eval_batch_size=1 for full-resolution variable-size video frames.")
    if cfg.get("frame_cache_root") and cfg.get("train_frame_cache_root"):
        raise ValueError("Choose frame_cache_root or train_frame_cache_root, not both.")
    keep_clips = bool(cfg.get("preserve_clip_dimension", False))
    if keep_clips and (cfg.get("frame_cache_root") or cfg.get("train_frame_cache_root")):
        raise ValueError("Recurrent clip training requires online video input so every sampled clip is consecutive.")
    cached = read_temporal_cache(cfg, manifest) if cfg.get("frame_cache_root") else None
    resized_train = bool(cfg.get("train_frame_cache_root"))
    train_cached = read_resized_train_cache(cfg, manifest) if resized_train else cached
    configured_clip_length = int(cfg.get("train_clip_length", 1))
    if configured_clip_length < 1:
        raise ValueError("train_clip_length must be positive.")
    # Random seeks disappear when individual PNGs are available, so cached
    # training uses independent frames regardless of the online clip setting.
    clip_length = 1 if resized_train else configured_clip_length
    eval_clip_length = int(cfg.get("eval_clip_length", 1))
    if eval_clip_length < 1:
        raise ValueError("eval_clip_length must be positive.")
    datasets = {}
    for split in ("train", "val", "test"):
        split_cache = train_cached if split == "train" else (None if resized_train else cached)
        if split == "train" and clip_length > 1:
            dataset = TemporalClipSDRVideoDataset(manifest, cfg, split_cache)
        elif split != "train" and eval_clip_length > 1:
            if split_cache is not None:
                raise ValueError("Consecutive evaluation clips require original video input, not sparse frame caches.")
            dataset = TemporalEvalClipSDRVideoDataset(manifest, cfg, split, eval_clip_length)
        else:
            dataset = TemporalSDRVideoDataset(manifest, cfg, split, split_cache,
                                              resized_train_cache=resized_train and split == "train")
        datasets[split] = dataset
    workers = int(cfg.get("num_workers", 4))
    batch = int(cfg.get("batch_size", 8))
    if workers < 0 or batch <= 0:
        raise ValueError("num_workers must be >=0 and batch_size must be >0.")
    batch_size_is_clips = bool(cfg.get("batch_size_is_clips", False))
    if clip_length > 1 and not batch_size_is_clips and batch % clip_length:
        raise ValueError("batch_size must be divisible by train_clip_length.")
    common = dict(num_workers=workers, pin_memory=bool(cfg.get("pin_memory", True)),
                  persistent_workers=workers > 0 and bool(cfg.get("persistent_workers", True)),
                  worker_init_fn=seed_worker)
    loaders = {}
    for split, dataset in datasets.items():
        generator = torch.Generator().manual_seed(int(cfg.get("sampling_seed", 2025)))
        loaders[split] = DataLoader(
            dataset,
            batch_size=(batch if batch_size_is_clips else batch // clip_length)
            if split == "train" and clip_length > 1 else batch if split == "train" else 1,
            shuffle=split == "train", drop_last=False, generator=generator,
            collate_fn=(collate_temporal_sequences if keep_clips else collate_temporal_clips)
            if split == "train" and clip_length > 1 else
            collate_temporal_sequences if split != "train" and eval_clip_length > 1 else None,
            **common,
        )
    return loaders, manifest
