from __future__ import annotations

import difflib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


@dataclass(frozen=True)
class VideoPair:
    name: str
    source: Path
    target: Path


def _normalise_name(path: Path) -> str:
    name = path.stem.casefold().strip()
    # The server files shown by the user use 709_ and SDR增强_ as the domain
    # prefixes. Remove only leading domain markers and retain the actual title.
    name = re.sub(r"^(?:sdr[\s_-]*增强|sdr增强|709)[\s_-]*", "", name)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", name)


def _title_key(path: Path) -> str:
    """Extract the stable title prefix used by the shown server filenames."""
    name = path.stem.casefold().strip()
    name = re.sub(r"^(?:sdr[\s_-]*增强|sdr增强|709)[\s_-]*", "", name)
    title = re.split(r"[_\s]+", name, maxsplit=1)[0]
    # One source filename uses 709_增强权利的游戏 while its paired target uses
    # SDR增强_权利的游戏.
    title = re.sub(r"^增强", "", title)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", title)


def discover_video_pairs(root: str | Path, source_dir: str, target_dir: str, minimum_pairs=3):
    root = Path(root).expanduser()
    source_root = root / source_dir
    target_root = root / target_dir
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source video directory does not exist: {source_root}")
    if not target_root.is_dir():
        raise FileNotFoundError(f"Target video directory does not exist: {target_root}")

    source_files = sorted(
        path for path in source_root.iterdir() if path.suffix.casefold() in VIDEO_EXTENSIONS
    )
    target_files = sorted(
        path for path in target_root.iterdir() if path.suffix.casefold() in VIDEO_EXTENSIONS
    )
    if not source_files or not target_files:
        raise ValueError(
            f"No videos found under {source_root} and {target_root}. "
            f"Supported extensions: {sorted(VIDEO_EXTENSIONS)}"
        )

    target_by_key = {}
    target_by_title = {}
    for target in target_files:
        key = _normalise_name(target)
        if key in target_by_key:
            raise ValueError(
                "Ambiguous target video names after prefix normalization: "
                f"{target_by_key[key].name!r} and {target.name!r}"
            )
        target_by_key[key] = target
        title = _title_key(target)
        if title in target_by_title:
            raise ValueError(
                f"Ambiguous target title prefixes: {target_by_title[title].name!r} "
                f"and {target.name!r}"
            )
        target_by_title[title] = target

    unmatched_targets = set(target_files)
    pairs = []
    missing = []
    for source in source_files:
        key = _normalise_name(source)
        target = target_by_key.get(key)
        if target not in unmatched_targets:
            target = None
        if target is None:
            title_match = target_by_title.get(_title_key(source))
            if title_match in unmatched_targets:
                target = title_match
        if target is None:
            scored = sorted(
                (
                    difflib.SequenceMatcher(None, key, _normalise_name(item)).ratio(),
                    item,
                )
                for item in unmatched_targets
            )
            if scored and scored[-1][0] >= 0.80:
                target = scored[-1][1]
            else:
                missing.append(source.name)
                continue

        unmatched_targets.discard(target)
        pairs.append(VideoPair(name=key, source=source, target=target))

    if missing or unmatched_targets:
        details = []
        if missing:
            details.append(f"sources without targets={missing}")
        if unmatched_targets:
            details.append(
                "targets without sources="
                + repr(sorted(path.name for path in unmatched_targets))
            )
        raise ValueError(
            "Could not establish a one-to-one source/target video pairing: "
            + "; ".join(details)
        )
    if len(pairs) < int(minimum_pairs):
        raise ValueError(f"At least {minimum_pairs} video pairs are required.")
    return sorted(pairs, key=lambda pair: pair.name)


def _select_named_pair(pairs, requested_name, split_name):
    requested_key = _normalise_name(Path(requested_name))
    matches = [
        pair
        for pair in pairs
        if pair.name == requested_key
        or requested_key in pair.name
        or pair.name in requested_key
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{split_name}_video={requested_name!r} matched {len(matches)} pairs; "
            "use a unique full or partial video name."
        )
    return matches[0]


def split_video_pairs(pairs, seed, val_video=None, test_video=None):
    """Hold out exactly one validation and one test video pair."""
    pairs = list(pairs)
    shuffled = pairs.copy()
    random.Random(int(seed)).shuffle(shuffled)

    val_pair = (
        _select_named_pair(pairs, str(val_video), "val")
        if val_video
        else shuffled[0]
    )
    remaining = [pair for pair in shuffled if pair != val_pair]
    test_pair = (
        _select_named_pair(pairs, str(test_video), "test")
        if test_video
        else remaining[0]
    )
    if test_pair == val_pair:
        raise ValueError("Validation and test videos must be different.")

    train_pairs = [pair for pair in pairs if pair not in {val_pair, test_pair}]
    return {"train": train_pairs, "val": [val_pair], "test": [test_pair]}


def write_split_manifest(splits, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        split: [
            {
                "name": pair.name,
                "source": str(pair.source),
                "target": str(pair.target),
            }
            for pair in pairs
        ]
        for split, pairs in splits.items()
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
