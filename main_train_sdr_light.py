#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from datasets.sdr_temporal_video_pairs import (
    build_temporal_manifest, build_temporal_dataloaders, save_temporal_manifest,
    TemporalClipSDRVideoDataset,
)
from models.colorfm_l_spatial_grid_shared_vit import parameter_counts
from solvers.colorfm_l_spatial_grid_light_solver import ColorFM_L as Solver


def build_model(cfg):
    architecture = str(cfg.model.architecture)
    if architecture == "colorfm_l_spatial_grid_shared_vit":
        from models.colorfm_l_spatial_grid_shared_vit import ColorFM_L
    elif architecture == "colorfm_l_spatial_grid_shared_vit_recurrent":
        from models.colorfm_l_spatial_grid_shared_vit_recurrent import ColorFM_L
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    model = ColorFM_L(cfg)
    if bool(cfg.model.get("freeze_encoder", False)):
        model.encoder.requires_grad_(False)
    return model


def resolve_checkpoint(value, directory):
    if str(value).lower() == "none":
        return None
    if str(value).lower() == "auto":
        path = directory / "last.ckpt"
        return str(path) if path.is_file() else None
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def check_split_checkpoint(path, fingerprint):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("sdr_split_fingerprint") != fingerprint:
        raise ValueError("Resume checkpoint has a different/unknown temporal split. Start a new experiment.")
    return payload


def build_checkpoint_callbacks(cfg, checkpoints):
    best_metric = str(cfg.solver.get("best_checkpoint_metric", "val_loss"))
    best_mode = str(cfg.solver.get("best_checkpoint_mode", "min"))
    best_callback = ModelCheckpoint(
        dirpath=checkpoints,
        filename=f"best-epoch={{epoch:03d}}-{best_metric}={{{best_metric}:.6f}}",
        monitor=best_metric, mode=best_mode,
        save_top_k=int(cfg.solver.get("save_top_k", 3)),
        save_last=False, auto_insert_metric_name=False,
    )
    # In Lightning 2.6, save_last=True on a monitored top-k callback only
    # updates last.ckpt when that epoch enters the top-k. Keep recovery
    # checkpoints independent of metric-based selection.
    last_callback = ModelCheckpoint(
        dirpath=checkpoints, filename="last", monitor=None, save_top_k=1,
        save_last=False, every_n_epochs=1, save_on_train_epoch_end=True,
        enable_version_counter=False, auto_insert_metric_name=False,
    )
    return best_callback, last_callback


def main(default_config=None, solver_type=Solver):
    parser = argparse.ArgumentParser(description="Train Global + Grid using disjoint tails of every paired video.")
    parser.add_argument("--config", default=str(default_config or Path(__file__).resolve().parent / "configs/colorfm_l_spatial_grid_light_sdr.yaml"))
    parser.add_argument("--data-root")
    parser.add_argument("--frame-cache-root")
    parser.add_argument("--train-frame-cache-root", help="Train-only resized PNG cache; val/test stay on original videos.")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", default="auto", help="Checkpoint, auto (last.ckpt), or none.")
    parser.add_argument("--test-only", action="store_true", help="Evaluate reserved test tails using --resume.")
    parser.add_argument("--test-after-fit", action="store_true", help="Explicitly test the best validation checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Print model sizes and split/sample manifest; do not train.")
    parser.add_argument("--model-only", action="store_true", help="Print parameter counts without accessing videos.")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    if args.frame_cache_root is not None:
        cfg.data.frame_cache_root = args.frame_cache_root
    if args.train_frame_cache_root is not None:
        cfg.data.train_frame_cache_root = args.train_frame_cache_root
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if cfg.model.get("pretrained"):
        raise ValueError("This new architecture starts from scratch. Use --resume only with its matching training checkpoint.")
    cfg.data.sampling_seed = int(cfg.data.get("sampling_seed", cfg.seed))
    L.seed_everything(int(cfg.seed), workers=True)
    torch.set_float32_matmul_precision(str(cfg.trainer.get("matmul_precision", "high")))
    model = build_model(cfg)
    counts = parameter_counts(model)
    print(f"Architecture: {cfg.model.architecture}")
    print(json.dumps(counts, indent=2))
    if args.model_only:
        return

    manifest = build_temporal_manifest(cfg.data)
    output = Path(cfg.output_dir).expanduser() / str(cfg.exp_name)
    checkpoints = output / "checkpoints"
    resume = resolve_checkpoint(args.resume, checkpoints)
    if args.test_only and resume is None:
        raise ValueError("--test-only requires a trained --resume checkpoint.")
    if resume:
        check_split_checkpoint(resume, manifest["fingerprint"])
    manifest_path = output / "temporal_splits.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Experiment directory already records another split. Change exp_name/output_dir.")
    loaders, manifest = build_temporal_dataloaders(cfg.data, manifest)
    print("Half-open frame intervals [start, end); source and GT use the SAME frame ID.")
    for record in manifest["videos"]:
        print(f"Video: {record['name']}")
        for split, span in record["ranges_seconds"].items():
            print(f"  {split}: {span[0]:.3f}s .. {span[1]:.3f}s | frames={record['ranges'][split]}")
    print("Frames per epoch: " + ", ".join(
        f"{s}={getattr(loader.dataset, 'frame_sample_count', len(loader.dataset))}"
        for s, loader in loaders.items()
    ))
    if isinstance(loaders["train"].dataset, TemporalClipSDRVideoDataset):
        groups_per_batch = (int(cfg.data.batch_size)
                            if cfg.data.get("batch_size_is_clips", False)
                            else int(cfg.data.batch_size) // int(cfg.data.train_clip_length))
        print(f"Train clip groups: {len(loaders['train'].dataset)} "
              f"x {int(cfg.data.train_clip_length)} frames; "
              f"groups per batch: {groups_per_batch}")
    elif cfg.data.get("train_frame_cache_root"):
        print(f"Train source: resized PNG cache at {cfg.data.train_frame_cache_root}; "
              "validation/test source: original videos")
    print("Evaluation scope: " + manifest["evaluation_scope"])
    print(f"Output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    save_temporal_manifest(manifest, manifest_path)
    # Keep the original training config when running test-only or dry-run later.
    if not args.test_only and not args.dry_run:
        OmegaConf.save(cfg, output / "config.yaml")
    (output / "parameter_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    if args.dry_run:
        return

    solver = solver_type(cfg, model)
    solver.split_fingerprint = manifest["fingerprint"]
    best_callback, last_callback = build_checkpoint_callbacks(cfg, checkpoints)
    best_metric, best_mode = best_callback.monitor, best_callback.mode
    callbacks = [best_callback, last_callback, LearningRateMonitor(logging_interval="epoch")]
    patience = int(cfg.solver.get("early_stopping_patience", 8))
    if patience > 0:
        callbacks.append(EarlyStopping(
            monitor=best_metric, mode=best_mode, patience=patience,
            min_delta=float(cfg.solver.get("early_stopping_min_delta", 0.0001)),
        ))
    devices = cfg.trainer.get("devices", "auto")
    if OmegaConf.is_config(devices):
        devices = OmegaConf.to_container(devices)
    if args.test_only and (isinstance(devices, list) and len(devices) != 1
                           or isinstance(devices, int) and devices != 1 or devices == "auto"):
        raise ValueError("For exact test means set trainer.devices to 1 or a single GPU [0]; DDP pads eval samples.")
    if args.test_after_fit and (devices == "auto" or isinstance(devices, list) and len(devices) != 1
                               or isinstance(devices, int) and devices != 1):
        raise ValueError("Run --test-only on one device after multi-GPU training to avoid padded test samples.")
    trainer = L.Trainer(
        accelerator=str(cfg.trainer.get("accelerator", "auto")), devices=devices,
        precision=str(cfg.trainer.get("precision", "32-true")),
        max_epochs=int(cfg.train.max_epochs),
        check_val_every_n_epoch=int(cfg.solver.get("check_val_every_n_epoch", 1)),
        gradient_clip_val=float(cfg.train.get("gradient_clip_val", 1.0)),
        accumulate_grad_batches=int(cfg.trainer.get("accumulate_grad_batches", 1)),
        num_sanity_val_steps=int(cfg.trainer.get("num_sanity_val_steps", 0)),
        log_every_n_steps=int(cfg.trainer.get("log_every_n_steps", 20)),
        enable_progress_bar=bool(cfg.trainer.get("enable_progress_bar", True)),
        logger=TensorBoardLogger(str(output / "logs"), name=""),
        callbacks=callbacks,
    )
    if args.test_only:
        metrics = trainer.test(solver, dataloaders=loaders["test"], ckpt_path=resume)
    else:
        trainer.fit(solver, loaders["train"], loaders["val"], ckpt_path=resume)
        print(f"Best {best_metric} checkpoint: {best_callback.best_model_path}")
        print(f"Latest training checkpoint: {checkpoints / 'last.ckpt'}")
        if not args.test_after_fit:
            if manifest["settings"]["validation_uses_test_tail"]:
                print("Validation already evaluates the shared test tail. Use --test-only --resume <best checkpoint> to report its test metrics.")
            else:
                print("Test tails remain unevaluated. Use --test-only --resume <best checkpoint> for the final test.")
            return
        metrics = trainer.test(solver, dataloaders=loaders["test"], ckpt_path=best_callback.best_model_path)
    if trainer.is_global_zero:
        (output / "test_metrics.json").write_text(json.dumps({
            "split_fingerprint": manifest["fingerprint"],
            "checkpoint": resume if args.test_only else best_callback.best_model_path,
            "sample_count": len(loaders["test"].dataset), "metrics": metrics,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
