#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyperparameter search for experimental 4D-STEM denoising (no GT)

This script tunes the four key HashGrid encoding hyperparameters using Optuna:
  - n_levels
  - log2_hashmap_size
  - base_resolution
  - n_features_per_level

Objective: minimize validation loss (e.g., Anscombe or MSE) from the
self-reconstruction pipeline in `full_script_6_experimental.py`.

Outputs per run (in --out_dir):
  - trials.csv: All trials with params and metrics
  - study_summary.json: Optuna best params and value
  - best_ckpt.pth: Best model weights with encoding/network configs
  - search_meta.json: Run metadata

Usage (example):
  conda activate py310_env
  python hparam_search_experimental.py \
    --npy /mnt/data_0/radin/hash_encoding_denoising/expermintal_dataset/n4_defocus-5test_512.npy \
    --trials 20 --steps 3000 --batch 262144 --loss anscombe --ema --train_on_centers \
    --out_dir /mnt/data_0/radin/hash_encoding_denoising/output/hparam_exp_search
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import optuna
import torch
import torch.nn.functional as F

# Import the experimental pipeline bits
from full_script_6_experimental import (
    Image4DExperimental,
    build_model,
    detect_preferred_device,
)


def parse_args():
    p = argparse.ArgumentParser(description="Optuna search for 4D-STEM experimental denoising (no GT)")
    p.add_argument("--npy", required=True, help="Path to 4D .npy file [H,W,A1,A2]")
    p.add_argument("--trials", type=int, default=50, help="#Optuna trials")
    p.add_argument("--steps", type=int, default=3000, help="#training steps per trial")
    p.add_argument("--batch", type=int, default=2**18, help="#coords per step")
    p.add_argument("--val_frac", type=float, default=0.02, help="Validation fraction (coords)")
    p.add_argument("--patience", type=int, default=600, help="Early stopping patience (steps)")
    p.add_argument("--loss", choices=["mse", "poisson", "anscombe"], default="mse")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--ema", action="store_true", help="Use EMA for evaluation")
    p.add_argument("--ema_decay", type=float, default=0.9995)
    p.add_argument("--train_on_centers", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="output/hparam_exp_search")
    # Network kept fixed here (these were stable in prior runs); override if needed
    p.add_argument("--n_neurons", type=int, default=128)
    p.add_argument("--n_hidden_layers", type=int, default=3)
    # Search spaces (can override defaults)
    p.add_argument("--levels", type=int, nargs="*", default=[4, 8, 16])
    p.add_argument("--log2_sizes", type=int, nargs="*", default=[ 17, 18, 19, 20, 21])
    p.add_argument("--bases", type=int, nargs="*", default=[8, 16, 32])
    p.add_argument("--features", type=int, nargs="*", default=[2, 4, 8])
    # Pruning and storage/resume
    p.add_argument("--pruner", choices=["none","median","sha"], default="median", help="Optuna pruner type")
    p.add_argument("--storage", type=str, help="Optuna storage URL for resume (e.g., sqlite:///search.db)")
    p.add_argument("--study", type=str, help="Optuna study name (use with --storage)")
    p.add_argument("--prune_warmup", type=int, default=300, help="Steps before considering pruning")
    return p.parse_args()


@dataclass
class TrialResult:
    step: int
    best_val: float
    best_state: Dict[str, torch.Tensor]


def loss_fn(kind: str, preds: torch.Tensor, targets: torch.Tensor, norm_scale: float) -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(preds, targets.to(preds.dtype))
    elif kind == "poisson":
        dtype = preds.dtype
        dev = preds.device
        scale_t = torch.tensor(float(norm_scale), dtype=dtype, device=dev)
        eps = torch.tensor(1e-8, dtype=dtype, device=dev)
        p = (preds.clamp(min=0) * scale_t) + eps
        t = (targets.to(dtype).clamp(min=0) * scale_t) + eps
        return torch.mean(p - t * torch.log(p))
    elif kind == "anscombe":
        dtype = preds.dtype
        dev = preds.device
        scale_t = torch.tensor(float(norm_scale), dtype=dtype, device=dev)
        c = torch.tensor(3.0/8.0, dtype=dtype, device=dev)
        two = torch.tensor(2.0, dtype=dtype, device=dev)
        p = (preds.clamp(min=0) * scale_t)
        t = (targets.to(dtype).clamp(min=0) * scale_t)
        ap = two * torch.sqrt(p + c)
        at = two * torch.sqrt(t + c)
        return F.mse_loss(ap, at)
    else:
        raise ValueError(f"Unknown loss: {kind}")


def run_single_training(
    image: Image4DExperimental,
    encoding_cfg: Dict[str, Any],
    network_cfg: Dict[str, Any],
    *,
    steps: int,
    batch: int,
    val_frac: float,
    patience: int,
    loss_kind: str,
    lr: float,
    ema: bool,
    ema_decay: float,
    train_on_centers: bool,
    device: torch.device,
    trial: optuna.Trial | None = None,
    prune_warmup: int = 300,
) -> TrialResult:
    n_input_dims = 4
    n_channels = 1

    model = build_model(encoding_cfg, network_cfg, device, n_input_dims=n_input_dims, n_output_dims=n_channels)
    model_ema = None
    if ema:
        model_ema = build_model(encoding_cfg, network_cfg, device, n_input_dims=n_input_dims, n_output_dims=n_channels)
        for p in model_ema.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr*0.1)

    H, W, A1, A2, _ = image.shape
    n_val = max(10_000, int(batch * val_frac))
    if train_on_centers:
        ix = torch.randint(H, (n_val,), device=device)
        iy = torch.randint(W, (n_val,), device=device)
        iz = torch.randint(A1, (n_val,), device=device)
        iw = torch.randint(A2, (n_val,), device=device)
        val_coords = torch.stack(((ix.float()+0.5)/H, (iy.float()+0.5)/W, (iz.float()+0.5)/A1, (iw.float()+0.5)/A2), dim=1)
    else:
        val_coords = torch.rand([n_val, n_input_dims], device=device, dtype=torch.float32)

    best_val = float('inf')
    best_step = -1
    best_state = None

    for step in range(steps):
        model.train()
        if train_on_centers:
            ix = torch.randint(H, (batch,), device=device)
            iy = torch.randint(W, (batch,), device=device)
            iz = torch.randint(A1, (batch,), device=device)
            iw = torch.randint(A2, (batch,), device=device)
            coords = torch.stack(((ix.float()+0.5)/H, (iy.float()+0.5)/W, (iz.float()+0.5)/A1, (iw.float()+0.5)/A2), dim=1)
        else:
            coords = torch.rand([batch, n_input_dims], device=device, dtype=torch.float32)

        targets = image(coords)  # [B,1]
        preds = model(coords)
        loss = loss_fn(loss_kind, preds, targets, image.normalization_scale)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # EMA update
        if model_ema is not None:
            with torch.no_grad():
                beta = ema_decay
                for p, pe in zip(model.parameters(), model_ema.parameters()):
                    pe.data.mul_(beta).add_(p.data, alpha=1.0 - beta)

        # Validate periodically (every 50 steps)
        if step % 50 == 0:
            eval_model = model_ema if model_ema is not None else model
            eval_model.eval()
            with torch.no_grad():
                vp = eval_model(val_coords)
                vt = image(val_coords)
                vloss = float(loss_fn(loss_kind, vp, vt, image.normalization_scale).item())
            scheduler.step()
            if vloss < best_val - 1e-8:
                best_val = vloss
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}

            # Report to Optuna for pruning
            if trial is not None:
                trial.report(best_val, step)
                if step >= prune_warmup and trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned at step {step} with best_val={best_val:.6e}")

        # Early stopping
        if best_step >= 0 and (step - best_step) > patience:
            break

    return TrialResult(step=best_step, best_val=float(best_val), best_state=best_state)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = detect_preferred_device()
    print("Using device:", device)

    # Dataset
    image = Image4DExperimental(args.npy, device, norm="max", max_clip_percentile=100.0)
    H, W, A1, A2, _ = image.shape
    print(f"Data {args.npy} | shape={H,W,A1,A2} | norm_scale={image.normalization_scale:.6g}")

    def make_network_cfg():
        return {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": int(args.n_neurons),
            "n_hidden_layers": int(args.n_hidden_layers),
        }

    def make_encoding_cfg(n_levels, log2_hsize, base_res, n_feat):
        return {
            "otype": "HashGrid",
            "n_levels": int(n_levels),
            "n_features_per_level": int(n_feat),
            "log2_hashmap_size": int(log2_hsize),
            "base_resolution": int(base_res),
            "per_level_scale": 1.5,
        }

    # Prepare Optuna study (minimize validation loss)
    # Create or load study with optional storage and pruner
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=args.prune_warmup, interval_steps=50)
    elif args.pruner == "sha":
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=200, reduction_factor=3, min_early_stopping_rate=0)
    else:
        pruner = optuna.pruners.NopPruner()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    if args.storage:
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, storage=args.storage, study_name=(args.study or "hashgrid_search"), load_if_exists=True)
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial):
        # Sample params
        # Slightly narrower defaults to speed up search; can still be overridden via CLI
        levels = args.levels if args.levels else [4, 8, 16]
        sizes = args.log2_sizes if args.log2_sizes else [18, 19, 20, 21]
        bases = args.bases if args.bases else [8, 16]
        feats = args.features if args.features else [2, 4, 8]

        n_levels = trial.suggest_categorical("n_levels", levels)
        log2_hashmap_size = trial.suggest_categorical("log2_hashmap_size", sizes)
        base_resolution = trial.suggest_categorical("base_resolution", bases)
        n_features_per_level = trial.suggest_categorical("n_features_per_level", feats)

        enc = make_encoding_cfg(n_levels, log2_hashmap_size, base_resolution, n_features_per_level)
        net = make_network_cfg()

        t0 = time.perf_counter()
        result = run_single_training(
            image,
            enc,
            net,
            steps=args.steps,
            batch=args.batch,
            val_frac=args.val_frac,
            patience=args.patience,
            loss_kind=args.loss,
            lr=args.lr,
            ema=args.ema,
            ema_decay=args.ema_decay,
            train_on_centers=args.train_on_centers,
            device=device,
            trial=trial,
            prune_warmup=args.prune_warmup,
        )
        dt = time.perf_counter() - t0

        # Log useful attrs
        trial.set_user_attr("best_step", int(result.step))
        trial.set_user_attr("time_s", float(dt))
        trial.set_user_attr("encoding", enc)
        trial.set_user_attr("network", net)

        return result.best_val

    print("Starting Optuna search…")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Recover best trial result by re-running once to get state dict
    best = study.best_trial
    print("Best trial:", best.number, "value (val_loss)=", best.value)
    bp = best.params
    enc_best = make_encoding_cfg(bp["n_levels"], bp["log2_hashmap_size"], bp["base_resolution"], bp["n_features_per_level"])
    net_best = make_network_cfg()

    final = run_single_training(
        image,
        enc_best,
        net_best,
        steps=args.steps,
        batch=args.batch,
        val_frac=args.val_frac,
        patience=args.patience,
        loss_kind=args.loss,
        lr=args.lr,
        ema=args.ema,
        ema_decay=args.ema_decay,
        train_on_centers=args.train_on_centers,
        device=device,
    )

    # Save checkpoint
    ckpt = {
        "state_dict": final.best_state,
        "encoding": enc_best,
        "network": net_best,
    }
    ckpt_out = os.path.join(args.out_dir, "best_ckpt.pth")
    torch.save(ckpt, ckpt_out)
    print("Saved best checkpoint:", ckpt_out)

    # Save trials to CSV and summary JSON
    import pandas as pd

    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial": t.number,
            "value": float(t.value),
            "n_levels": t.params.get("n_levels"),
            "log2_hashmap_size": t.params.get("log2_hashmap_size"),
            "base_resolution": t.params.get("base_resolution"),
            "n_features_per_level": t.params.get("n_features_per_level"),
            "best_step": int(t.user_attrs.get("best_step", -1)),
            "time_s": float(t.user_attrs.get("time_s", np.nan)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_out = os.path.join(args.out_dir, "trials.csv")
    df.to_csv(csv_out, index=False)
    print("Saved trials CSV:", csv_out)

    summary = {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "out_dir": args.out_dir,
        "npy": args.npy,
        "search_space": {
            "n_levels": args.levels,
            "log2_hashmap_size": args.log2_sizes,
            "base_resolution": args.bases,
            "n_features_per_level": args.features,
        },
        "training": {
            "steps": int(args.steps),
            "batch": int(args.batch),
            "val_frac": float(args.val_frac),
            "patience": int(args.patience),
            "loss": args.loss,
            "lr": float(args.lr),
            "ema": bool(args.ema),
            "ema_decay": float(args.ema_decay),
            "train_on_centers": bool(args.train_on_centers),
        },
        "data": {
            "shape": [int(H), int(W), int(A1), int(A2)],
            "normalization_scale": float(image.normalization_scale),
        },
    }
    with open(os.path.join(args.out_dir, "study_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved study summary JSON")


if __name__ == "__main__":
    main()
