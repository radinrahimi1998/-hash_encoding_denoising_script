#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Experimental 4D-STEM denoising with hash encoding (no ground truth)
# - Fits an implicit network to the noisy measurements (self-reconstruction)
# - Relies on the implicit prior + early stopping to avoid overfitting noise
# - Supports chunked inference and optional full-volume export via memmap
# Note: This script expects a .npy file [H, W, A1, A2]. 

import argparse
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import tinycudann as tcnn
from numpy.lib.format import open_memmap
import math
import csv

from common import ROOT_DIR


def detect_preferred_device(prefer_name_substring="H200", default_idx=0):
    device = torch.device(f"cuda:{default_idx}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            if prefer_name_substring in name:
                return torch.device(f"cuda:{i}")
    return device


class Image4DExperimental(torch.nn.Module):
    """Tensor fetcher for a 4D dataset stored as a single .npy.

    Expects an array of shape [H, W, A1, A2] with real-valued counts.
    Adds a channel dim => [H, W, A1, A2, 1]
    Provides normalized data in [0, 1] by dividing with a chosen scale.
    """

    def __init__(self, npy_path: str, device: torch.device, norm: str = "max", max_clip_percentile: float = 100.0):
        super().__init__()

        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"File not found: {npy_path}")

        arr = np.load(npy_path)
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D array [H, W, A1, A2], got shape {arr.shape}")

        # Convert to float32 tensor on device
        data = torch.as_tensor(arr, dtype=torch.float32, device=device)

        # Choose normalization scale
        if norm == "max":
            if max_clip_percentile < 100.0:
                # robust max to avoid extreme outliers
                scale = np.percentile(arr.reshape(-1), max_clip_percentile)
            else:
                scale = float(data.max().item() if data.numel() > 0 else 1.0)
            scale = max(scale, 1e-6)
        elif norm == "mean":
            scale = float(data.mean().item() if data.numel() > 0 else 1.0)
            scale = max(scale, 1e-6)
        else:
            raise ValueError("norm must be one of {'max','mean'}")

        self.normalization_scale = scale
        data = data / scale
        # Keep normalized range bounded to [0,1] to avoid upward bias
        data = data.clamp(min=0.0, max=1.0)

        # Add channel dim
        self.data = data[..., None]  # [H, W, A1, A2, 1]
        self.shape = self.data.shape
        self.npy_path = npy_path

    def forward(self, xs):
        # xs in [0,1]^4 -> sample voxel with nearest neighbor (rounded indices)
        with torch.no_grad():
            shape = torch.tensor(self.shape[:-1], device=xs.device).float()  # [H,W,A1,A2]
            xs = xs * shape
            idx = torch.round(xs)
            x0 = idx[:, 0].clamp(min=0, max=shape[0]-1).long()
            y0 = idx[:, 1].clamp(min=0, max=shape[1]-1).long()
            z0 = idx[:, 2].clamp(min=0, max=shape[2]-1).long()
            w0 = idx[:, 3].clamp(min=0, max=shape[3]-1).long()
            return self.data[x0, y0, z0, w0]  # [N,1]


def build_model(encoding_cfg, network_cfg, device, n_input_dims=4, n_output_dims=1):
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=n_input_dims,
        n_output_dims=n_output_dims,
        encoding_config=encoding_cfg,
        network_config=network_cfg,
    ).to(device)
    return model


def chunked_inference(model, resolution, device, out_dtype=torch.float16, scan_block=(32, 32), batch_points=1_000_000, out_path=None, normalization_scale=1.0, clamp_mode="sigmoid"):
    """
    Predict the entire 4D volume in chunks to avoid OOM.

    - resolution: (H, W, A1, A2)
    - scan_block: process scanning tiles of size (bh, bw), always over full angular grid per tile
    - batch_points: feed that many coords per forward pass for speed/memory tradeoff
    - out_path: if provided, writes a .npy memmap (float16 by default) incrementally
    - normalization_scale: used for metadata reporting; prediction stays normalized unless caller rescales
    """

    H, W, A1, A2 = resolution

    # Prepare output container
    if out_path is not None:
        # Create a proper .npy file with header using open_memmap
        np_dtype = np.float16 if out_dtype == torch.float16 else np.float32
        out_mm = open_memmap(out_path, mode='w+', dtype=np_dtype, shape=(H, W, A1, A2))
    else:
        out_buf = np.empty((H, W, A1, A2), dtype=np.float16 if out_dtype == torch.float16 else np.float32)

    model.eval()
    with torch.no_grad():
        bh, bw = scan_block
        xs_1d = torch.linspace(0.5/H, 1-0.5/H, H, device=device)
        ys_1d = torch.linspace(0.5/W, 1-0.5/W, W, device=device)
        zs_1d = torch.linspace(0.5/A1, 1-0.5/A1, A1, device=device)
        ws_1d = torch.linspace(0.5/A2, 1-0.5/A2, A2, device=device)

        # Precompute angular mesh once
        zv, wv = torch.meshgrid(zs_1d, ws_1d, indexing='ij')
        zw_flat = torch.stack((zv.flatten(), wv.flatten()), dim=1)  # [A1*A2, 2]

        t0 = time.perf_counter()
        tiles_done = 0
        for x_start in range(0, H, bh):
            x_end = min(x_start + bh, H)
            x_slice = xs_1d[x_start:x_end]
            for y_start in range(0, W, bw):
                y_end = min(y_start + bw, W)
                y_slice = ys_1d[y_start:y_end]

                xv, yv = torch.meshgrid(x_slice, y_slice, indexing='ij')  # [bh,bw]
                xy_flat = torch.stack((xv.flatten(), yv.flatten()), dim=1)  # [bh*bw,2]

                # Combine with angular grid (cartesian product)
                # final coords: [N,4]
                xy_rep = xy_flat[:, None, :].expand(-1, zw_flat.shape[0], -1).reshape(-1, 2)
                zw_rep = zw_flat.repeat(xy_flat.shape[0], 1)
                coords = torch.cat((xy_rep, zw_rep), dim=1)  # [bh*bw*A1*A2,4]

                # Batched forward passes
                preds = []
                for s in range(0, coords.shape[0], batch_points):
                    batch = coords[s:s+batch_points]
                    out = model(batch)
                    preds.append(out.detach())
                tile = torch.cat(preds, dim=0).view(x_end - x_start, y_end - y_start, A1, A2)
                if clamp_mode == "sigmoid":
                    tile = tile.clamp(min=0.0, max=1.0)
                elif clamp_mode == "relu":
                    tile = torch.relu(tile)
                elif clamp_mode == "none":
                    pass
                else:
                    tile = tile.clamp(min=0.0, max=1.0)

                tile_np = tile.detach().float().cpu().numpy()
                if out_path is not None:
                    out_mm[x_start:x_end, y_start:y_end, :, :] = tile_np.astype(out_mm.dtype)
                else:
                    out_buf[x_start:x_end, y_start:y_end, :, :] = tile_np.astype(out_buf.dtype)

                tiles_done += 1
                if tiles_done % 16 == 0:
                    dt = time.perf_counter() - t0
                    print(f"Processed {tiles_done} tiles in {dt:.1f}s")

        if out_path is not None:
            del out_mm  # flush and finalize .npy

    return out_path if out_path is not None else out_buf


# === New helper utilities ===
def summarize_dataset(array_path, max_print=10):
    import numpy as _np, os as _os
    if not _os.path.isfile(array_path):
        raise FileNotFoundError(array_path)
    arr = _np.load(array_path, mmap_mode='r')
    stats = {
        "path": array_path,
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "size_mb": round(arr.nbytes / 1024**2, 3),
        "min": float(_np.min(arr)),
        "mean": float(_np.mean(arr)),
        "max": float(_np.max(arr)),
        "pcts": {str(p): float(v) for p, v in zip([0.1,1,5,50,95,99,99.9], _np.percentile(arr, [0.1,1,5,50,95,99,99.9]))},
    }
    print("== Dataset summary ==")
    for k,v in stats.items():
        if k != 'pcts':
            print(f"  {k}: {v}")
    print("  percentiles:")
    for pk, pv in stats['pcts'].items():
        print(f"    {pk}%: {pv}")
    return stats


def compute_param_count(model: torch.nn.Module):
    total = 0
    for p in model.parameters():
        total += p.numel()
    return total


def estimate_checkpoint_size_bytes(model: torch.nn.Module, dtype_bytes=4):
    # Assume most params are float32 unless model converted
    return compute_param_count(model) * dtype_bytes


def generate_quick_visuals(arr: np.ndarray, out_dir: str, bf_frac: float = 0.05, adf_min_frac: float = 0.15, adf_max_frac: float = 0.35):
    import matplotlib.pyplot as _plt, numpy as _np, os as _os
    _os.makedirs(out_dir, exist_ok=True)
    if arr.ndim != 4:
        print("[quick_viz] Skipping visuals: array not 4D")
        return
    H,W,A1,A2 = arr.shape
    arr32 = arr.astype(np.float32)
    mean_dp = arr32.mean(axis=(0,1))
    central_dp = arr32[H//2, W//2]
    total_int = arr32.sum(axis=(2,3))
    cy, cx = A1//2, A2//2
    yy, xx = np.ogrid[:A1,:A2]
    r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    r_norm = r / r.max()
    bf_mask = r_norm <= bf_frac
    adf_mask = (r_norm >= adf_min_frac) & (r_norm <= adf_max_frac)
    bf_map = arr32[..., bf_mask].reshape(H,W,-1).sum(-1)
    adf_map = arr32[..., adf_mask].reshape(H,W,-1).sum(-1)

    def save_img(data, fname, cmap='magma', log=False):
        _plt.figure(figsize=(4,4))
        disp = np.log1p(data) if log else data
        _plt.imshow(disp, cmap=cmap)
        _plt.colorbar()
        _plt.tight_layout()
        _plt.savefig(os.path.join(out_dir, fname), dpi=150)
        _plt.close()

    save_img(mean_dp, 'mean_dp.png', log=True)
    save_img(central_dp, 'central_dp.png', log=True)
    save_img(total_int, 'total_intensity.png', cmap='gray', log=False)
    save_img(bf_map, f'bf_map_frac{bf_frac}.png', cmap='gray', log=False)
    save_img(adf_map, f'adf_map_{adf_min_frac}-{adf_max_frac}.png', cmap='gray', log=False)

    # Overlay masks on mean DP
    import matplotlib.pyplot as _plt2
    overlay = np.zeros((A1,A2,3), dtype=np.float32)
    norm_mean = np.log1p(mean_dp); norm_mean /= (norm_mean.max() + 1e-9)
    overlay[...,0] = norm_mean  # R
    overlay[...,1] = norm_mean  # G
    overlay[...,2] = norm_mean  # B
    # red for ADF ring edges, green for BF
    bf_edge = np.logical_and(np.abs(r_norm - bf_frac) < (1.0 / A1), True)
    adf_edge_min = np.logical_and(np.abs(r_norm - adf_min_frac) < (1.0 / A1), True)
    adf_edge_max = np.logical_and(np.abs(r_norm - adf_max_frac) < (1.0 / A1), True)
    overlay[bf_edge] = [0,1,0]
    overlay[adf_edge_min] = [1,0,0]
    overlay[adf_edge_max] = [1,0,0]
    _plt2.figure(figsize=(4,4))
    _plt2.imshow(overlay)
    _plt2.title('Mean DP with BF (green) / ADF (red) rings')
    _plt2.axis('off')
    _plt2.tight_layout()
    _plt2.savefig(os.path.join(out_dir, 'mask_overlay.png'), dpi=150)
    _plt2.close()
    print(f"[quick_viz] Saved visualization set to {out_dir}")


def add_compression_info(meta: dict, raw_path: str, ckpt_path: str | None):
    import os as _os
    def sz(p):
        return _os.path.getsize(p) if p and _os.path.exists(p) else None
    raw_b = sz(raw_path)
    ckpt_b = sz(ckpt_path)
    ratio = (raw_b / ckpt_b) if (raw_b and ckpt_b and ckpt_b > 0) else None
    meta['compression'] = {
        'raw_file': raw_path,
        'raw_size_mb': None if raw_b is None else round(raw_b/1024**2,3),
        'checkpoint_file': ckpt_path,
        'checkpoint_size_mb': None if ckpt_b is None else round(ckpt_b/1024**2,3),
        'raw_to_model_ratio': None if ratio is None else round(ratio,2),
    }
    if ratio is not None:
        print(f"Compression: raw -> model = {ratio:.2f}:1 (raw {meta['compression']['raw_size_mb']} MB, model {meta['compression']['checkpoint_size_mb']} MB)")
    return meta


def parse_args():
    p = argparse.ArgumentParser(description="Experimental 4D-STEM denoising with hash encoding (no GT)")
    p.add_argument("--npy", required=True, help="Path to 4D .npy file [H,W,A1,A2]. Use convert_h5_to_npy.py for .h5/.hdf5.")
    p.add_argument("--config", default=os.path.join(ROOT_DIR, "hash_encoding_denoising/data/config_hash.json"), help="tiny-cuda-nn config json")
    p.add_argument("--steps", type=int, default=8000, help="Training steps")
    p.add_argument("--batch", type=int, default=2**18, help="Batch size (#coords per step)")
    p.add_argument("--val_frac", type=float, default=0.02, help="Validation fraction of random coords")
    p.add_argument("--patience", type=int, default=400, help="Early stopping patience (steps)")
    p.add_argument("--norm", choices=["max","mean"], default="max", help="Normalization strategy")
    p.add_argument("--clip_pct", type=float, default=100.0, help="Percentile for robust max (<=100)")
    p.add_argument("--out_dir", default="output", help="Directory to save outputs")
    p.add_argument("--export_full", action="store_true", help="Export full 4D volume (large!)")
    p.add_argument("--export_in_counts", action="store_true", help="Export outputs scaled back to counts domain (x normalization scale)")
    p.add_argument("--scan_block", type=int, nargs=2, default=(32,32), help="Scan block size H W for chunked export")
    p.add_argument("--batch_points", type=int, default=1_000_000, help="#coords per forward during export")
    p.add_argument("--seed", type=int, default=42)
    # Training improvements
    p.add_argument("--loss", choices=["mse","poisson","anscombe"], default="mse", help="Training loss; poisson/anscombe operate in count domain using normalization scale")
    p.add_argument("--lr", type=float, default=3e-5, help="Optimizer learning rate")
    p.add_argument("--grad_clip", type=float, default=0.0, help="Clip grad-norm if > 0")
    p.add_argument("--scheduler", choices=["plateau","cosine"], default="plateau", help="LR scheduler type")
    p.add_argument("--ema", action="store_true", help="Enable EMA of weights for validation/export")
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay (close to 1.0)")
    p.add_argument("--train_on_centers", action="store_true", help="Sample coords exactly at pixel centers for training/validation to avoid rounding artifacts")
    # Encoding/network overrides
    p.add_argument("--optimized", action="store_true", help="Use best-found encoding defaults from search (recommended)")
    p.add_argument("--n_levels", type=int, help="Override encoding n_levels")
    p.add_argument("--n_features_per_level", type=int, help="Override encoding n_features_per_level")
    p.add_argument("--log2_hashmap_size", type=int, help="Override encoding log2_hashmap_size")
    p.add_argument("--base_resolution", type=int, help="Override encoding base_resolution")
    p.add_argument("--per_level_scale", type=float, help="Override encoding per_level_scale")
    p.add_argument("--output_activation", choices=["None","Sigmoid"], help="Override network output activation")
    p.add_argument("--n_neurons", type=int, help="Override network hidden width")
    p.add_argument("--n_hidden_layers", type=int, help="Override network depth (hidden layers)")
    p.add_argument("--clamp_output", choices=["sigmoid","relu","none"], default="sigmoid", help="Post-activation clamp for outputs during export/diagnostic")
    # Checkpointing
    p.add_argument("--save_ckpt", action="store_true", help="Save best weights checkpoint (.pth) to out_dir")
    p.add_argument("--ckpt_path", type=str, help="Explicit path to save/load checkpoint (.pth)")
    p.add_argument("--load_ckpt", type=str, help="Path to checkpoint to load weights from")
    p.add_argument("--export_only", action="store_true", help="Skip training, load checkpoint and export diagnostics/full")
    # New options
    p.add_argument("--quick_viz", action="store_true", help="Generate quick visualization PNGs (mean DP, central DP, intensity, BF/ADF)")
    p.add_argument("--bf_frac", type=float, default=0.05, help="Fraction of detector radius for BF mask (0-1)")
    p.add_argument("--adf_min_frac", type=float, default=0.15, help="Min fraction of detector radius for ADF ring")
    p.add_argument("--adf_max_frac", type=float, default=0.35, help="Max fraction of detector radius for ADF ring")
    p.add_argument("--report_compression", action="store_true", help="Report raw->model compression (requires --save_ckpt)")
    # Diagnostics / logging
    p.add_argument("--diagnostics", action="store_true", help="Enable training diagnostics plots/CSV (loss curves, LR, grad norms)")
    p.add_argument("--log_interval", type=int, default=50, help="Interval (steps) for validation & logging")
    p.add_argument("--diag_dir", type=str, help="Directory to write diagnostics (default: out_dir)")
    # Advanced diagnostics
    p.add_argument("--holdout_frac", type=float, default=0.0, help="Fraction of coords reserved as holdout (not used for training or standard val)")
    p.add_argument("--residual_spectrum_interval", type=int, default=0, help="If >0 compute residual FFT radial profile every N steps (costly)")
    p.add_argument("--radial_profile", action="store_true", help="Compute raw vs denoised radial diffraction profile at end")
    p.add_argument("--track_bf_adf", action="store_true", help="Track BF/ADF intensity (central tile) over training intervals")
    p.add_argument("--bf_frac_track", type=float, default=0.05, help="BF fraction radius for tracking")
    p.add_argument("--adf_min_frac_track", type=float, default=0.15, help="ADF min fraction")
    p.add_argument("--adf_max_frac_track", type=float, default=0.35, help="ADF max fraction")
    p.add_argument("--early_stop_rel_improve", type=float, default=0.0, help="If >0 enable relative improvement early stop criterion (val window)")
    p.add_argument("--early_stop_window", type=int, default=400, help="Window size (steps) for relative improvement early stop")
    # Snapshot exports
    p.add_argument("--snapshot_interval", type=int, default=0, help="If >0, export central diagnostic block every N steps")
    p.add_argument("--max_snapshots", type=int, default=20, help="Maximum number of snapshot blocks to save (oldest pruned after exceed)")
    return p.parse_args()
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = detect_preferred_device()
    print("== Device selection ==")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {name} ({mem:.1f} GB)")
    print(f"Using device: {device}")

    # Load config and enforce intended keys for encoding
    with open(args.config) as f:
        config = json.load(f)

    # Ensure keys exist
    enc = config.get("encoding", {})
    enc.setdefault("otype", "HashGrid")
    enc.setdefault("n_levels", 8)
    enc.setdefault("n_features_per_level", 8)
    enc.setdefault("log2_hashmap_size", 19)
    enc.setdefault("base_resolution", 8)
    enc.setdefault("per_level_scale", 1.5)
    # Apply optimized defaults or explicit overrides
    if args.optimized:
        enc.update({
            # Best from hparam search (50 trials, anscombe, steps=3000, batch=262144)
            "n_levels": 4,
            "n_features_per_level": 8,
            "log2_hashmap_size": 21,
            "base_resolution": 8,
            "per_level_scale": 1.5,
        })
    # Individual overrides
    for k in ["n_levels", "n_features_per_level", "log2_hashmap_size", "base_resolution", "per_level_scale"]:
        v = getattr(args, k)
        if v is not None:
            enc[k] = v
    config["encoding"] = enc

    net = config.get("network", {})
    net.setdefault("otype", "FullyFusedMLP")
    net.setdefault("activation", "ReLU")
    net.setdefault("output_activation", "None")
    net.setdefault("n_neurons", 128)
    net.setdefault("n_hidden_layers", 3)
    if args.output_activation is not None:
        net["output_activation"] = args.output_activation
    if args.n_neurons is not None:
        net["n_neurons"] = int(args.n_neurons)
    if args.n_hidden_layers is not None:
        net["n_hidden_layers"] = int(args.n_hidden_layers)
    # If optimized preset is requested, also adopt best-found network widths/depths
    if args.optimized:
        net.update({
            "n_neurons": 128,
            "n_hidden_layers": 3,
        })
    config["network"] = net

    print("== Encoding config ==")
    for k, v in config["encoding"].items():
        print(f"  {k}: {v}")
    print("== Network config ==")
    for k, v in config["network"].items():
        print(f"  {k}: {v}")

    # Resolve input: require a .npy file (use convert_h5_to_npy.py for HDF5)
    resolved_npy = args.npy
    if not os.path.isfile(resolved_npy):
        raise FileNotFoundError(f".npy file not found: {resolved_npy}")
    if not resolved_npy.lower().endswith('.npy'):
        raise ValueError("This script requires a .npy input. Use convert_h5_to_npy.py for .h5/.hdf5.")

    # Dataset
    image = Image4DExperimental(resolved_npy, device, norm=args.norm, max_clip_percentile=args.clip_pct)
    print(f"Loaded data from: {resolved_npy}")
    print(f"Data shape (with channel): {tuple(image.shape)}  normalization_scale: {image.normalization_scale:.6g}")
    # Base name early (needed for snapshot exports during training)
    base_name = os.path.splitext(os.path.basename(resolved_npy))[0]
    # Dataset summary if requested
    summarize_dataset(resolved_npy)

    # Model
    n_input_dims = 4
    n_channels = 1
    model = build_model(config["encoding"], config["network"], device, n_input_dims=n_input_dims, n_output_dims=n_channels)
    print(model)
    # Parameter / size estimates
    param_count = compute_param_count(model)
    est_bytes = estimate_checkpoint_size_bytes(model, dtype_bytes=4)  # assuming fp32
    print(f"Model parameter count: {param_count:,}  (~{est_bytes/1024**2:.2f} MB fp32)")

    # Optional EMA model for validation/export
    model_ema = None
    # If loading checkpoint or export_only, load weights now
    if args.load_ckpt or args.export_only:
        ckpt_path = args.load_ckpt or args.ckpt_path
        if ckpt_path is None:
            base_name = os.path.splitext(os.path.basename(args.npy))[0]
            ckpt_path = os.path.join(args.out_dir, f"{base_name}_best.pth")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--load_ckpt specified but file not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        # Rebuild model if checkpoint contains encoding/network overrides
        enc_ck = ckpt.get('encoding')
        net_ck = ckpt.get('network')
        if enc_ck is not None and net_ck is not None:
            model = build_model(enc_ck, net_ck, device, n_input_dims=n_input_dims, n_output_dims=n_channels)
        model.load_state_dict(ckpt['state_dict'])
        if ckpt.get('ema_state_dict') is not None:
            model_ema = build_model(enc_ck or config['encoding'], net_ck or config['network'], device, n_input_dims=n_input_dims, n_output_dims=n_channels)
            model_ema.load_state_dict(ckpt['ema_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr*0.1)

    if args.ema and model_ema is None:
        model_ema = build_model(config["encoding"], config["network"], device, n_input_dims=n_input_dims, n_output_dims=n_channels)
        for p in model_ema.parameters():
            p.requires_grad = False

    # Validation coordinate pool (fixed)
    H, W, A1, A2, _ = image.shape
    n_val = max(10_000, int(args.batch * args.val_frac))
    n_holdout = 0
    holdout_coords = None
    if args.train_on_centers:
        ix = torch.randint(H, (n_val,), device=device)
        iy = torch.randint(W, (n_val,), device=device)
        iz = torch.randint(A1, (n_val,), device=device)
        iw = torch.randint(A2, (n_val,), device=device)
        val_coords = torch.stack(((ix.float()+0.5)/H, (iy.float()+0.5)/W, (iz.float()+0.5)/A1, (iw.float()+0.5)/A2), dim=1)
    else:
        val_coords = torch.rand([n_val, n_input_dims], device=device, dtype=torch.float32)
    if args.holdout_frac > 0.0:
        n_holdout = int(n_val * args.holdout_frac)
        if n_holdout > 0:
            holdout_coords = val_coords[:n_holdout].clone()
            val_coords = val_coords[n_holdout:]
            print(f"Created holdout set: {n_holdout} coords; validation now {val_coords.shape[0]}")
    traced_image = image
    try:
        _trace_in = (torch.rand([args.batch, n_input_dims], device=device, dtype=torch.float32)
                     if not args.train_on_centers else torch.stack((
                         torch.randint(H,(args.batch,),device=device).float().add_(0.5).div_(H),
                         torch.randint(W,(args.batch,),device=device).float().add_(0.5).div_(W),
                         torch.randint(A1,(args.batch,),device=device).float().add_(0.5).div_(A1),
                         torch.randint(A2,(args.batch,),device=device).float().add_(0.5).div_(A2),
                     ), dim=1))
        traced_image = torch.jit.trace(image, _trace_in)
    except Exception as e:
        print(f"JIT trace failed (fallback to eager): {e}")

    # Training loop (self-reconstruction, no GT)
    n_steps = args.steps
    batch_size = args.batch
    losses = []
    val_losses = []
    best_val = float('inf')
    best_step = -1
    best_state = None
    grad_norms = []
    lrs = []
    holdout_losses = []
    bf_adf_track = []  # (step, bf_pred, adf_pred)
    residual_psd = []  # list of dicts {step: int, radial_freq: np.array, radial_power: np.array}

    # Precompute masks for BF/ADF tracking & radial FFT (central tile) if requested
    central_tile_coords = None
    if args.track_bf_adf or args.residual_spectrum_interval > 0:
        # Central scan tile 16x16 or smaller
        tH = min(16, H); tW = min(16, W)
        xs = torch.linspace(0.5/H, 1-0.5/H, H, device=device)[H//2 - tH//2: H//2 - tH//2 + tH]
        ys = torch.linspace(0.5/W, 1-0.5/W, W, device=device)[W//2 - tW//2: W//2 - tW//2 + tW]
        zs = torch.linspace(0.5/A1, 1-0.5/A1, A1, device=device)
        ws = torch.linspace(0.5/A2, 1-0.5/A2, A2, device=device)
        xv, yv, zv, wv = torch.meshgrid(xs, ys, zs, ws, indexing='ij')
        central_tile_coords = torch.stack((xv.flatten(), yv.flatten(), zv.flatten(), wv.flatten()), dim=1)
        # radial masks for diffraction plane
        import numpy as _np
        cy, cx = A1//2, A2//2
        yy, xx = _np.ogrid[:A1, :A2]
        r = _np.sqrt((yy-cy)**2 + (xx-cx)**2)
        r_norm = r / r.max()
        bf_mask_np = (r_norm <= args.bf_frac_track)
        adf_mask_np = (r_norm >= args.adf_min_frac_track) & (r_norm <= args.adf_max_frac_track)
        bf_mask = torch.as_tensor(bf_mask_np, device=device)
        adf_mask = torch.as_tensor(adf_mask_np, device=device)
        # Binning for radial PSD
        if args.residual_spectrum_interval > 0:
            radial_bins = _np.linspace(0, 1, 48)
            radial_bin_centers = 0.5*(radial_bins[:-1]+radial_bins[1:])
    else:
        bf_mask = adf_mask = None

    t_start = time.perf_counter()
    peak_mem = torch.cuda.memory_allocated(device) / (1024**3) if device.type == 'cuda' else 0.0

    # Export-only path
    if args.export_only:
        print("--export_only specified: skipping training and proceeding to export.")
        best_state = (model_ema or model).state_dict()
        best_step = -1
        best_val = float('nan')
    else:
        print(f"Starting training for {n_steps} steps, batch={batch_size}")

    def loss_fn(preds, targets):
        """Stable loss computation. Always do heavy math in float32 to avoid fp16 underflow/NaNs."""
        dev = preds.device
        if args.loss == 'mse':
            return F.mse_loss(preds, targets.to(preds.dtype))
        elif args.loss == 'poisson':
            # Generalized KL in count domain; compute in float32 for stability
            # Use softplus to ensure strictly positive predicted counts to avoid log(0) and fp16 instabilities.
            scale_f = float(image.normalization_scale)
            p32 = F.softplus(preds.float()) * scale_f + 1e-8  # >0
            t32 = torch.relu(targets.float()) * scale_f + 1e-8  # >=0
            return torch.mean(p32 - t32 * torch.log(p32))
        elif args.loss == 'anscombe':
            # Anscombe variance-stabilizing transform; compute in float32 for stability
            scale_f = float(image.normalization_scale)
            p32 = (preds.float().clamp(min=0) * scale_f)
            t32 = (targets.float().clamp(min=0) * scale_f)
            ap = 2.0 * torch.sqrt(p32 + 3.0/8.0)
            at = 2.0 * torch.sqrt(t32 + 3.0/8.0)
            return F.mse_loss(ap, at)
        else:
            raise ValueError("Unknown loss")
    for step in range(n_steps):
        if args.export_only:
            break
        model.train()
        if args.train_on_centers:
            ix = torch.randint(H, (batch_size,), device=device)
            iy = torch.randint(W, (batch_size,), device=device)
            iz = torch.randint(A1, (batch_size,), device=device)
            iw = torch.randint(A2, (batch_size,), device=device)
            batch = torch.stack(((ix.float()+0.5)/H, (iy.float()+0.5)/W, (iz.float()+0.5)/A1, (iw.float()+0.5)/A2), dim=1)
        else:
            batch = torch.rand([batch_size, n_input_dims], device=device, dtype=torch.float32)
        targets = traced_image(batch)  # [B,1]
        preds = model(batch)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Capture gradient norm (L2) and LR
        if args.diagnostics:
            total_g2 = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_g2 += float(p.grad.data.norm(2).item() ** 2)
            grad_norms.append(math.sqrt(total_g2))
            lrs.append(float(optimizer.param_groups[0]['lr']))

        # EMA update
        if model_ema is not None:
            with torch.no_grad():
                beta = args.ema_decay
                for p, pe in zip(model.parameters(), model_ema.parameters()):
                    pe.data.mul_(beta).add_(p.data, alpha=1.0 - beta)

        losses.append(float(loss.item()))

        # Validation every 50 steps
        if step % (args.log_interval) == 0:
            eval_model = model_ema if model_ema is not None else model
            eval_model.eval()
            with torch.no_grad():
                vpred = eval_model(val_coords)
                vtarget = image(val_coords)
                vloss = loss_fn(vpred, vtarget).item()
            val_losses.append((step, vloss))
            if holdout_coords is not None:
                with torch.no_grad():
                    hpred = eval_model(holdout_coords)
                    htarget = image(holdout_coords)
                    hloss = loss_fn(hpred, htarget).item()
                holdout_losses.append((step, hloss))
            if args.scheduler == 'plateau':
                scheduler.step(vloss)
            else:
                scheduler.step()

            # Track best
            if vloss < best_val - 1e-8:
                best_val = vloss
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in (eval_model.state_dict()).items()}

            # Memory tracking
            if device.type == 'cuda':
                cur = torch.cuda.memory_allocated(device) / (1024**3)
                peak_mem = max(peak_mem, cur)

            extra = f" | holdout={hloss:.6e}" if holdout_coords is not None else ""
            print(f"Step {step:6d} | train_mse={loss.item():.6e} | val_mse={vloss:.6e}{extra} | lr={optimizer.param_groups[0]['lr']:.2e}")

            # BF/ADF tracking
            if central_tile_coords is not None and args.track_bf_adf:
                with torch.no_grad():
                    # predict full central scan tile (tH, tW, A1, A2)
                    pred_tile = eval_model(central_tile_coords)  # [tH*tW*A1*A2,1]
                    if pred_tile.dim() == 2 and pred_tile.shape[1] == 1:
                        pred_tile = pred_tile.squeeze(1)
                    # reshape
                    try:
                        pred_tile = pred_tile.view(tH, tW, A1, A2)
                        pred_tile_mean = pred_tile.mean(0, keepdim=False).mean(0)  # (A1,A2)
                        bf_val = float((pred_tile_mean * bf_mask).sum().item() / (bf_mask.sum().item()+1e-8))
                        adf_val = float((pred_tile_mean * adf_mask).sum().item() / (adf_mask.sum().item()+1e-8))
                        bf_adf_track.append((step, bf_val, adf_val))
                    except Exception as _e:
                        print(f"[track_bf_adf] reshape failed: {_e}")

            # Residual spectrum
            if args.residual_spectrum_interval > 0 and (step % args.residual_spectrum_interval == 0) and central_tile_coords is not None:
                import numpy as _np
                with torch.no_grad():
                    preds_tile = eval_model(central_tile_coords)
                    if preds_tile.dim() == 2 and preds_tile.shape[1] == 1:
                        preds_tile = preds_tile.squeeze(1)
                    try:
                        preds_tile = preds_tile.view(tH, tW, A1, A2).mean(0, keepdim=False).mean(0)  # (A1,A2)
                        noisy_block = image.data[H//2 - tH//2:H//2 - tH//2 + tH, W//2 - tW//2:W//2 - tW//2 + tW]  # (tH,tW,A1,A2,1)
                        noisy_tile = noisy_block.mean(0, keepdim=False).mean(0).squeeze(-1)  # (A1,A2)
                        residual = (preds_tile - noisy_tile).detach().cpu().float().numpy()
                    except Exception as _e:
                        print(f"[residual_spectrum] reshape failed: {_e}")
                        residual = None
                    # FFT
                    if residual is not None:
                        fft = _np.fft.fftshift(_np.fft.fftn(residual))
                        power = _np.abs(fft)**2
                        cy2, cx2 = power.shape[0]//2, power.shape[1]//2
                        yy2, xx2 = _np.ogrid[:power.shape[0], :power.shape[1]]
                        r2 = _np.sqrt((yy2-cy2)**2 + (xx2-cx2)**2)
                        r2n = r2 / r2.max()
                        radial_bins = _np.linspace(0,1,48)
                        radial_power = _np.zeros(len(radial_bins)-1, dtype=_np.float32)
                        for bi in range(len(radial_bins)-1):
                            m = (r2n >= radial_bins[bi]) & (r2n < radial_bins[bi+1])
                            if m.any():
                                radial_power[bi] = float(power[m].mean())
                        residual_psd.append({"step": step, "radial_center": radial_bin_centers.tolist(), "radial_power": radial_power.tolist()})
                        print(f"[residual_psd] logged step {step} (power mean={power.mean():.3e})")

            # Early stopping relative improvement
            if args.early_stop_rel_improve > 0 and step >= args.early_stop_window:
                # compute relative improvement over window
                recent = [v for s,v in val_losses if s > step - args.early_stop_window]
                if len(recent) > 5:
                    first = recent[0]
                    last = recent[-1]
                    rel_impr = (first - last) / max(first, 1e-12)
                    if rel_impr < args.early_stop_rel_improve:
                        print(f"Early stopping (relative improvement {rel_impr:.4f} < threshold {args.early_stop_rel_improve}) at step {step}")
                        break

        # Early stopping
        if step - best_step > args.patience:
            print(f"Early stopping at step {step} (no val improvement for {args.patience} steps). Best at {best_step} with val_mse={best_val:.6e}")
            break

        # Snapshot export outside the log_interval block so it can be higher frequency if desired
        if args.snapshot_interval > 0 and (step % args.snapshot_interval == 0):
            try:
                eval_model = model_ema if model_ema is not None else model
                eval_model.eval()
                cx, cy = H // 2, W // 2
                dx = min(16, H)
                dy = min(16, W)
                x0 = max(0, cx - dx//2)
                y0 = max(0, cy - dy//2)
                xs = torch.linspace(0.5/H, 1-0.5/H, H, device=device)[x0:x0+dx]
                ys = torch.linspace(0.5/W, 1-0.5/W, W, device=device)[y0:y0+dy]
                zs = torch.linspace(0.5/A1, 1-0.5/A1, A1, device=device)
                ws = torch.linspace(0.5/A2, 1-0.5/A2, A2, device=device)
                xv, yv, zv, wv = torch.meshgrid(xs, ys, zs, ws, indexing='ij')
                coords = torch.stack((xv.flatten(), yv.flatten(), zv.flatten(), wv.flatten()), dim=1)
                preds = []
                for s in range(0, coords.shape[0], 1_000_000):
                    preds.append(eval_model(coords[s:s+1_000_000]))
                block = torch.cat(preds, dim=0).view(dx, dy, A1, A2)
                if args.clamp_output == "sigmoid":
                    block = block.clamp(min=0.0, max=1.0)
                elif args.clamp_output == "relu":
                    block = torch.relu(block)
                snap_dir = os.path.join(args.out_dir, "snapshots")
                os.makedirs(snap_dir, exist_ok=True)
                snap_path = os.path.join(snap_dir, f"{base_name}_snap_step{step:06d}.npy")
                np.save(snap_path, block.detach().float().cpu().numpy().astype(np.float16))
                # Prune if exceeding max_snapshots
                snaps = sorted([f for f in os.listdir(snap_dir) if f.startswith(base_name+"_snap_step") and f.endswith('.npy')])
                if len(snaps) > args.max_snapshots:
                    to_del = snaps[:-args.max_snapshots]
                    for td in to_del:
                        try:
                            os.remove(os.path.join(snap_dir, td))
                        except OSError:
                            pass
                print(f"[snapshot] Saved {snap_path}")
            except Exception as e:
                print(f"[snapshot] Failed: {e}")

    t_end = time.perf_counter()
    print("== Training summary ==")
    print(f"Steps run: {step+1}")
    print(f"Best step: {best_step}  best val MSE: {best_val:.6e}")
    print(f"Time: {t_end - t_start:.2f} s  (~{(t_end - t_start)/60:.2f} min)")
    if device.type == 'cuda':
        print(f"Peak GPU memory: {peak_mem:.3f} GB")

    # Load best weights
    if best_state is not None and not args.export_only:
        # Load best into both model and EMA (if present) for export
        if model_ema is not None:
            model_ema.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)

    # Save checkpoint if requested
    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(resolved_npy))[0]
    if args.save_ckpt and not args.export_only:
        ckpt_out = args.ckpt_path or os.path.join(args.out_dir, f"{base_name}_best.pth")
        to_save = {
            'state_dict': (model_ema or model).state_dict(),
            'encoding': config['encoding'],
            'network': config['network'],
        }
        if model_ema is not None:
            to_save['ema_state_dict'] = model_ema.state_dict()
        torch.save(to_save, ckpt_out)
        print(f"Saved checkpoint: {ckpt_out}")
    else:
        ckpt_out = None

    # Prepare output dir (may already exist)
    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(resolved_npy))[0]

    # Export small diagnostic slices (safe and quick)
    diag_out = os.path.join(args.out_dir, f"{base_name}_denoised_diag.npy")
    print("Exporting diagnostic denoised slices (center scan tile, full angles)...")
    with torch.no_grad():
        cx, cy = H // 2, W // 2
        dx = min(16, H)
        dy = min(16, W)
        x0 = max(0, cx - dx//2)
        y0 = max(0, cy - dy//2)
        xs = torch.linspace(0.5/H, 1-0.5/H, H, device=device)[x0:x0+dx]
        ys = torch.linspace(0.5/W, 1-0.5/W, W, device=device)[y0:y0+dy]
        zs = torch.linspace(0.5/A1, 1-0.5/A1, A1, device=device)
        ws = torch.linspace(0.5/A2, 1-0.5/A2, A2, device=device)

        xv, yv, zv, wv = torch.meshgrid(xs, ys, zs, ws, indexing='ij')
        coords = torch.stack((xv.flatten(), yv.flatten(), zv.flatten(), wv.flatten()), dim=1)
        preds = []
        use_model = model_ema if model_ema is not None else model
        for s in range(0, coords.shape[0], 1_000_000):
            preds.append(use_model(coords[s:s+1_000_000]))
        block = torch.cat(preds, dim=0).view(dx, dy, A1, A2)
        if args.clamp_output == "sigmoid":
            block = block.clamp(min=0.0, max=1.0)
        elif args.clamp_output == "relu":
            block = torch.relu(block)
        # none -> no clamp
        if args.export_in_counts:
            block = block * image.normalization_scale
        np.save(diag_out, block.detach().float().cpu().numpy().astype(np.float16))
    print(f"Saved diagnostic block: {diag_out}")
    # Quick visualizations on raw data (before full export) if requested
    if args.quick_viz:
        try:
            raw_arr = np.load(resolved_npy, mmap_mode='r')
            viz_dir = os.path.join(args.out_dir, f"{base_name}_quick_viz")
            generate_quick_visuals(raw_arr, viz_dir, bf_frac=args.bf_frac, adf_min_frac=args.adf_min_frac, adf_max_frac=args.adf_max_frac)
        except Exception as e:
            print(f"[quick_viz] Failed: {e}")

    # Optional: full export (HUGE). Uses memmap and chunked inference.
    if args.export_full:
        suffix = "counts" if args.export_in_counts else "norm"
        full_out = os.path.join(args.out_dir, f"{base_name}_denoised_full_{suffix}.npy")
        print(f"Exporting FULL denoised volume to {full_out} (this may take hours and many GB)...")
        # Export in normalized units; if counts requested, rescale after loading tile
        tmp_out = full_out
        chunked_inference(
            model_ema if model_ema is not None else model,
            resolution=(H, W, A1, A2),
            device=device,
            out_dtype=torch.float16,
            scan_block=tuple(args.scan_block),
            batch_points=args.batch_points,
            out_path=tmp_out,
            normalization_scale=image.normalization_scale,
            clamp_mode=args.clamp_output,
        )
        if args.export_in_counts:
            # Post-scale entire file in-place (stream through chunks to avoid RAM blowup)
            import numpy as _np
            arr = _np.load(tmp_out, mmap_mode='r+')
            arr[:] = (arr * image.normalization_scale).astype(arr.dtype)
            del arr
        print(f"Full volume saved: {full_out}")

    # Save a small JSON with run info
    meta = {
        "npy": args.npy,
        "shape": [int(H), int(W), int(A1), int(A2)],
        "normalization": {
            "type": args.norm,
            "clip_percentile": args.clip_pct,
            "scale": float(image.normalization_scale),
        },
        "training": {
            "steps_run": int(step+1),
            "best_step": int(best_step),
            "best_val_mse": float(best_val),
            "batch": int(batch_size),
            "val_frac": float(args.val_frac),
            "patience": int(args.patience),
            "loss": args.loss,
            "lr": args.lr,
            "grad_clip": float(args.grad_clip),
            "scheduler": args.scheduler,
            "ema": bool(args.ema),
            "ema_decay": float(args.ema_decay),
        },
        "export_units": "counts" if args.export_in_counts else "normalized",
        "encoding": config["encoding"],
        "network": config["network"],
        "device": str(device),
        "model_parameters": int(param_count),
    }
    # Add compression info if requested and checkpoint saved
    if args.report_compression and ckpt_out is not None:
        add_compression_info(meta, raw_path=args.npy, ckpt_path=ckpt_out)

    # Diagnostics export (after meta augmentation so ratio can be available)
    if args.diagnostics:
        diag_dir = args.diag_dir or args.out_dir
        os.makedirs(diag_dir, exist_ok=True)
        csv_path = os.path.join(diag_dir, f"{base_name}_training_log.csv")
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(["step","train_loss","val_loss","lr","grad_norm"])
            val_dict = {s:v for s,v in val_losses}
            for s, t_loss in enumerate(losses):
                writer.writerow([s, t_loss, val_dict.get(s, ''), lrs[s] if s < len(lrs) else '', grad_norms[s] if s < len(grad_norms) else ''])
        print(f"[diagnostics] Wrote CSV: {csv_path}")
        try:
            import matplotlib.pyplot as _plt
            fig, axs = _plt.subplots(2,2, figsize=(10,8))
            axs[0,0].plot(range(len(losses)), losses, label='train')
            axs[0,0].set_title('Training Loss'); axs[0,0].set_yscale('log'); axs[0,0].grid(True, alpha=0.3)
            if val_losses:
                vs, vv = zip(*val_losses)
                axs[0,1].plot(vs, vv, 'o-', label='val')
            axs[0,1].set_title('Validation Loss'); axs[0,1].set_yscale('log'); axs[0,1].grid(True, alpha=0.3)
            if lrs:
                axs[1,0].plot(range(len(lrs)), lrs)
            axs[1,0].set_title('Learning Rate'); axs[1,0].grid(True, alpha=0.3)
            if grad_norms:
                axs[1,1].plot(range(len(grad_norms)), grad_norms)
            axs[1,1].set_title('Gradient Norm'); axs[1,1].grid(True, alpha=0.3)
            for ax in axs.ravel():
                ax.set_xlabel('Step')
            fig.suptitle(f"Training Diagnostics: {base_name}")
            fig.tight_layout(rect=[0,0,1,0.96])
            plot_path = os.path.join(diag_dir, f"{base_name}_training_diagnostics.png")
            fig.savefig(plot_path, dpi=140)
            _plt.close(fig)
            print(f"[diagnostics] Saved plot: {plot_path}")
        except Exception as e:
            print(f"[diagnostics] Plotting failed: {e}")
        # Save holdout, BF/ADF, PSD data if present
        import json as _json
        if holdout_losses:
            with open(os.path.join(diag_dir, f"{base_name}_holdout_losses.json"), 'w') as fhl:
                _json.dump({"holdout_losses": holdout_losses}, fhl)
        if bf_adf_track:
            with open(os.path.join(diag_dir, f"{base_name}_bf_adf_track.csv"), 'w', newline='') as fb:
                import csv as _csv
                w = _csv.writer(fb); w.writerow(["step","bf_mean","adf_mean"]); w.writerows(bf_adf_track)
        if residual_psd:
            with open(os.path.join(diag_dir, f"{base_name}_residual_psd.json"), 'w') as fr:
                _json.dump(residual_psd, fr)

    # Radial profile at end (raw vs predicted central mean DP)
    if args.radial_profile:
        try:
            import numpy as _np
            raw_arr = np.load(resolved_npy, mmap_mode='r')
            mean_raw = raw_arr.mean(axis=(0,1))
            # predict central tile & average predictions across scan tile
            eval_model = model_ema if model_ema is not None else model
            if central_tile_coords is None:
                # fallback to using full mean prediction (costly), skip
                pass
            else:
                with torch.no_grad():
                    pred_tile = eval_model(central_tile_coords)
                    if pred_tile.dim() == 2 and pred_tile.shape[1] == 1:
                        pred_tile = pred_tile.squeeze(1)
                    try:
                        pred_tile = pred_tile.view(tH, tW, A1, A2).mean(0, keepdim=False).mean(0).detach().cpu().numpy()
                    except Exception as _e:
                        print(f"[radial_profile] reshape failed: {_e}")
                        pred_tile = None
            cy, cx = mean_raw.shape[0]//2, mean_raw.shape[1]//2
            yy, xx = _np.ogrid[:mean_raw.shape[0], :mean_raw.shape[1]]
            r = _np.sqrt((yy-cy)**2 + (xx-cx)**2)
            rn = r / r.max()
            rbins = _np.linspace(0,1,64)
            centers = 0.5*(rbins[:-1]+rbins[1:])
            raw_prof = []
            pred_prof = []
            for bi in range(len(rbins)-1):
                m = (rn>=rbins[bi]) & (rn<rbins[bi+1])
                if m.any():
                    raw_prof.append(float(mean_raw[m].mean()))
                    if central_tile_coords is not None and pred_tile is not None:
                        pred_prof.append(float(pred_tile[m].mean()))
            rp = {
                "radial_center": centers[:len(raw_prof)].tolist(),
                "raw_profile": raw_prof,
                "pred_profile": pred_prof if pred_prof else None,
            }
            with open(os.path.join(args.out_dir, f"{base_name}_radial_profile.json"), 'w') as frp:
                import json as _json
                _json.dump(rp, frp)
            print("Saved radial profile JSON.")
        except Exception as e:
            print(f"[radial_profile] Failed: {e}")
    meta_out = os.path.join(args.out_dir, f"{base_name}_denoise_meta.json")
    with open(meta_out, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved run metadata: {meta_out}")


if __name__ == "__main__":
    main()
