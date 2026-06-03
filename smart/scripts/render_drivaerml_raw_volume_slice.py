#!/usr/bin/env python3
"""Render a full-raw-query DrivAerML volume slice for one run.

This script:
1) Uses preprocessed surface points for SMART/CAT inputs and CAT stage-1/2 surface context.
2) Uses the raw volume H5 file as the full query set and raw ground truth source.
3) Streams all raw volume points chunk-by-chunk to avoid loading the full volume into RAM.
4) Produces one PNG with y=0 slice views for pressure and velocity magnitude only.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2
from models.smart.cat import CAT
from models.smart.smart import SMART
from utils.utils import get_model_checkpoint_name


SURFACE_FIELDS = [
    "pressure",
    "normal_x",
    "normal_y",
    "normal_z",
    "wall_shear_x",
    "wall_shear_y",
    "wall_shear_z",
]
VOLUME_FIELDS = ["pressure", "velocity_x", "velocity_y", "velocity_z"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render raw-query DrivAerML volume slices for one run.")
    p.add_argument("--run-id", type=int, required=True, help="Run id to render.")
    p.add_argument("--smart-config", default="drivaerml")
    p.add_argument("--cat-config", default="drivaerml_cat")
    p.add_argument("--smart-checkpoint", default=None)
    p.add_argument("--cat-stage1-checkpoint", required=True)
    p.add_argument("--cat-stage2-checkpoint", required=True)
    p.add_argument("--raw-root", default="/mnt/ssdraid/drivaer_data")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smart-vol-chunk", type=int, default=262144)
    p.add_argument("--cat-surf-chunk", type=int, default=262144)
    p.add_argument("--cat-vol-chunk", type=int, default=262144)
    p.add_argument("--raw-read-chunk", type=int, default=262144)
    p.add_argument("--nx", type=int, default=3200, help="Slice pixels in x.")
    p.add_argument("--nz", type=int, default=800, help="Slice pixels in z.")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def load_cfg(name: str):
    path = SMART_ROOT / "config" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path).experiment


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_ckpt(config, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    stem = get_model_checkpoint_name(config)
    best = SMART_ROOT.parent / "checkpoints" / f"{stem}_best.pt"
    last = SMART_ROOT.parent / "checkpoints" / f"{stem}_last.pt"
    if best.is_file():
        return str(best)
    if last.is_file():
        return str(last)
    raise FileNotFoundError(f"SMART checkpoint not found: {best} or {last}")


def normalize_pos(pos: torch.Tensor, min_pos: torch.Tensor, max_pos: torch.Tensor) -> torch.Tensor:
    return (pos - min_pos) / torch.clamp(max_pos - min_pos, min=1e-12)


def denorm_fields(pred_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return pred_norm * std + mean


def sample_input_idx(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    return rng.integers(0, n, size=k, dtype=np.int64)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def setup_plot_style():
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
            "axes.grid": False,
        }
    )


class VolumeSliceAccumulator:
    def __init__(self, x_min: float, x_max: float, z_min: float, z_max: float, nx: int, nz: int):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.nx = int(nx)
        self.nz = int(nz)
        self.x_range = max(self.x_max - self.x_min, 1e-12)
        self.z_range = max(self.z_max - self.z_min, 1e-12)
        self.dx = self.x_range / max(self.nx - 1, 1)
        self.dz = self.z_range / max(self.nz - 1, 1)
        self.radius = 0.02
        self.y_tol = 0.02
        self.count_grid = np.zeros((self.nz, self.nx), dtype=np.float64)
        self.sum_grids = {
            "gt_pressure": np.zeros((self.nz, self.nx), dtype=np.float64),
            "cat_pressure": np.zeros((self.nz, self.nx), dtype=np.float64),
            "smart_pressure": np.zeros((self.nz, self.nx), dtype=np.float64),
            "gt_velocity_mag": np.zeros((self.nz, self.nx), dtype=np.float64),
            "cat_velocity_mag": np.zeros((self.nz, self.nx), dtype=np.float64),
            "smart_velocity_mag": np.zeros((self.nz, self.nx), dtype=np.float64),
        }

    def update(
        self,
        coords: np.ndarray,
        gt_pressure: np.ndarray,
        cat_pressure: np.ndarray,
        smart_pressure: np.ndarray,
        gt_velocity_mag: np.ndarray,
        cat_velocity_mag: np.ndarray,
        smart_velocity_mag: np.ndarray,
    ) -> None:
        x_all = coords[:, 0]
        y_all = coords[:, 1]
        z_all = coords[:, 2]
        mask = (
            (x_all >= self.x_min)
            & (x_all <= self.x_max)
            & (z_all >= self.z_min)
            & (z_all <= self.z_max)
            & (y_all >= -self.y_tol)
            & (y_all <= self.y_tol)
        )
        if not np.any(mask):
            return

        xyz = coords[mask]
        x = xyz[:, 0]
        z = xyz[:, 2]
        ix = np.clip(np.rint((x - self.x_min) / max(self.dx, 1e-12)).astype(np.int64), 0, self.nx - 1)
        iz = np.clip(np.rint((z - self.z_min) / max(self.dz, 1e-12)).astype(np.int64), 0, self.nz - 1)
        flat = iz * self.nx + ix
        bins = self.nz * self.nx

        self.count_grid += np.bincount(flat, minlength=bins).reshape(self.nz, self.nx)
        for key, values in [
            ("gt_pressure", gt_pressure[mask]),
            ("cat_pressure", cat_pressure[mask]),
            ("smart_pressure", smart_pressure[mask]),
            ("gt_velocity_mag", gt_velocity_mag[mask]),
            ("cat_velocity_mag", cat_velocity_mag[mask]),
            ("smart_velocity_mag", smart_velocity_mag[mask]),
        ]:
            self.sum_grids[key] += np.bincount(flat, weights=values.astype(np.float64, copy=False), minlength=bins).reshape(self.nz, self.nx)

    def _disk_kernel(self) -> torch.Tensor:
        rx = max(1, int(math.ceil(self.radius / max(self.dx, 1e-12))))
        rz = max(1, int(math.ceil(self.radius / max(self.dz, 1e-12))))
        xs = np.arange(-rx, rx + 1, dtype=np.float32) * self.dx
        zs = np.arange(-rz, rz + 1, dtype=np.float32) * self.dz
        dist2 = zs[:, None] ** 2 + xs[None, :] ** 2
        kernel = (dist2 <= self.radius * self.radius).astype(np.float32)
        return torch.from_numpy(kernel)[None, None, :, :]

    def finalize(self) -> Tuple[dict[str, np.ndarray], float]:
        kernel = self._disk_kernel()
        pad_h = kernel.shape[-2] // 2
        pad_w = kernel.shape[-1] // 2

        count_t = torch.from_numpy(self.count_grid.astype(np.float32, copy=False))[None, None, :, :]
        smooth_count = F.conv2d(count_t, kernel, padding=(pad_h, pad_w))[0, 0].numpy()

        out = {}
        for key, grid in self.sum_grids.items():
            sum_t = torch.from_numpy(grid.astype(np.float32, copy=False))[None, None, :, :]
            smooth_sum = F.conv2d(sum_t, kernel, padding=(pad_h, pad_w))[0, 0].numpy()
            avg = np.zeros_like(smooth_sum, dtype=np.float32)
            valid = smooth_count > 0
            avg[valid] = smooth_sum[valid] / smooth_count[valid]
            out[key] = avg
        return out, self.radius


def scan_volume_extents(volume_h5_path: Path, chunk_rows: int) -> Tuple[float, float, float, float]:
    x_min = -2.0
    x_max = 8.0
    z_min = -0.4
    z_max = 1.5
    with h5py.File(volume_h5_path, "r") as hv:
        coords_ds = hv["coords"]
        n = int(coords_ds.shape[0])
        for start in tqdm(range(0, n, chunk_rows), desc="Scanning raw volume extents", unit="chunk"):
            coords = np.asarray(coords_ds[start : start + chunk_rows], dtype=np.float32)
    return x_min, x_max, z_min, z_max


def prepare_smart_context(model: SMART, geo_norm: torch.Tensor, device: torch.device):
    with torch.inference_mode():
        b_geo = geo_norm.unsqueeze(0).to(device)
        inter_latents, latent_geo_pos = model.encode(b_geo, None)
    return inter_latents, latent_geo_pos


def predict_smart_volume_chunk(
    model: SMART,
    inter_latents,
    latent_geo_pos,
    vol_query_norm: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    preds = []
    with torch.inference_mode():
        n = int(vol_query_norm.shape[0])
        for start in range(0, n, int(chunk_size)):
            q = vol_query_norm[start : start + int(chunk_size)].unsqueeze(0).to(device)
            pred = model.decode(inter_latents, latent_geo_pos, None, q)
            preds.append(pred[0, :, model.surface_channels :].detach().cpu())
    return torch.cat(preds, dim=0)


def prepare_cat_stage2_context(
    model: CAT,
    surf_input_norm: torch.Tensor,
    surf_query_norm: torch.Tensor,
    chunk_size: int,
    device: torch.device,
):
    with torch.inference_mode():
        b_input = surf_input_norm.unsqueeze(0).to(device)
        input_pos = b_input[..., : model.spatial_dim]
        geom_latents, anchor_pos, geom_final = model._encode_stage1(input_pos)

        surf_preds = []
        n = surf_query_norm.shape[0]
        for start in range(0, n, chunk_size):
            q = surf_query_norm[start : start + chunk_size].unsqueeze(0).to(device)
            q_emb = model._decode(q[..., : model.spatial_dim], geom_latents, anchor_pos, model.surface_decoder_blocks)
            surf_preds.append(model.stage2_head(q_emb)[0].detach().cpu())
        surface_pred = torch.cat(surf_preds, dim=0).unsqueeze(0).to(device)
        b_surf_q = surf_query_norm.unsqueeze(0).to(device)

        prev_latents, _ = model._encode_stage2(b_surf_q[..., : model.spatial_dim], surface_pred, anchor_pos, initial_latent=geom_final)
        new_latents, _ = model._encode_stage2(b_surf_q[..., : model.spatial_dim], surface_pred, anchor_pos, initial_latent=None)
        w_couple, w_fuse = model._compute_dynamic_skip_weights(geom_latents, prev_latents, new_latents, surface_pred)

        fused_latents = []
        for m in range(model.loops):
            wc = w_couple[:, m, :].unsqueeze(-1)
            wf = w_fuse[:, m, :].unsqueeze(-1)
            geom_m = model.surface_to_volume_latent_norm(geom_latents[m])
            prev_m = model.surface_to_volume_latent_norm(prev_latents[m])
            new_m = model.surface_to_volume_latent_norm(new_latents[m])
            coupled = prev_m + wc * (geom_m - prev_m)
            fused = new_m + wf * (coupled - new_m)
            fused_latents.append(fused)

    return fused_latents, anchor_pos


def predict_cat_volume_chunk(
    model: CAT,
    fused_latents,
    anchor_pos,
    vol_query_norm: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    preds = []
    with torch.inference_mode():
        n = int(vol_query_norm.shape[0])
        for start in range(0, n, int(chunk_size)):
            q = vol_query_norm[start : start + int(chunk_size)].unsqueeze(0).to(device)
            qv = model._decode(q[..., : model.spatial_dim], fused_latents, anchor_pos, model.volume_decoder_blocks)
            preds.append(model.volume_head(qv)[0].detach().cpu())
    return torch.cat(preds, dim=0)


def plot_volume_slice_grid(
    out_path: Path,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    slices: dict[str, np.ndarray],
    radius: float,
) -> None:
    extent = [x_min, x_max, z_min, z_max]

    pressure_stack = np.concatenate(
        [
            slices["gt_pressure"].reshape(-1),
            slices["cat_pressure"].reshape(-1),
            slices["smart_pressure"].reshape(-1),
        ]
    )
    velocity_stack = np.concatenate(
        [
            slices["gt_velocity_mag"].reshape(-1),
            slices["cat_velocity_mag"].reshape(-1),
            slices["smart_velocity_mag"].reshape(-1),
        ]
    )
    pressure_vmin = float(np.percentile(pressure_stack, 1))
    pressure_vmax = float(np.percentile(pressure_stack, 99))
    velocity_vmin = float(np.percentile(velocity_stack, 1))
    velocity_vmax = float(np.percentile(velocity_stack, 99))

    pressure_err_cat = np.abs(slices["cat_pressure"] - slices["gt_pressure"])
    pressure_err_smart = np.abs(slices["smart_pressure"] - slices["gt_pressure"])
    velocity_err_cat = np.abs(slices["cat_velocity_mag"] - slices["gt_velocity_mag"])
    velocity_err_smart = np.abs(slices["smart_velocity_mag"] - slices["gt_velocity_mag"])
    pressure_evmax = float(max(np.percentile(pressure_err_cat, 99), np.percentile(pressure_err_smart, 99)))
    velocity_evmax = float(max(np.percentile(velocity_err_cat, 99), np.percentile(velocity_err_smart, 99)))
    pressure_evmax = pressure_evmax if pressure_evmax > 0 else 1e-12
    velocity_evmax = velocity_evmax if velocity_evmax > 0 else 1e-12

    fig, axes = plt.subplots(2, 5, figsize=(24, 9), constrained_layout=True)
    fig.suptitle(
        "DrivAerML Raw-Query Volume Slice at y=0\n"
        f"Full raw volume queries, preprocessed surface context, 800x200 slice, gather radius={radius:.4f}"
    )

    rows = [
        (
            "pressure",
            slices["gt_pressure"],
            slices["cat_pressure"],
            slices["smart_pressure"],
            pressure_err_cat,
            pressure_err_smart,
            pressure_vmin,
            pressure_vmax,
            pressure_evmax,
        ),
        (
            "velocity_mag",
            slices["gt_velocity_mag"],
            slices["cat_velocity_mag"],
            slices["smart_velocity_mag"],
            velocity_err_cat,
            velocity_err_smart,
            velocity_vmin,
            velocity_vmax,
            velocity_evmax,
        ),
    ]

    for row_idx, (field, gt, cat, smart, err_cat, err_smart, vmin, vmax, evmax) in enumerate(rows):
        panels = [
            (gt, "GT", "coolwarm", vmin, vmax),
            (cat, "CAT-S2", "coolwarm", vmin, vmax),
            (smart, "SMART", "coolwarm", vmin, vmax),
            (err_cat, "|CAT-S2 - GT|", "magma", 0.0, evmax),
            (err_smart, "|SMART - GT|", "magma", 0.0, evmax),
        ]
        for col_idx, (img, subtitle, cmap, pvmin, pvmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(img, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmin=pvmin, vmax=pvmax)
            ax.set_title(f"{field} - {subtitle}")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_aspect("equal", adjustable="box")
            cbar = fig.colorbar(im, ax=ax, shrink=0.82)
            cbar.ax.tick_params(labelsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    setup_plot_style()

    smart_cfg = load_cfg(args.smart_config)
    cat_cfg = load_cfg(args.cat_config)
    device = resolve_device(args.device)
    run_id = int(args.run_id)

    dataset = AhmedMLDatasetV2(
        saved_folder=str(smart_cfg.data_path),
        if_test=False,
        geometry_points=int(smart_cfg.num_body_points),
        surface_points=int(smart_cfg.num_surface_points),
        volume_points=int(smart_cfg.num_volume_points),
        scale_positions=bool(smart_cfg.scale_positions),
        require_preprocessed=True,
    )
    mean_v = dataset.mean_vol_data
    std_v = torch.clamp(dataset.std_vol_data, min=1e-12)
    min_pos = dataset.min_pos
    max_pos = dataset.max_pos

    pre_dir = Path(smart_cfg.data_path) / f"run_{run_id}"
    if not pre_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed run folder not found: {pre_dir}")
    raw_run_dir = Path(args.raw_root) / f"run_{run_id}"
    raw_volume_h5 = raw_run_dir / f"volume_{run_id}_filtered.h5"
    if not raw_volume_h5.is_file():
        raise FileNotFoundError(f"Raw volume H5 not found: {raw_volume_h5}")

    surface_coords = np.load(pre_dir / "surface_coords.npy", mmap_mode="r")
    surface_coords_t = torch.from_numpy(np.asarray(surface_coords, dtype=np.float32))
    surf_query_norm = normalize_pos(surface_coords_t, min_pos, max_pos)

    rng = np.random.default_rng(args.seed + run_id)
    smart_geo_idx = sample_input_idx(surface_coords.shape[0], int(smart_cfg.num_body_points), rng)
    cat_input_k = int(getattr(cat_cfg, "single_surface_input_points", cat_cfg.num_body_points))
    cat_in_idx = sample_input_idx(surface_coords.shape[0], cat_input_k, rng)
    smart_geo_norm = surf_query_norm[torch.from_numpy(smart_geo_idx)]
    cat_input_norm = surf_query_norm[torch.from_numpy(cat_in_idx)]

    smart_arch = OmegaConf.to_container(smart_cfg.architecture, resolve=True)
    smart_model = SMART(
        spatial_dim=3,
        surface_channels=len(SURFACE_FIELDS),
        volume_channels=len(VOLUME_FIELDS),
        parameter_channels=0,
        **smart_arch,
    ).to(device)
    smart_ckpt_path = choose_ckpt(smart_cfg, args.smart_checkpoint)
    smart_ckpt = torch.load(smart_ckpt_path, map_location=device)
    smart_model.load_state_dict(smart_ckpt["model_state_dict"], strict=True)
    smart_model.eval()

    cat_arch = OmegaConf.to_container(cat_cfg.architecture, resolve=True)
    cat_arch["stage2_surface_channels"] = len(SURFACE_FIELDS)

    cat_stage2 = CAT(
        spatial_dim=3,
        surface_channels=len(SURFACE_FIELDS),
        volume_channels=len(VOLUME_FIELDS),
        parameter_channels=0,
        **cat_arch,
    ).to(device)
    ckpt_s2 = torch.load(args.cat_stage2_checkpoint, map_location=device)
    cat_stage2.load_state_dict(ckpt_s2["model_state_dict"], strict=True)
    cat_stage2.eval()

    # Validate stage-1 checkpoint path explicitly even though stage-2 weights already contain stage-1 modules.
    if not Path(args.cat_stage1_checkpoint).is_file():
        raise FileNotFoundError(f"CAT stage1 checkpoint not found: {args.cat_stage1_checkpoint}")

    print(f"Device: {device}")
    print(f"Run id: {run_id}")
    print("Preparing model contexts from preprocessed surface points...")
    smart_inter_latents, smart_latent_geo_pos = prepare_smart_context(smart_model, smart_geo_norm, device)
    cat_fused_latents, cat_anchor_pos = prepare_cat_stage2_context(
        cat_stage2,
        cat_input_norm,
        surf_query_norm,
        chunk_size=int(args.cat_surf_chunk),
        device=device,
    )

    print("Scanning raw volume extents...")
    x_min, x_max, z_min, z_max = scan_volume_extents(raw_volume_h5, int(args.raw_read_chunk))
    accumulator = VolumeSliceAccumulator(x_min, x_max, z_min, z_max, int(args.nx), int(args.nz))

    print("Streaming raw volume queries and accumulating slice...")
    with h5py.File(raw_volume_h5, "r") as hv:
        coords_ds = hv["coords"]
        p_ds = hv["pMeanTrim"]
        u_ds = hv["UMeanTrim"]
        n = int(coords_ds.shape[0])
        for start in tqdm(range(0, n, int(args.raw_read_chunk)), desc="Streaming raw volume queries", unit="chunk"):
            end = min(start + int(args.raw_read_chunk), n)
            coords = np.asarray(coords_ds[start:end], dtype=np.float32)
            gt_pressure = np.asarray(p_ds[start:end], dtype=np.float32).reshape(-1)
            gt_velocity = np.asarray(u_ds[start:end], dtype=np.float32)
            gt_velocity_mag = np.linalg.norm(gt_velocity, axis=1)

            vol_query_norm = normalize_pos(torch.from_numpy(coords), min_pos, max_pos)
            smart_chunk_norm = predict_smart_volume_chunk(
                smart_model,
                smart_inter_latents,
                smart_latent_geo_pos,
                vol_query_norm,
                device,
                chunk_size=int(args.smart_vol_chunk),
            )
            cat_chunk_norm = predict_cat_volume_chunk(
                cat_stage2,
                cat_fused_latents,
                cat_anchor_pos,
                vol_query_norm,
                device,
                chunk_size=int(args.cat_vol_chunk),
            )

            smart_chunk = denorm_fields(smart_chunk_norm, mean_v, std_v).numpy()
            cat_chunk = denorm_fields(cat_chunk_norm, mean_v, std_v).numpy()
            smart_pressure = smart_chunk[:, 0]
            cat_pressure = cat_chunk[:, 0]
            smart_velocity_mag = np.linalg.norm(smart_chunk[:, 1:4], axis=1)
            cat_velocity_mag = np.linalg.norm(cat_chunk[:, 1:4], axis=1)

            accumulator.update(
                coords,
                gt_pressure,
                cat_pressure,
                smart_pressure,
                gt_velocity_mag,
                cat_velocity_mag,
                smart_velocity_mag,
            )

    slices, radius = accumulator.finalize()

    out_dir = Path(args.output_dir or (SMART_ROOT.parent / "results" / "drivaerml_raw_volume_slice" / f"run_{run_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "volume_slice_raw_queries.png"
    plot_volume_slice_grid(out_path, x_min, x_max, z_min, z_max, slices, radius)
    print(f"Saved slice image to: {out_path}")


if __name__ == "__main__":
    main()
