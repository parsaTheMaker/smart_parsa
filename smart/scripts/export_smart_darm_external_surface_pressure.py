#!/usr/bin/env python3
"""Run SMART and DARM on an external surface cloud and export pressure-only VTKs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SMART_ROOT.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.datasets import get_dataset
from models.smart.darm import DARM
from models.smart.smart import SMART


DEFAULT_SMART_CKPT = "/home/parsa/smart_parsa/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt"
DEFAULT_DARM_CKPT = "/home/parsa/smart_parsa/checkpoints/darm-darm-drivaerml-residualadapter-v2-drivaerml-s42_best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export SMART and DARM pressure predictions on an external surface point cloud.")
    p.add_argument(
        "--surface-dir",
        required=True,
        help="Directory containing surface_coords.npy and surface_pMeanTrim.npy, or a parent directory whose subfolders contain them.",
    )
    p.add_argument("--smart-config", default="drivaerml", help="SMART config name under smart/config without .yaml")
    p.add_argument("--darm-config", default="drivaerml_darm", help="DARM config name under smart/config without .yaml")
    p.add_argument("--smart-checkpoint", default=DEFAULT_SMART_CKPT, help="Path to SMART checkpoint.")
    p.add_argument("--darm-checkpoint", default=DEFAULT_DARM_CKPT, help="Path to DARM checkpoint.")
    p.add_argument("--seed", type=int, default=42, help="Seed for external encoder-input sampling.")
    p.add_argument("--surface-query-chunk", type=int, default=0, help="Surface chunk size. Default: training surface query budget.")
    p.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    p.add_argument(
        "--darm-outlier-abs-threshold",
        type=float,
        default=2.0e4,
        help="If |DARM pressure| exceeds this threshold anywhere, save debug outputs and fail.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: results/smart_darm_external_surface_pressure/<surface-dir-name>",
    )
    return p.parse_args()


def load_cfg(name: str):
    path = SMART_ROOT / "config" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path).experiment


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def write_polydata_vtk(path: Path, points_xyz: np.ndarray, point_data: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must have shape [N, 3]")
    n = pts.shape[0]
    connectivity = np.empty((n, 2), dtype=">i4")
    connectivity[:, 0] = 1
    connectivity[:, 1] = np.arange(n, dtype=np.int32)

    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(b"SMART/DARM external surface pressure export\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n".encode("ascii"))
        f.write(pts.astype(">f4", copy=False).tobytes())
        f.write(b"\n")
        f.write(f"VERTICES {n} {2 * n}\n".encode("ascii"))
        f.write(connectivity.tobytes())
        f.write(b"\n")
        f.write(f"POINT_DATA {n}\n".encode("ascii"))
        for name, arr in point_data.items():
            a = np.asarray(arr, dtype=np.float32).reshape(n, -1)
            nm = safe_name(name)
            f.write(f"SCALARS {nm} float {a.shape[1]}\n".encode("ascii"))
            f.write(b"LOOKUP_TABLE default\n")
            f.write(a.astype(">f4", copy=False).tobytes())
            f.write(b"\n")


def normalize_pos(pos: torch.Tensor, min_pos: torch.Tensor, max_pos: torch.Tensor) -> torch.Tensor:
    return (pos - min_pos) / torch.clamp(max_pos - min_pos, min=1.0e-12)


def denorm_fields(pred_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return pred_norm * std + mean


def sample_indices(num_items: int, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    if sample_count <= 0:
        return np.arange(num_items, dtype=np.int64)
    replace = sample_count > num_items
    if not replace and sample_count >= num_items:
        return np.arange(num_items, dtype=np.int64)
    if replace:
        return rng.integers(0, num_items, size=sample_count, dtype=np.int64)
    return rng.choice(num_items, size=sample_count, replace=False).astype(np.int64, copy=False)


def is_surface_dir(path: Path) -> bool:
    return (path / "surface_coords.npy").is_file() and (path / "surface_pMeanTrim.npy").is_file()


def resolve_surface_dirs(path: Path) -> list[Path]:
    if is_surface_dir(path):
        return [path]
    surface_dirs = [subdir for subdir in sorted(path.iterdir()) if subdir.is_dir() and is_surface_dir(subdir)]
    if surface_dirs:
        return surface_dirs
    raise FileNotFoundError(
        f"No valid surface directory found at {path}. "
        "Expected surface_coords.npy and surface_pMeanTrim.npy in the directory itself or in its immediate subfolders."
    )


def rel_l2(gt: np.ndarray, pred: np.ndarray, eps: float = 1.0e-12) -> float:
    num = float(np.linalg.norm(pred - gt))
    den = float(np.linalg.norm(gt))
    return num / max(den, eps)


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def build_model(model_cls, cfg, spatial_dim: int, surf_channels: int, vol_channels: int, params_dim: int, device: torch.device):
    model_kwargs = {
        "spatial_dim": spatial_dim,
        "surface_channels": surf_channels,
        "volume_channels": vol_channels,
        "parameter_channels": params_dim,
    }
    merged_kwargs = {**model_kwargs, **cfg.architecture} if "architecture" in cfg else model_kwargs
    model = model_cls(**merged_kwargs).to(device)
    return model, merged_kwargs


def run_smart_surface_inference(
    model: SMART,
    geo_norm: torch.Tensor,
    surf_query_norm: torch.Tensor,
    mean_surf: torch.Tensor,
    std_surf: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        geo_b = geo_norm.unsqueeze(0).to(device, non_blocking=True)
        intermediate_latent_geometries, latent_geo_pos = model.encode(geo_b, None)
        out_chunks = []
        for start in range(0, surf_query_norm.shape[0], chunk_size):
            end = min(start + chunk_size, surf_query_norm.shape[0])
            surf_chunk = surf_query_norm[start:end].unsqueeze(0).to(device, non_blocking=True)
            pred_norm = model.decode(intermediate_latent_geometries, latent_geo_pos, None, surf_chunk)
            pred = denorm_fields(pred_norm[0, :, : mean_surf.shape[0]].to(torch.float32).cpu(), mean_surf, std_surf)
            out_chunks.append(pred[:, 0:1].numpy())
    return np.concatenate(out_chunks, axis=0)


def run_darm_surface_inference(
    model: DARM,
    geo_norm: torch.Tensor,
    surf_query_norm: torch.Tensor,
    mean_surf: torch.Tensor,
    std_surf: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    model.subregion_size = int(chunk_size)
    if hasattr(model, "prediction_query_chunk_size"):
        model.prediction_query_chunk_size = min(int(getattr(model, "prediction_query_chunk_size", chunk_size)), int(chunk_size))
    empty_vol = geo_norm.new_zeros(1, 0, geo_norm.shape[-1])
    with torch.inference_mode():
        pred_norm, _ = model.inference(
            geo_norm.unsqueeze(0).to(device, non_blocking=True),
            surf_query_norm.unsqueeze(0).to(device, non_blocking=True),
            empty_vol.to(device, non_blocking=True),
            None,
            return_aux=False,
        )
        pred = denorm_fields(pred_norm[0].to(torch.float32).cpu(), mean_surf, std_surf)
    return pred[:, 0:1].numpy()


def run_darm_base_surface_inference(
    model: DARM,
    geo_norm: torch.Tensor,
    surf_query_norm: torch.Tensor,
    mean_surf: torch.Tensor,
    std_surf: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        geo_b = geo_norm.unsqueeze(0).to(device, non_blocking=True)
        intermediate_latent_geometries, latent_geo_pos, final_latent_geo = model.encode(geo_b, None, return_final=True)
        empty_vol = geo_norm.new_zeros(1, 0, geo_norm.shape[-1]).to(device, non_blocking=True)
        out_chunks = []
        for start in range(0, surf_query_norm.shape[0], chunk_size):
            end = min(start + chunk_size, surf_query_norm.shape[0])
            surf_chunk = surf_query_norm[start:end].unsqueeze(0).to(device, non_blocking=True)
            pred_norm, _, aux = model.predict_from_encoded(
                intermediate_latent_geometries,
                latent_geo_pos,
                final_latent_geo,
                surf_chunk,
                empty_vol,
                None,
                return_aux=True,
            )
            pred_base = denorm_fields(aux["surface_base"][0].to(torch.float32).cpu(), mean_surf, std_surf)
            out_chunks.append(pred_base[:, 0:1].numpy())
    return np.concatenate(out_chunks, axis=0)


def summarize_array(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q001": float(np.quantile(arr, 0.001)),
        "q999": float(np.quantile(arr, 0.999)),
    }


def export_surface_dir(
    surface_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    smart_cfg,
    darm_cfg,
    test_data,
    stats,
    spatial_dim: int,
    surf_channels: int,
    vol_channels: int,
    params_dim: int,
    training_surface_query_budget: int,
) -> None:
    if int(smart_cfg.num_body_points) != int(darm_cfg.num_body_points):
        raise ValueError(
            f"SMART and DARM geometry input budgets differ: "
            f"{smart_cfg.num_body_points} vs {darm_cfg.num_body_points}."
        )

    surface_coords = np.load(surface_dir / "surface_coords.npy").astype(np.float32, copy=False)
    surface_pressure = np.load(surface_dir / "surface_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
    finite_mask = np.isfinite(surface_coords).all(axis=1) & np.isfinite(surface_pressure).all(axis=1)
    surface_coords = surface_coords[finite_mask]
    surface_pressure = surface_pressure[finite_mask]
    if surface_coords.shape[0] == 0:
        raise ValueError(f"No valid finite surface points found in {surface_dir}")

    input_budget = int(smart_cfg.num_body_points)
    chunk_size = int(args.surface_query_chunk) if int(args.surface_query_chunk) > 0 else int(training_surface_query_budget)
    if chunk_size <= 0:
        raise ValueError("surface_query_chunk must be positive after resolving the training budget.")

    rng = np.random.default_rng(args.seed)
    shared_geo_idx = sample_indices(surface_coords.shape[0], input_budget, rng)

    geo_coords_raw = torch.from_numpy(np.asarray(surface_coords[shared_geo_idx], dtype=np.float32))
    surf_query_coords_raw = torch.from_numpy(np.asarray(surface_coords, dtype=np.float32))
    surf_query_pressure_gt = surface_pressure[:, 0].astype(np.float32, copy=False)

    min_pos = test_data.min_pos.float()
    max_pos = test_data.max_pos.float()
    mean_surf = stats[0][:surf_channels].float()
    std_surf = stats[1][:surf_channels].float()

    geo_norm = normalize_pos(geo_coords_raw, min_pos, max_pos)
    surf_query_norm = normalize_pos(surf_query_coords_raw, min_pos, max_pos)

    smart_model, smart_kwargs = build_model(SMART, smart_cfg, spatial_dim, surf_channels, vol_channels, params_dim, device)
    load_checkpoint_into_model(smart_model, args.smart_checkpoint, device)

    darm_model, darm_kwargs = build_model(DARM, darm_cfg, spatial_dim, surf_channels, vol_channels, params_dim, device)
    load_checkpoint_into_model(darm_model, args.darm_checkpoint, device)

    smart_pred = run_smart_surface_inference(
        smart_model,
        geo_norm,
        surf_query_norm,
        mean_surf,
        std_surf,
        chunk_size=chunk_size,
        device=device,
    )
    darm_pred = run_darm_surface_inference(
        darm_model,
        geo_norm,
        surf_query_norm,
        mean_surf,
        std_surf,
        chunk_size=chunk_size,
        device=device,
    )
    darm_base_pred = run_darm_base_surface_inference(
        darm_model,
        geo_norm,
        surf_query_norm,
        mean_surf,
        std_surf,
        chunk_size=chunk_size,
        device=device,
    )

    smart_pressure = smart_pred[:, 0].astype(np.float32, copy=False)
    darm_pressure = darm_pred[:, 0].astype(np.float32, copy=False)
    darm_base_pressure = darm_base_pred[:, 0].astype(np.float32, copy=False)
    smart_abs_err = np.abs(smart_pressure - surf_query_pressure_gt)
    darm_abs_err = np.abs(darm_pressure - surf_query_pressure_gt)
    max_abs_darm_pred = float(np.max(np.abs(darm_pressure)))

    metrics = {
        "surface_dir": str(surface_dir),
        "seed": int(args.seed),
        "surface_points": int(surface_coords.shape[0]),
        "shared_geometry_input_points": int(shared_geo_idx.shape[0]),
        "surface_query_chunk": int(chunk_size),
        "smart_checkpoint": str(Path(args.smart_checkpoint).resolve()),
        "darm_checkpoint": str(Path(args.darm_checkpoint).resolve()),
        "smart_config": args.smart_config,
        "darm_config": args.darm_config,
        "smart_model_kwargs": smart_kwargs,
        "darm_model_kwargs": darm_kwargs,
        "pressure_metrics": {
            "smart": {
                "mae": float(smart_abs_err.mean()),
                "rmse": float(np.sqrt(np.mean((smart_pressure - surf_query_pressure_gt) ** 2))),
                "rel_l2": rel_l2(surf_query_pressure_gt, smart_pressure),
                "max_abs_err": float(smart_abs_err.max()),
            },
            "darm": {
                "mae": float(darm_abs_err.mean()),
                "rmse": float(np.sqrt(np.mean((darm_pressure - surf_query_pressure_gt) ** 2))),
                "rel_l2": rel_l2(surf_query_pressure_gt, darm_pressure),
                "max_abs_err": float(darm_abs_err.max()),
            },
            "darm_base_only": {
                "mae": float(np.mean(np.abs(darm_base_pressure - surf_query_pressure_gt))),
                "rmse": float(np.sqrt(np.mean((darm_base_pressure - surf_query_pressure_gt) ** 2))),
                "rel_l2": rel_l2(surf_query_pressure_gt, darm_base_pressure),
                "max_abs_err": float(np.max(np.abs(darm_base_pressure - surf_query_pressure_gt))),
            },
        },
        "prediction_stats": {
            "gt_pressure": summarize_array(surf_query_pressure_gt),
            "smart_pressure": summarize_array(smart_pressure),
            "darm_pressure": summarize_array(darm_pressure),
            "darm_base_pressure": summarize_array(darm_base_pressure),
        },
    }

    if args.output_dir:
        out_root = Path(args.output_dir) / surface_dir.name
    else:
        out_root = REPO_ROOT / "results" / "smart_darm_external_surface_pressure" / surface_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    smart_vtk_path = out_root / "smart_surface_pressure.vtk"
    darm_vtk_path = out_root / "darm_surface_pressure.vtk"
    metrics_path = out_root / "surface_pressure_metrics.json"
    np.save(out_root / "shared_geometry_indices.npy", shared_geo_idx)

    write_polydata_vtk(
        smart_vtk_path,
        surface_coords,
        {
            "gt_pressure": surf_query_pressure_gt,
            "pred_pressure": smart_pressure,
            "abs_error_pressure": smart_abs_err,
        },
    )
    write_polydata_vtk(
        darm_vtk_path,
        surface_coords,
        {
            "gt_pressure": surf_query_pressure_gt,
            "pred_pressure": darm_pressure,
            "abs_error_pressure": darm_abs_err,
        },
    )
    write_polydata_vtk(
        out_root / "darm_base_surface_pressure.vtk",
        surface_coords,
        {
            "gt_pressure": surf_query_pressure_gt,
            "pred_pressure": darm_base_pressure,
            "abs_error_pressure": np.abs(darm_base_pressure - surf_query_pressure_gt),
        },
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if max_abs_darm_pred > float(args.darm_outlier_abs_threshold):
        raise RuntimeError(
            f"DARM produced implausible pressure spikes on {surface_dir.name}: "
            f"max_abs_prediction={max_abs_darm_pred:.3f} exceeds threshold {float(args.darm_outlier_abs_threshold):.3f}. "
            f"See {out_root / 'darm_base_surface_pressure.vtk'} and {metrics_path} for debugging."
        )

    print(f"Surface dir: {surface_dir}")
    print(f"SMART VTK: {smart_vtk_path}")
    print(f"DARM VTK: {darm_vtk_path}")
    print(f"Metrics: {metrics_path}")
    print(f"SMART pressure rel_l2: {metrics['pressure_metrics']['smart']['rel_l2']:.6f}")
    print(f"DARM pressure rel_l2: {metrics['pressure_metrics']['darm']['rel_l2']:.6f}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    root_path = Path(args.surface_dir).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Surface directory not found: {root_path}")
    surface_dirs = resolve_surface_dirs(root_path)

    smart_cfg = load_cfg(args.smart_config)
    darm_cfg = load_cfg(args.darm_config)
    training_surface_query_budget = min(int(smart_cfg.num_surface_points), int(darm_cfg.num_surface_points))

    smart_cfg.num_surface_points = 0
    smart_cfg.num_volume_points = 0
    smart_cfg.num_body_points = int(smart_cfg.num_body_points)
    smart_cfg.batch_size = 1
    smart_cfg.model_name = "SMART"

    _train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(smart_cfg)
    if fields["surface"][0] != "pressure":
        raise RuntimeError(f"Expected pressure to be the first surface field, got {fields['surface']}")

    for surface_dir in surface_dirs:
        export_surface_dir(
            surface_dir=surface_dir,
            args=args,
            device=device,
            smart_cfg=smart_cfg,
            darm_cfg=darm_cfg,
            test_data=test_data,
            stats=stats,
            spatial_dim=spatial_dim,
            surf_channels=surf_channels,
            vol_channels=vol_channels,
            params_dim=params_dim,
            training_surface_query_budget=training_surface_query_budget,
        )


if __name__ == "__main__":
    main()
