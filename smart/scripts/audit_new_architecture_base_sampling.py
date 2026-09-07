#!/usr/bin/env python3
"""Focused sampling-shift audit for AB-UPT and Geo-FNO base models.

The script deliberately keeps the training query budgets fixed and changes
only the encoder geometry.  It is a compact diagnostic, not a replacement for
the paper comparison pipelines.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

from data.datasets import get_dataset
from models.ab_upt import ABUPT
from models.geo_fno import GeoFNO
from train_consistency_common import sample_geometry_view


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "smart" / "config"
METHODS = ("feature", "quadric", "voxel")
FACTORS = (5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("drivaerml", "heat_exchanger", "pump"), required=True)
    parser.add_argument("--ab-upt-config", required=True)
    parser.add_argument("--ab-upt-checkpoint", type=Path, required=True)
    parser.add_argument("--geo-fno-config", required=True)
    parser.add_argument("--geo-fno-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--remesh-root", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--case-ids", default="", help="Optional comma-separated held-out IDs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def compose_config(name: str):
    with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
        return compose(config_name=str(name))


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("model_state", payload.get("state_dict", payload)))
    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a model state dictionary.")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def build_model(kind: str, cfg, checkpoint: Path, device: torch.device, channels: tuple[int, int, int]):
    surface_channels, volume_channels, parameter_channels = channels
    architecture = OmegaConf.to_container(cfg.experiment.architecture, resolve=True)
    if not isinstance(architecture, dict):
        raise TypeError(f"{kind} architecture must be a mapping.")
    ctor = ABUPT if kind == "AB_UPT" else GeoFNO
    model = ctor(
        spatial_dim=3,
        surface_channels=surface_channels,
        volume_channels=volume_channels,
        parameter_channels=parameter_channels,
        **architecture,
    ).to(device)
    state = checkpoint_state(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{kind} checkpoint/config mismatch for {checkpoint}: "
            f"missing={len(missing)}, unexpected={len(unexpected)}."
        )
    return model.eval()


def model_input_budget(cfg) -> int:
    value = int(getattr(cfg.experiment, "num_body_points", 0))
    if value <= 0:
        raise ValueError("Base audit configurations must define a positive training geometry budget.")
    return value


def parse_vtp_array(root: ElementTree.Element, selector: str) -> np.ndarray:
    array = root.find(selector)
    if array is None:
        raise RuntimeError(f"VTP array not found: {selector}")
    vtk_type = array.attrib.get("type", "Float32")
    dtype_map = {"Float32": "f4", "Float64": "f8", "Int32": "i4", "Int64": "i8"}
    if vtk_type not in dtype_map:
        raise RuntimeError(f"Unsupported VTP type {vtk_type!r}")
    endian = "<" if root.attrib.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
    dtype = np.dtype(endian + dtype_map[vtk_type])
    text = "".join((array.text or "").split())
    mode = array.attrib.get("format", "ascii").lower()
    if mode == "ascii":
        return np.fromstring(text, sep=" ", dtype=dtype)
    if mode != "binary":
        raise RuntimeError(f"Unsupported VTP point format {mode!r}")
    raw = base64.b64decode(text, validate=True)
    header_kind = root.attrib.get("header_type", "UInt32")
    header_format = {"UInt32": "I", "UInt64": "Q"}.get(header_kind)
    if header_format is None:
        raise RuntimeError(f"Unsupported VTP header type {header_kind!r}")
    header_size = struct.calcsize(header_format)
    count = struct.unpack(endian + header_format, raw[:header_size])[0]
    return np.frombuffer(raw, dtype=dtype, count=count // dtype.itemsize, offset=header_size)


def read_vtp_points(path: Path) -> np.ndarray:
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        raise RuntimeError(f"Cannot parse VTP {path}") from exc
    values = parse_vtp_array(root, ".//Points/DataArray")
    if values.size % 3:
        raise RuntimeError(f"Invalid XYZ point array in {path}")
    points = np.asarray(values.reshape(-1, 3), dtype=np.float32)
    if not len(points) or not np.isfinite(points).all():
        raise RuntimeError(f"Invalid points in {path}")
    return np.ascontiguousarray(points)


def read_vtp_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = read_vtp_points(path)
    root = ElementTree.parse(path).getroot()
    connectivity = parse_vtp_array(root, ".//Polys/DataArray[@Name='connectivity']").astype(np.int64, copy=False)
    offsets = parse_vtp_array(root, ".//Polys/DataArray[@Name='offsets']").astype(np.int64, copy=False)
    start = np.concatenate([np.zeros(1, dtype=np.int64), offsets[:-1]])
    lengths = offsets - start
    if not np.all(lengths == 3):
        raise RuntimeError(f"{path} contains non-triangular polygons.")
    faces = np.stack([connectivity[offset - 3 : offset] for offset in offsets], axis=0)
    if faces.size == 0 or faces.max(initial=-1) >= len(points):
        raise RuntimeError(f"Invalid triangle indices in {path}")
    return points, np.ascontiguousarray(faces)


def sample_mesh_geometry(path: Path, budget: int, seed: int, dataset_name: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dataset_name != "heat_exchanger":
        points = read_vtp_points(path)
        if len(points) < budget:
            raise ValueError(f"{path} has {len(points)} vertices, below encoder budget {budget}.")
        return np.ascontiguousarray(points[rng.choice(len(points), budget, replace=False)], dtype=np.float32)
    # The Heat Exchanger benchmark represents mesh density by equal-face
    # sampling, matching its established remesh evaluation convention.
    points, faces = read_vtp_mesh(path)
    selected = faces[rng.integers(0, len(faces), size=budget)]
    vertices = points[selected]
    barycentric = -np.log(np.maximum(rng.random((budget, 3)), 1.0e-12))
    barycentric /= barycentric.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(np.einsum("ni,nij->nj", barycentric, vertices), dtype=np.float32)


def remesh_path(dataset_name: str, root: Path, case_id: int, method: str, factor: int) -> Path:
    if dataset_name == "drivaerml":
        return root / method / f"run_{case_id}" / f"drivaer_{case_id}_faces_div{factor}.vtp"
    if dataset_name == "heat_exchanger":
        stem = f"heat_exchange_case_{case_id:05d}_surface_faces_div{factor}.vtp"
        return root / method / f"case_{case_id:05d}" / stem
    return root / method / f"sample_{case_id:06d}" / f"merged_surfaces_faces_div{factor}.vtp"


def has_complete_remesh_set(dataset_name: str, root: Path, case_id: int) -> bool:
    return all(
        remesh_path(dataset_name, root, case_id, method, factor).is_file()
        for method in METHODS
        for factor in FACTORS
    )


def bounds(dataset) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(dataset.min_pos, dtype=np.float32)
    if hasattr(dataset, "max_pos"):
        upper = np.asarray(dataset.max_pos, dtype=np.float32)
    else:
        upper = lower + np.asarray(dataset.position_span, dtype=np.float32)
    return lower, np.maximum(upper - lower, 1.0e-12)


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(pred.float() - target.float()) / torch.linalg.vector_norm(target.float()).clamp_min(1.0e-12))


def predict_metrics(model, device: torch.device, geo: torch.Tensor, surf_q: torch.Tensor, surf_y: torch.Tensor, vol_q: torch.Tensor, vol_y: torch.Tensor, params, dataset) -> dict[str, float]:
    geo = geo.unsqueeze(0).to(device, non_blocking=True)
    surf_q = surf_q.unsqueeze(0).to(device, non_blocking=True)
    vol_q = vol_q.unsqueeze(0).to(device, non_blocking=True)
    params = None if params is None else params.unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        pred_s, pred_v = model.inference(geo, surf_q, vol_q, params)
    pred_s, pred_v = pred_s[0].float().cpu(), pred_v[0].float().cpu()
    standard_surface = rel_l2(pred_s, surf_y)
    standard_volume = rel_l2(pred_v, vol_y)
    surf_mean, surf_std = dataset.mean_surf_data.float(), dataset.std_surf_data.float()
    vol_mean, vol_std = dataset.mean_vol_data.float(), dataset.std_vol_data.float()
    physical_surface = rel_l2(pred_s * surf_std + surf_mean, surf_y * surf_std + surf_mean)
    physical_volume = rel_l2(pred_v * vol_std + vol_mean, vol_y * vol_std + vol_mean)
    return {
        "surface_standardized_rel_l2": standard_surface,
        "volume_standardized_rel_l2": standard_volume,
        "combined_standardized_rel_l2": 0.5 * (standard_surface + standard_volume),
        "surface_physical_rel_l2": physical_surface,
        "volume_physical_rel_l2": physical_volume,
        "combined_physical_rel_l2": 0.5 * (physical_surface + physical_volume),
    }


def unpack_item(item, parameter_channels: int):
    if parameter_channels:
        geo, surf_q, surf_y, vol_q, vol_y, params, density = item
    else:
        geo, surf_q, surf_y, vol_q, vol_y, density = item
        params = None
    return geo, surf_q, surf_y, vol_q, vol_y, params, density


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        condition = row["condition"]
        if condition.startswith("remesh_"):
            condition = f"remesh_{row['method']}_mean_div5_div10"
        grouped.setdefault((row["model"], condition), []).append(row)
    summary = []
    for (model, condition), values in sorted(grouped.items()):
        item: dict[str, Any] = {"model": model, "condition": condition, "count": len(values)}
        for key in values[0]:
            if key.endswith("_rel_l2"):
                scores = np.asarray([float(value[key]) for value in values], dtype=np.float64)
                item[key] = float(scores.mean())
                item[f"{key}_std"] = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        summary.append(item)
    return summary


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.remesh_root = args.remesh_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.ab_upt_checkpoint.is_file() or not args.geo_fno_checkpoint.is_file():
        raise FileNotFoundError("Both completed base checkpoints are required.")

    ab_cfg = compose_config(args.ab_upt_config)
    geo_cfg = compose_config(args.geo_fno_config)
    if str(ab_cfg.experiment.dataset) != str(geo_cfg.experiment.dataset):
        raise ValueError("AB-UPT and Geo-FNO configurations must target the same dataset.")
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(ab_cfg, resolve=True))
    dataset_cfg.experiment.data_path = str(args.data_root)
    dataset_cfg.experiment.num_body_points = 0
    dataset_cfg.experiment.geometry_epoch_seeded_sampling = True
    # Density is used only to construct shifted encoder views, never passed to either base model.
    dataset_cfg.experiment.model_name = "BASE_SAMPLING_AUDIT_DEAL_SAMPLER"
    _, dataset, _, spatial_dim, surf_channels, vol_channels, parameter_channels, _ = get_dataset(dataset_cfg.experiment)
    if spatial_dim != 3:
        raise ValueError("This audit supports the 3D datasets only.")
    dataset.set_epoch(0)
    channels = (surf_channels, vol_channels, parameter_channels)
    devices = [torch.device(value.strip()) for value in args.devices.split(",") if value.strip()]
    if len(devices) != 2:
        raise ValueError("Provide exactly two devices, one per model.")
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA devices requested but CUDA is unavailable.")
    models = {
        "AB-UPT": (build_model("AB_UPT", ab_cfg, args.ab_upt_checkpoint, devices[0], channels), devices[0], model_input_budget(ab_cfg)),
        "Geo-FNO": (build_model("GEOFNO", geo_cfg, args.geo_fno_checkpoint, devices[1], channels), devices[1], model_input_budget(geo_cfg)),
    }
    if len({budget for _, _, budget in models.values()}) != 1:
        raise ValueError("This focused comparison requires matched base encoder budgets.")
    budget = next(iter(models.values()))[2]
    requested = [int(x) for x in args.case_ids.split(",") if x.strip()] if args.case_ids else []
    if requested:
        indices = [dataset.data.index(case_id) for case_id in requested]
        incomplete = [int(dataset.data[index]) for index in indices if not has_complete_remesh_set(args.dataset, args.remesh_root, int(dataset.data[index]))]
        if incomplete:
            raise FileNotFoundError(f"Requested cases lack one or more remeshes: {incomplete}")
    else:
        indices = [
            index for index, case_id in enumerate(dataset.data)
            if has_complete_remesh_set(args.dataset, args.remesh_root, int(case_id))
        ][: int(args.num_cases)]
    if not indices:
        raise ValueError("No held-out cases selected.")
    print(
        f"[{args.dataset}] cases={len(indices)}, encoder_points={budget}, "
        f"surface_queries={dataset.surface_points}, volume_queries={dataset.volume_points}, "
        f"case_ids={[int(dataset.data[index]) for index in indices]}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    lower, span = bounds(dataset)
    for case_index in tqdm(indices, desc=f"{args.dataset} cases", dynamic_ncols=True):
        geo, surf_q, surf_y, vol_q, vol_y, params, density = unpack_item(dataset[case_index], parameter_channels)
        case_id = int(dataset.data[case_index])
        uniform, _, _ = sample_geometry_view(geo.unsqueeze(0), density.unsqueeze(0), budget, "uniform_wor", 0.0, 0.0, args.seed + case_id)
        sine_x, _, _ = sample_geometry_view(geo.unsqueeze(0), density.unsqueeze(0), budget, "sinusoidal_axis_mixture_wor", 0.0, 0.0, args.seed + case_id + 101, sinusoidal_axis=0, sinusoidal_mix_fraction=1.0)
        sine_y, _, _ = sample_geometry_view(geo.unsqueeze(0), density.unsqueeze(0), budget, "sinusoidal_axis_mixture_wor", 0.0, 0.0, args.seed + case_id + 211, sinusoidal_axis=1, sinusoidal_mix_fraction=1.0)
        conditions: list[tuple[str, torch.Tensor, str, int]] = [
            ("unshifted", uniform[0], "", 0),
            ("sine_x", sine_x[0], "", 0),
            ("sine_y", sine_y[0], "", 0),
        ]
        for method in METHODS:
            for factor in FACTORS:
                path = remesh_path(args.dataset, args.remesh_root, case_id, method, factor)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing remesh: {path}")
                physical = sample_mesh_geometry(path, budget, args.seed + case_id * 1000 + factor * 10 + len(method), args.dataset)
                normalized = torch.from_numpy((physical - lower[None, :]) / span[None, :])
                conditions.append((f"remesh_{method}_div{factor}", normalized, method, factor))

        def evaluate_one(model_name: str, model, device, _input_budget: int, condition):
            name, view, method, factor = condition
            metrics = predict_metrics(model, device, view, surf_q, surf_y, vol_q, vol_y, params, dataset)
            return {"case_id": case_id, "model": model_name, "condition": name, "method": method, "factor": factor, **metrics}

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(evaluate_one, name, model, device, input_budget, condition)
                for condition in conditions
                for name, (model, device, input_budget) in models.items()
            ]
            rows.extend(future.result() for future in futures)

    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output_dir / "per_case_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"dataset": args.dataset, "query_budgets": {"surface": int(dataset.surface_points), "volume": int(dataset.volume_points)}, "encoder_budget": budget, "rows": summary}, handle, indent=2)
    print("\nModel | Condition | combined physical rel-L2 (mean +/- SD)")
    for item in summary:
        print(f"{item['model']} | {item['condition']} | {item['combined_physical_rel_l2']:.4f} +/- {item['combined_physical_rel_l2_std']:.4f}")


if __name__ == "__main__":
    main()
