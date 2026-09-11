#!/usr/bin/env python3
"""Export pointwise surface errors averaged over all held-out input shifts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy


SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from scripts import compare_shift_endpoint_strategies as endpoint  # noqa: E402
from scripts.export_full_surface_qualitative_predictions import (  # noqa: E402
    DEFAULTS,
    dataset_for,
    decode_surface,
    heat_flux_at_mesh_points,
    make_models,
    parse_devices,
    point_values,
    read_polydata,
)


REMESH_ROOTS = {
    "pump": Path("/mnt/data/parsa/shift_pump_random1400_surface_vtp_remesh_v4"),
    "heat_exchanger": Path("/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_remesh_v4"),
}
REMESH_METHODS = OrderedDict(
    (
        ("feature", "angle"),
        ("quadric", "isotropic"),
        ("voxel", "voxel"),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(REMESH_ROOTS), required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--surface-mesh", type=Path)
    parser.add_argument("--native-field-source", type=Path)
    parser.add_argument("--remesh-root", type=Path)
    parser.add_argument("--base-config")
    parser.add_argument("--deal-config")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--deal-checkpoint", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--query-chunk-size", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conditions",
        default="all",
        help="Comma-separated condition names, or 'all' for the eight held-out shifts.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remesh_path(root: Path, dataset: str, run_id: int, method: str, factor: int) -> Path:
    if dataset == "pump":
        return root / method / f"sample_{run_id:06d}" / f"merged_surfaces_faces_div{factor}.vtp"
    return (
        root
        / method
        / f"case_{run_id:05d}"
        / f"heat_exchange_case_{run_id:05d}_surface_faces_div{factor}.vtp"
    )


def encoder_conditions(
    dataset_name: str,
    dataset,
    run_id: int,
    remesh_root: Path,
    budget: int,
    seed: int,
) -> OrderedDict[str, dict[str, object]]:
    original = endpoint.native_geometry(run_id, dataset_name, dataset)
    modes = {
        "sine_x": endpoint.mode_records(["sine_x"], (0.0, 1.0))["sine_x_1.00"],
        "sine_y": endpoint.mode_records(["sine_y"], (0.0, 1.0))["sine_y_1.00"],
    }
    conditions: OrderedDict[str, dict[str, object]] = OrderedDict()
    for name, mode in modes.items():
        category = f"{name}_1"
        condition_seed = (
            seed
            + 1000003 * run_id
            + endpoint.stable_tag(category)
            + endpoint.stable_tag("original")
        )
        indices = endpoint.sample_encoder_indices(original, None, mode, budget, condition_seed)
        conditions[name] = {
            "geometry": np.ascontiguousarray(original[indices], dtype=np.float32),
            "source": "original",
            "seed": condition_seed,
        }

    for factor in (5, 10):
        category = f"remeshing_div{factor}_mean"
        for method, source_prefix in REMESH_METHODS.items():
            source_name = f"{source_prefix}_div{factor}"
            path = remesh_path(remesh_root, dataset_name, run_id, method, factor)
            if not path.is_file():
                raise FileNotFoundError(path)
            source = endpoint.load_vtp_points(path)
            condition_seed = (
                seed
                + 1000003 * run_id
                + endpoint.stable_tag(category)
                + endpoint.stable_tag(source_name)
            )
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [condition_seed, endpoint.stable_tag("remeshing_uniform")]
                )
            )
            indices = endpoint.sample_uniform(source.shape[0], budget, rng)
            conditions[f"{method}_{factor}x"] = {
                "geometry": np.ascontiguousarray(source[indices], dtype=np.float32),
                "source": str(path.resolve()),
                "seed": condition_seed,
            }
    return conditions


def select_conditions(
    available: OrderedDict[str, dict[str, object]], requested: str
) -> OrderedDict[str, dict[str, object]]:
    if requested.strip().lower() == "all":
        return available
    names = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown conditions {unknown}; available conditions are {list(available)}")
    if not names:
        raise ValueError("At least one condition is required.")
    return OrderedDict((name, available[name]) for name in names)


def ground_truth_for(dataset_name: str, surface: vtk.vtkPolyData, native_path: Path) -> np.ndarray:
    if dataset_name == "pump":
        return np.linalg.norm(point_values(surface, "Wall Shear Stress (N/m²)"), axis=1)
    return heat_flux_at_mesh_points(surface, native_path)


def displayed_field(dataset_name: str, prediction: np.ndarray) -> np.ndarray:
    if dataset_name == "pump":
        return np.linalg.norm(prediction[:, 4:7], axis=1)
    return prediction[:, 0]


def main() -> None:
    args = parse_args()
    defaults = DEFAULTS[args.dataset]
    data_root = args.data_root or defaults["data_root"]
    surface_path = args.surface_mesh or Path(
        str(defaults["surface_template"]).format(run_id=args.run_id)
    )
    native_path = args.native_field_source or Path(
        str(defaults["native_template"]).format(run_id=args.run_id)
    )
    remesh_root = args.remesh_root or REMESH_ROOTS[args.dataset]
    base_config = args.base_config or str(defaults["base_config"])
    deal_config = args.deal_config or str(defaults["deal_config"])
    required = (
        data_root,
        surface_path,
        native_path,
        remesh_root,
        args.base_checkpoint,
        args.deal_checkpoint,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    surface = read_polydata(surface_path)
    queries = np.asarray(vtk_to_numpy(surface.GetPoints().GetData()), dtype=np.float32)
    ground_truth = ground_truth_for(args.dataset, surface, native_path)
    dataset = dataset_for(args.dataset, data_root)
    base_device, deal_device = parse_devices(args.devices)
    base_model, deal_model, encoder_budget = make_models(
        args.dataset,
        base_config,
        deal_config,
        args.base_checkpoint,
        args.deal_checkpoint,
        base_device,
        deal_device,
        args.query_chunk_size,
    )
    conditions = select_conditions(
        encoder_conditions(
            args.dataset,
            dataset,
            args.run_id,
            remesh_root,
            encoder_budget,
            args.seed,
        ),
        args.conditions,
    )

    minimum = dataset.min_pos.detach().cpu().numpy().astype(np.float32)
    span = dataset.position_span.detach().cpu().numpy().astype(np.float32)
    surface_mean = dataset.mean_surf_data.detach().cpu().numpy().astype(np.float32)
    surface_std = dataset.std_surf_data.detach().cpu().numpy().astype(np.float32)
    params = dataset.get_case_params(args.run_id) if hasattr(dataset, "get_case_params") else None
    valid = np.zeros(queries.shape[0], dtype=bool)
    connectivity = np.asarray(
        vtk_to_numpy(surface.GetPolys().GetConnectivityArray()), dtype=np.int64
    )
    valid[np.unique(connectivity)] = True

    base_prediction_sum = np.zeros(queries.shape[0], dtype=np.float64)
    deal_prediction_sum = np.zeros(queries.shape[0], dtype=np.float64)
    base_error_sum = np.zeros(queries.shape[0], dtype=np.float64)
    deal_error_sum = np.zeros(queries.shape[0], dtype=np.float64)
    raw_arrays: dict[str, np.ndarray] = {}
    condition_manifest = []
    print(
        f"[{args.dataset}] {len(conditions)} conditions, {encoder_budget:,} encoder points, "
        f"{queries.shape[0]:,} native queries; Base->{base_device}, DeAL->{deal_device}",
        flush=True,
    )
    for index, (name, record) in enumerate(conditions.items()):
        geometry = record["geometry"]
        condition_seed = int(record["seed"])
        with ThreadPoolExecutor(max_workers=2 if base_device != deal_device else 1) as pool:
            futures = [
                pool.submit(
                    decode_surface,
                    model,
                    device,
                    geometry,
                    queries,
                    params,
                    minimum,
                    span,
                    surface_mean,
                    surface_std,
                    args.query_chunk_size,
                    condition_seed,
                )
                for model, device in ((base_model, base_device), (deal_model, deal_device))
            ]
            base_full, deal_full = (future.result() for future in futures)
        base_field = displayed_field(args.dataset, base_full)
        deal_field = displayed_field(args.dataset, deal_full)
        base_error = np.abs(base_field - ground_truth)
        deal_error = np.abs(deal_field - ground_truth)
        if not all(
            np.isfinite(values[valid]).all()
            for values in (ground_truth, base_field, deal_field, base_error, deal_error)
        ):
            raise RuntimeError(f"Condition {name} contains non-finite rendered values.")
        base_prediction_sum += base_field
        deal_prediction_sum += deal_field
        base_error_sum += base_error
        deal_error_sum += deal_error
        raw_arrays[f"base_absolute_error__{name}"] = base_error.astype(np.float32)
        raw_arrays[f"deal_absolute_error__{name}"] = deal_error.astype(np.float32)
        condition_manifest.append(
            {
                "name": name,
                "source": record["source"],
                "sampling_seed": condition_seed,
                "encoder_points": int(geometry.shape[0]),
                "base_mean_absolute_error": float(base_error[valid].mean()),
                "deal_mean_absolute_error": float(deal_error[valid].mean()),
            }
        )
        print(f"[{index + 1}/{len(conditions)}] exported {name}", flush=True)

    count = float(len(conditions))
    field = str(defaults["field"])
    arrays = {
        f"ground_truth_{field}": ground_truth.astype(np.float32),
        f"base_mean_prediction_{field}": (base_prediction_sum / count).astype(np.float32),
        f"deal_mean_prediction_{field}": (deal_prediction_sum / count).astype(np.float32),
        f"base_mean_absolute_error_{field}": (base_error_sum / count).astype(np.float32),
        f"deal_mean_absolute_error_{field}": (deal_error_sum / count).astype(np.float32),
        "rendered_vertex_mask": valid,
        **raw_arrays,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    summary = {
        "dataset": args.dataset,
        "run_id": args.run_id,
        "field": field,
        "aggregation": "equal-weight pointwise mean absolute error over held-out encoder conditions",
        "condition_count": len(conditions),
        "conditions": condition_manifest,
        "native_surface_queries": int(queries.shape[0]),
        "rendered_surface_vertices": int(valid.sum()),
        "surface_mesh": str(surface_path.resolve()),
        "native_field_source": str(native_path.resolve()),
        "base_config": base_config,
        "deal_config": deal_config,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_sha256": checkpoint_sha256(args.base_checkpoint),
        "deal_checkpoint": str(args.deal_checkpoint.resolve()),
        "deal_checkpoint_sha256": checkpoint_sha256(args.deal_checkpoint),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
