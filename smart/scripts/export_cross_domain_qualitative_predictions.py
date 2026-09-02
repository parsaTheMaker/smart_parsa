#!/usr/bin/env python3
"""Export matched non-Audi Base/DeAL fields for publication figures.

The exporter intentionally reuses the comparison scripts' configuration,
sampling, normalization, and inference helpers.  Every condition is evaluated
at one fixed set of physical surface and volume queries; only the encoder
representation changes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SMART_ROOT.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2  # noqa: E402
from data.pump_dataset import PumpDataset  # noqa: E402
from models.smart.smart import SMART  # noqa: E402
from scripts import compare_shift_endpoint_strategies as endpoint  # noqa: E402


DRIVAER_FIELDS = (
    "pressure",
    "normal_x",
    "normal_y",
    "normal_z",
    "wall_shear_x",
    "wall_shear_y",
    "wall_shear_z",
)
PUMP_FIELDS = (
    "pressure",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "wall_shear_x",
    "wall_shear_y",
    "wall_shear_z",
)


def load_experiment_config(name: str, stack: tuple[str, ...] = ()):
    """Resolve the local Hydra defaults graph without initializing Hydra."""
    from omegaconf import OmegaConf

    if name in stack:
        raise ValueError(f"Circular config defaults: {' -> '.join((*stack, name))}")
    path = SMART_ROOT / "config" / f"{name}.yaml"
    root = OmegaConf.load(path)
    merged = OmegaConf.create()
    for default in root.get("defaults", []):
        if not isinstance(default, str) or default == "_self_" or default.startswith("override "):
            continue
        merged = OmegaConf.merge(merged, load_experiment_config(default.rsplit("/", 1)[-1], (*stack, name)))
    return OmegaConf.merge(merged, root.get("experiment", OmegaConf.create()))


def load_smart(config_name: str, checkpoint: Path, device: torch.device, query_chunk_size: int) -> SMART:
    from omegaconf import OmegaConf

    config = load_experiment_config(config_name)
    architecture = OmegaConf.to_container(config.architecture, resolve=True)
    model = SMART(
        spatial_dim=3,
        surface_channels=len(DRIVAER_FIELDS),
        volume_channels=4,
        parameter_channels=0,
        **architecture,
    )
    model.subregion_size = max(int(getattr(model, "subregion_size", 262144)), int(query_chunk_size))
    endpoint.load_checkpoint(model, checkpoint, device)
    return model


def uniform_indices(count: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    if budget <= 0 or budget >= count:
        return np.arange(count, dtype=np.int64)
    return rng.choice(count, size=budget, replace=False).astype(np.int64, copy=False)


def weighted_indices(weights: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    if budget <= 0 or budget >= weights.shape[0]:
        return np.arange(weights.shape[0], dtype=np.int64)
    probabilities = np.clip(np.asarray(weights, dtype=np.float64), 1.0e-24, None)
    probabilities /= probabilities.sum()
    return rng.choice(weights.shape[0], size=budget, replace=False, p=probabilities).astype(np.int64, copy=False)


def sine_weights(points: np.ndarray, axis: int) -> np.ndarray:
    coordinate = np.asarray(points[:, axis], dtype=np.float64)
    span = max(float(coordinate.max() - coordinate.min()), 1.0e-12)
    normalized = np.clip((coordinate - coordinate.min()) / span, 0.0, 1.0)
    return np.clip(np.sin(np.pi * normalized) ** 2 + 1.0e-6, 1.0e-6, None)


def fixed_indices(count: int, budget: int, seed_components: list[int]) -> np.ndarray:
    indices = uniform_indices(count, budget, np.random.default_rng(np.random.SeedSequence(seed_components)))
    indices.sort()
    return indices


def normalize_positions(points: np.ndarray, minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32))
    return (tensor - minimum) / torch.clamp(maximum - minimum, min=1.0e-12)


def relative_l2(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), 1.0e-12))


def write_point_cloud(path: Path, points: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
    """Write a portable binary legacy VTK point cloud with named arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    count = points.shape[0]
    connectivity = np.empty((count, 2), dtype=">i4")
    connectivity[:, 0] = 1
    connectivity[:, 1] = np.arange(count, dtype=np.int32)
    with path.open("wb") as handle:
        handle.write(b"# vtk DataFile Version 3.0\n")
        handle.write(b"Matched geometry-to-field qualitative export\n")
        handle.write(b"BINARY\nDATASET POLYDATA\n")
        handle.write(f"POINTS {count} float\n".encode("ascii"))
        handle.write(points.astype(">f4", copy=False).tobytes())
        handle.write(b"\n")
        handle.write(f"VERTICES {count} {2 * count}\n".encode("ascii"))
        handle.write(connectivity.tobytes())
        handle.write(b"\n")
        if not arrays:
            return
        handle.write(f"POINT_DATA {count}\n".encode("ascii"))
        for raw_name, raw_values in arrays.items():
            values = np.asarray(raw_values, dtype=np.float32)
            if values.ndim == 1:
                values = values[:, None]
            if values.shape[0] != count:
                raise ValueError(f"Array {raw_name!r} has {values.shape[0]} rows; expected {count}.")
            name = re.sub(r"[^A-Za-z0-9_]", "_", raw_name)
            if values.shape[1] == 3:
                handle.write(f"VECTORS {name} float\n".encode("ascii"))
            else:
                handle.write(f"SCALARS {name} float {values.shape[1]}\nLOOKUP_TABLE default\n".encode("ascii"))
            handle.write(values.astype(">f4", copy=False).tobytes())
            handle.write(b"\n")


def predict_drivaer(
    model: SMART,
    device: torch.device,
    geometry: np.ndarray,
    surface_query: torch.Tensor,
    volume_query: torch.Tensor,
    dataset: AhmedMLDatasetV2,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    geometry_tensor = normalize_positions(geometry, dataset.min_pos, dataset.max_pos).unsqueeze(0).to(device)
    surface_tensor = surface_query.unsqueeze(0).to(device)
    volume_tensor = volume_query.unsqueeze(0).to(device)
    seed_tensor = torch.tensor([seed], device=device, dtype=torch.long)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        surface, volume = model.inference(
            geometry_tensor,
            surface_tensor,
            volume_tensor,
            None,
            geometry_sampling_seeds=seed_tensor,
        )
    surface = surface.float().cpu()[0] * dataset.std_surf_data + dataset.mean_surf_data
    volume = volume.float().cpu()[0] * dataset.std_vol_data + dataset.mean_vol_data
    return surface.numpy(), volume.numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("drivaerml", "pump"), required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--remesh-root", type=Path, required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--deal-config", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--deal-checkpoint", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--surface-query-points", type=int, default=65536)
    parser.add_argument("--volume-query-points", type=int, default=65536)
    parser.add_argument("--query-chunk-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_devices(value: str) -> tuple[torch.device, torch.device]:
    devices = [torch.device(item.strip()) for item in value.split(",") if item.strip()]
    if not devices:
        devices = [torch.device("cpu")]
    if len(devices) == 1:
        devices *= 2
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were requested but CUDA is unavailable.")
    return devices[0], devices[1]


def require_inputs(args: argparse.Namespace) -> None:
    for path in (args.data_root, args.remesh_root, args.base_checkpoint, args.deal_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if "audi" in str(args.output_dir).lower():
        raise ValueError("This exporter does not accept Audi-named output paths.")


def remesh_path(dataset: str, root: Path, run_id: int, method: str) -> Path:
    if dataset == "drivaerml":
        return root / method / f"run_{run_id}" / f"drivaer_{run_id}_faces_div10.vtp"
    return root / method / f"sample_{run_id:06d}" / "merged_surfaces_faces_div10.vtp"


def sampled_conditions(
    dataset: str,
    full_geometry: np.ndarray,
    remesh_root: Path,
    run_id: int,
    budget: int,
    seed: int,
) -> dict[str, np.ndarray]:
    conditions: dict[str, np.ndarray] = {}
    for name, axis in (("sine_x", 0), ("sine_y", 1)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, run_id, 101 + axis]))
        weights = sine_weights(full_geometry, axis=axis)
        indices = weighted_indices(weights, budget, rng)
        conditions[name] = np.ascontiguousarray(full_geometry[indices], dtype=np.float32)

    for label, method in (("feature_div10", "feature"), ("qem_div10", "quadric"), ("voxel_div10", "voxel")):
        path = remesh_path(dataset, remesh_root, run_id, method)
        if not path.is_file():
            raise FileNotFoundError(path)
        source = endpoint.load_vtp_points(path)
        rng = np.random.default_rng(np.random.SeedSequence([seed, run_id, 701, len(conditions)]))
        indices = uniform_indices(source.shape[0], budget, rng)
        conditions[label] = np.ascontiguousarray(source[indices], dtype=np.float32)
    return conditions


def field_payload(
    names: tuple[str, ...],
    ground_truth: np.ndarray,
    base: np.ndarray,
    deal: np.ndarray,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        payload[f"ground_truth_{name}"] = ground_truth[:, index]
        payload[f"base_{name}"] = base[:, index]
        payload[f"deal_{name}"] = deal[:, index]
        payload[f"base_absolute_error_{name}"] = np.abs(base[:, index] - ground_truth[:, index])
        payload[f"deal_absolute_error_{name}"] = np.abs(deal[:, index] - ground_truth[:, index])

    vector_groups = []
    if names == DRIVAER_FIELDS:
        vector_groups = [("normal", 1, 4), ("wall_shear", 4, 7)]
    elif names == PUMP_FIELDS:
        vector_groups = [("velocity", 1, 4), ("wall_shear", 4, 7)]
    for group, start, stop in vector_groups:
        for prefix, values in (("ground_truth", ground_truth), ("base", base), ("deal", deal)):
            payload[f"{prefix}_{group}"] = values[:, start:stop]
            payload[f"{prefix}_{group}_magnitude"] = np.linalg.norm(values[:, start:stop], axis=1)
        payload[f"base_absolute_error_{group}_magnitude"] = np.linalg.norm(
            base[:, start:stop] - ground_truth[:, start:stop], axis=1
        )
        payload[f"deal_absolute_error_{group}_magnitude"] = np.linalg.norm(
            deal[:, start:stop] - ground_truth[:, start:stop], axis=1
        )
    return payload


def write_manifest(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "qualitative_export_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def export_drivaerml(args: argparse.Namespace, devices: tuple[torch.device, torch.device]) -> None:
    dataset = AhmedMLDatasetV2(
        saved_folder=str(args.data_root),
        if_test=True,
        geometry_points=131072,
        surface_points=args.surface_query_points,
        volume_points=args.volume_query_points,
        scale_positions=False,
        require_preprocessed=True,
        geometry_density_knn_k=16,
        geometry_density_estimator="kde",
    )
    run_dir = args.data_root / f"run_{args.run_id}"
    surface_coords = np.asarray(np.load(run_dir / "surface_coords.npy", mmap_mode="r"), dtype=np.float32)
    surface_gt = np.concatenate(
        [
            np.asarray(np.load(run_dir / "surface_pMeanTrim.npy", mmap_mode="r"), dtype=np.float32)[:, None],
            np.asarray(np.load(run_dir / "surface_normals.npy", mmap_mode="r"), dtype=np.float32),
            *[
                np.asarray(
                    np.load(run_dir / f"surface_wallShearStressMeanTrim_{axis}.npy", mmap_mode="r"),
                    dtype=np.float32,
                )[:, None]
                for axis in "xyz"
            ],
        ],
        axis=1,
    )
    volume_coords = np.asarray(np.load(run_dir / "volume_coords.npy", mmap_mode="r"), dtype=np.float32)
    volume_gt = np.concatenate(
        [
            np.asarray(np.load(run_dir / "volume_pMeanTrim.npy", mmap_mode="r"), dtype=np.float32)[:, None],
            np.asarray(np.load(run_dir / "volume_UMeanTrim.npy", mmap_mode="r"), dtype=np.float32),
        ],
        axis=1,
    )
    surface_idx = fixed_indices(
        surface_coords.shape[0], args.surface_query_points, [args.seed, args.run_id, 3001]
    )
    volume_idx = fixed_indices(
        volume_coords.shape[0], args.volume_query_points, [args.seed, args.run_id, 3002]
    )
    surface_query = surface_coords[surface_idx]
    volume_query = volume_coords[volume_idx]
    surface_gt = surface_gt[surface_idx]
    volume_gt = volume_gt[volume_idx]
    conditions = sampled_conditions(
        "drivaerml", surface_coords, args.remesh_root, args.run_id, 131072, args.seed
    )

    base = load_smart(args.base_config, args.base_checkpoint, devices[0], args.query_chunk_size)
    deal = load_smart(args.deal_config, args.deal_checkpoint, devices[1], args.query_chunk_size)
    surface_norm = normalize_positions(surface_query, dataset.min_pos, dataset.max_pos)
    volume_norm = normalize_positions(volume_query, dataset.min_pos, dataset.max_pos)

    def infer(model_name: str, model, device: torch.device, geometry: np.ndarray, condition_index: int):
        del model_name
        return predict_drivaer(
            model,
            device,
            geometry,
            surface_norm,
            volume_norm,
            dataset,
            args.seed + 1000 * args.run_id + condition_index,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for condition_index, (condition, geometry) in enumerate(conditions.items()):
            base_future = pool.submit(infer, "SMART", base, devices[0], geometry, condition_index)
            deal_future = pool.submit(infer, "SMART_SATLOSS7_RANGE100", deal, devices[1], geometry, condition_index)
            base_surface, _ = base_future.result()
            deal_surface, _ = deal_future.result()
            write_point_cloud(
                output_dir / f"drivaerml_run_{args.run_id}_{condition}_surface_fields.vtk",
                surface_query,
                field_payload(DRIVAER_FIELDS, surface_gt, base_surface, deal_surface),
            )
            write_point_cloud(
                output_dir / f"drivaerml_run_{args.run_id}_{condition}_encoder_input.vtk",
                geometry,
                {},
            )
            metrics[condition] = {
                "base_surface_relative_l2": relative_l2(surface_gt, base_surface),
                "deal_surface_relative_l2": relative_l2(surface_gt, deal_surface),
            }
            print(f"[DrivAerML] exported {condition}", flush=True)

    write_manifest(
        output_dir,
        {
            "dataset": "DrivAerML",
            "run_id": args.run_id,
            "surface_queries": int(surface_query.shape[0]),
            "volume_queries": int(volume_query.shape[0]),
            "encoder_points": 131072,
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "deal_checkpoint": str(args.deal_checkpoint.resolve()),
            "conditions": metrics,
        },
    )


def export_pump(args: argparse.Namespace, devices: tuple[torch.device, torch.device]) -> None:
    dataset = PumpDataset(
        args.data_root,
        if_test=True,
        geometry_points=0,
        surface_points=args.surface_query_points,
        volume_points=args.volume_query_points,
    )
    queries = endpoint.fixed_queries(
        args.run_id,
        args.data_root,
        dataset,
        args.surface_query_points,
        args.volume_query_points,
        args.seed,
    )
    full_geometry = endpoint.native_geometry(args.run_id, "pump", dataset)
    conditions = sampled_conditions(
        "pump", full_geometry, args.remesh_root, args.run_id, 16384, args.seed
    )
    base, _, _ = endpoint.make_model(
        "pump", args.base_config, args.base_checkpoint, devices[0], args.query_chunk_size
    )
    deal, _, _ = endpoint.make_model(
        "pump", args.deal_config, args.deal_checkpoint, devices[1], args.query_chunk_size
    )
    minimum = queries["minimum"]
    span = queries["span"]

    def infer(model, device: torch.device, geometry: np.ndarray):
        normalized = endpoint.normalize_positions(geometry, minimum, span)
        surface, volume = endpoint.predict_batch(model, device, [normalized], queries)
        return surface[0], volume[0]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_mean = dataset.mean_surf_data.numpy()
    surface_std = dataset.std_surf_data.numpy()
    surface_gt = queries["surface_y"] * surface_std + surface_mean
    metrics = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for condition, geometry in conditions.items():
            base_future = pool.submit(infer, base, devices[0], geometry)
            deal_future = pool.submit(infer, deal, devices[1], geometry)
            base_surface_norm, _ = base_future.result()
            deal_surface_norm, _ = deal_future.result()
            base_surface = base_surface_norm * surface_std + surface_mean
            deal_surface = deal_surface_norm * surface_std + surface_mean
            write_point_cloud(
                output_dir / f"pump_run_{args.run_id}_{condition}_surface_fields.vtk",
                queries["surface_q_physical"],
                field_payload(PUMP_FIELDS, surface_gt, base_surface, deal_surface),
            )
            write_point_cloud(
                output_dir / f"pump_run_{args.run_id}_{condition}_encoder_input.vtk",
                geometry,
                {},
            )
            metrics[condition] = {
                "base_surface_relative_l2": endpoint.relative_l2(queries["surface_y"], base_surface_norm),
                "deal_surface_relative_l2": endpoint.relative_l2(queries["surface_y"], deal_surface_norm),
            }
            print(f"[Pump] exported {condition}", flush=True)

    write_manifest(
        output_dir,
        {
            "dataset": "SHIFT-Pump",
            "run_id": args.run_id,
            "surface_queries": int(queries["surface_q_physical"].shape[0]),
            "volume_queries": int(queries["volume_q"].shape[0]),
            "encoder_points": 16384,
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "deal_checkpoint": str(args.deal_checkpoint.resolve()),
            "conditions": metrics,
        },
    )


def main() -> None:
    args = parse_args()
    require_inputs(args)
    devices = parse_devices(args.devices)
    print(f"Using Base on {devices[0]} and DeAL on {devices[1]}", flush=True)
    if args.dataset == "drivaerml":
        export_drivaerml(args, devices)
    else:
        export_pump(args, devices)


if __name__ == "__main__":
    main()
