#!/usr/bin/env python3
"""Export native-query error maps for a low-DeAL, large-gap qualitative case."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import vtk
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.datasets import get_dataset  # noqa: E402
from models.smart.smart import SMART  # noqa: E402
from train_consistency_common import sample_geometry_view  # noqa: E402


TASKS = {
    "drivaerml": {
        "query_kind": "surface",
        "scalar_mode": "first",
        "selection_metric": "surface_physical_rel_l2",
        "methods": ("feature", "quadric", "voxel"),
        "field": "surface_pressure",
    },
    "pump": {
        "query_kind": "volume",
        "scalar_mode": "velocity_magnitude",
        "selection_metric": "volume_physical_rel_l2",
        "methods": ("feature", "quadric", "voxel"),
        "field": "velocity_magnitude",
    },
    "heat_exchanger": {
        "query_kind": "volume",
        "scalar_mode": "first",
        "selection_metric": "volume_physical_rel_l2",
        "methods": ("feature", "quadric"),
        "field": "temperature",
    },
    "c_core": {
        "query_kind": "surface",
        "scalar_mode": "vector_magnitude",
        "selection_metric": "surface_physical_rel_l2",
        "methods": ("feature", "quadric", "voxel"),
        "field": "magnetic_flux_density_magnitude",
    },
}
CONFIG_DIR = SMART_ROOT / "config"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset", choices=tuple(TASKS), required=True)
    prepare.add_argument("--run-id", type=int, required=True)
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--remesh-root", type=Path, required=True)
    prepare.add_argument("--per-case-metrics", type=Path, required=True)
    prepare.add_argument("--native-mesh", type=Path, required=True)
    prepare.add_argument("--native-field-source", type=Path)
    prepare.add_argument("--pump-slice-z", type=float)
    prepare.add_argument("--pump-slice-fraction", type=float, default=0.605)
    prepare.add_argument("--heat-slice-fraction", type=float, default=0.5)
    prepare.add_argument("--balanced-top-k", type=int, default=1)
    prepare.add_argument("--min-channels", type=int, default=0)
    prepare.add_argument("--waviness-category")
    prepare.add_argument("--candidate-split", choices=("validation", "all"), default="validation")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--bundle", type=Path, required=True)
    prepare.add_argument("--query-mesh", type=Path, required=True)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--dataset", choices=tuple(TASKS), required=True)
    infer.add_argument("--run-id", type=int, required=True)
    infer.add_argument("--base-config", required=True)
    infer.add_argument("--deal-config", required=True)
    infer.add_argument("--data-root", type=Path, required=True)
    infer.add_argument("--base-checkpoint", type=Path, required=True)
    infer.add_argument("--deal-checkpoint", type=Path, required=True)
    infer.add_argument("--bundle", type=Path, required=True)
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--query-chunk-size", type=int, default=131072)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compose_config(name: str):
    with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
        return compose(config_name=str(name))


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("model_state", payload.get("state_dict", payload)))
    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a model state dictionary")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def checkpoint_metadata(path: Path) -> dict[str, str | int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": str(path.resolve()), "epoch": int(payload.get("epoch", -1)), "sha256": sha256(path)}


def input_budget(cfg) -> int:
    experiment = cfg.experiment
    for key in ("primary_view_geometry_points", "view_geometry_points", "num_body_points"):
        value = int(getattr(experiment, key, 0))
        if value > 0:
            return value
    raise ValueError("Configuration has no positive encoder input budget")


def bounds(dataset) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(dataset.min_pos, dtype=np.float32)
    upper = (
        np.asarray(dataset.max_pos, dtype=np.float32)
        if hasattr(dataset, "max_pos")
        else lower + np.asarray(dataset.position_span, dtype=np.float32)
    )
    return lower, np.maximum(upper - lower, 1.0e-12)


def unpack_item(item, parameter_channels: int):
    if parameter_channels:
        geometry, surface_q, surface_y, volume_q, volume_y, params, density = item
    else:
        geometry, surface_q, surface_y, volume_q, volume_y, density = item
        params = None
    return geometry, surface_q, surface_y, volume_q, volume_y, params, density


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def build_smart(cfg, checkpoint: Path, device: torch.device, channels: tuple[int, int, int]):
    architecture = OmegaConf.to_container(cfg.experiment.architecture, resolve=True)
    if not isinstance(architecture, dict):
        raise TypeError("experiment.architecture must be a mapping")
    model = SMART(
        spatial_dim=3,
        surface_channels=channels[0],
        volume_channels=channels[1],
        parameter_channels=channels[2],
        **architecture,
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint_state(checkpoint), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/config mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return model.eval()


def read_unstructured_grid(path: Path) -> vtk.vtkUnstructuredGrid:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    output = vtk.vtkUnstructuredGrid()
    output.ShallowCopy(reader.GetOutput())
    if output.GetNumberOfPoints() == 0:
        raise ValueError(f"No points found in {path}")
    return output


def read_polydata(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    output = vtk.vtkPolyData()
    output.ShallowCopy(reader.GetOutput())
    if output.GetNumberOfPoints() == 0:
        raise ValueError(f"No points found in {path}")
    return output


def cell_values_at_points(poly: vtk.vtkPolyData, name: str) -> np.ndarray:
    if poly.GetCellData().GetArray(name) is None:
        raise KeyError(f"Cell array {name!r} is absent")
    conversion = vtk.vtkCellDataToPointData()
    conversion.SetInputData(poly)
    conversion.PassCellDataOff()
    conversion.Update()
    array = conversion.GetOutput().GetPointData().GetArray(name)
    if array is None:
        raise KeyError(f"Converted point array {name!r} is absent")
    return np.asarray(vtk_to_numpy(array), dtype=np.float32)


def native_volume_slice(
    volume: vtk.vtkDataSet,
    normal: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> vtk.vtkPolyData:
    plane = vtk.vtkPlane()
    plane.SetNormal(*normal)
    plane.SetOrigin(*origin)
    cutter = vtk.vtkPlaneCutter()
    cutter.SetInputData(volume)
    cutter.SetPlane(plane)
    cutter.Update()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(cutter.GetOutputPort())
    triangles.PassLinesOff()
    triangles.PassVertsOff()
    triangles.Update()
    output = vtk.vtkPolyData()
    output.ShallowCopy(triangles.GetOutput())
    if output.GetNumberOfPoints() == 0 or output.GetNumberOfPolys() == 0:
        raise ValueError("Native volume cut produced an empty slice")
    return output


def tetrahedral_mesh(points: np.ndarray, tetrahedra: np.ndarray) -> vtk.vtkUnstructuredGrid:
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    tetrahedra = np.ascontiguousarray(tetrahedra, dtype=np.int64)
    offsets = np.arange(0, 4 * (len(tetrahedra) + 1), 4, dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtk(offsets, deep=True, array_type=vtk.VTK_ID_TYPE),
        numpy_to_vtk(tetrahedra.reshape(-1), deep=True, array_type=vtk.VTK_ID_TYPE),
    )
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)
    grid.SetCells(vtk.VTK_TETRA, cells)
    return grid


def add_point_array(dataset: vtk.vtkDataSet, name: str, values: np.ndarray) -> None:
    array = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
    array.SetName(name)
    dataset.GetPointData().AddArray(array)


def write_polydata(path: Path, poly: vtk.vtkPolyData, ground_truth: np.ndarray) -> None:
    output = vtk.vtkPolyData()
    output.DeepCopy(poly)
    output.GetPointData().Initialize()
    add_point_array(output, "ground_truth", ground_truth)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(output)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOn()
    if writer.Write() != 1:
        raise IOError(f"Could not write {path}")


def ranked_cases(path: Path, metric: str) -> list[dict[str, float | int]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    shifted = sorted({row["condition"] for row in rows if row["condition"] != "original"})
    by_case: dict[tuple[int, str], dict[str, float]] = {}
    for row in rows:
        if row["condition"] in shifted:
            by_case.setdefault((int(row["case_id"]), row["variant"]), {})[
                row["condition"]
            ] = float(row[metric])
    rankings = []
    for case_id in sorted({case for case, _ in by_case}):
        base = by_case.get((case_id, "base"), {})
        deal = by_case.get((case_id, "deal"), {})
        if not all(condition in base and condition in deal for condition in shifted):
            continue
        base_mean = float(np.mean([base[condition] for condition in shifted]))
        deal_mean = float(np.mean([deal[condition] for condition in shifted]))
        rankings.append(
            {
                "case_id": case_id,
                "base_mean": base_mean,
                "deal_mean": deal_mean,
                "gap": base_mean - deal_mean,
            }
        )
    if not rankings:
        return rankings
    deal_values = np.asarray([row["deal_mean"] for row in rankings], dtype=np.float64)
    gap_values = np.asarray([row["gap"] for row in rankings], dtype=np.float64)
    deal_span = max(float(np.ptp(deal_values)), 1.0e-12)
    gap_span = max(float(np.ptp(gap_values)), 1.0e-12)
    for row in rankings:
        normalized_deal = (float(row["deal_mean"]) - float(deal_values.min())) / deal_span
        normalized_gap_deficit = (float(gap_values.max()) - float(row["gap"])) / gap_span
        row["balanced_score"] = float(np.hypot(normalized_deal, normalized_gap_deficit))
    return sorted(rankings, key=lambda row: (float(row["balanced_score"]), int(row["case_id"])))


def dataset_context(config_name: str, data_root: Path, run_id: int, seed: int):
    cfg = compose_config(config_name)
    cfg.experiment.data_path = str(data_root.resolve())
    cfg.experiment.num_body_points = 0
    cfg.experiment.model_name = "PAIRED_DEAL_SAMPLING_AUDIT"
    train, validation, _, spatial_dim, surf_channels, vol_channels, parameter_channels, _ = get_dataset(
        cfg.experiment
    )
    if spatial_dim != 3:
        raise ValueError("Only 3D datasets are supported")
    if run_id in validation.data:
        dataset = validation
    elif run_id in train.data:
        dataset = train
    else:
        raise ValueError(f"Case {run_id} is absent from both configured dataset splits")
    dataset.set_epoch(0)
    if hasattr(dataset, "geometry_epoch_seeded_sampling"):
        dataset.geometry_epoch_seeded_sampling = True
    dataset.deterministic_evaluation_seed = seed
    index = dataset.data.index(run_id)
    case_seed = seed + run_id * 1_000_003
    random.seed(case_seed)
    np.random.seed(case_seed % (2**32))
    torch.manual_seed(case_seed)
    item = unpack_item(dataset[index], parameter_channels)
    return cfg, dataset, item, (surf_channels, vol_channels, parameter_channels)


def query_data(args: argparse.Namespace) -> tuple[vtk.vtkPolyData, np.ndarray, np.ndarray, str]:
    if args.dataset == "drivaerml":
        surface = read_polydata(args.native_mesh)
        native = read_polydata(args.native_field_source)
        points = np.asarray(vtk_to_numpy(surface.GetPoints().GetData()), dtype=np.float32)
        native_points = np.asarray(vtk_to_numpy(native.GetPoints().GetData()), dtype=np.float32)
        if points.shape != native_points.shape or not np.array_equal(points, native_points):
            raise ValueError("DrivAerML geometry and native CFD boundary do not share point ordering")
        ground_truth = cell_values_at_points(native, "pMeanTrim")
        return surface, points, ground_truth, "native CFD surface"
    if args.dataset == "c_core":
        surface = read_polydata(args.native_mesh)
        points = np.asarray(vtk_to_numpy(surface.GetPoints().GetData()), dtype=np.float32)
        values = np.asarray(vtk_to_numpy(surface.GetPointData().GetArray("B_T")), dtype=np.float32)
        return surface, points, np.linalg.norm(values, axis=1), "native FEM surface"
    if args.dataset == "pump":
        volume = read_unstructured_grid(args.native_mesh)
        volume_bounds = volume.GetBounds()
        slice_z = (
            float(args.pump_slice_z)
            if args.pump_slice_z is not None
            else volume_bounds[4]
            + float(args.pump_slice_fraction) * (volume_bounds[5] - volume_bounds[4])
        )
        args.pump_slice_z = slice_z
        slice_mesh = native_volume_slice(
            volume,
            normal=(0.0, 0.0, 1.0),
            origin=(0.0, 0.0, slice_z),
        )
        points = np.asarray(vtk_to_numpy(slice_mesh.GetPoints().GetData()), dtype=np.float32)
        velocity = np.asarray(
            vtk_to_numpy(slice_mesh.GetPointData().GetArray("Velocity (m/s)")), dtype=np.float32
        )
        return slice_mesh, points, np.linalg.norm(velocity, axis=1), "native CFD blade-midspan slice"
    directory = args.native_mesh
    points = np.asarray(np.load(directory / "surface_mesh_points.npy", mmap_mode="r"))
    tetrahedra = np.asarray(np.load(directory / "volume_mesh_tetra.npy", mmap_mode="r"))
    temperature = np.asarray(np.load(directory / "fem_nodal_temperature.npy", mmap_mode="r"))
    volume = tetrahedral_mesh(points, tetrahedra)
    add_point_array(volume, "temperature", temperature)
    bounds = volume.GetBounds()
    slice_y = bounds[2] + float(args.heat_slice_fraction) * (bounds[3] - bounds[2])
    slice_mesh = native_volume_slice(volume, normal=(0.0, 1.0, 0.0), origin=(0.0, slice_y, 0.0))
    query_points = np.asarray(vtk_to_numpy(slice_mesh.GetPoints().GetData()), dtype=np.float32)
    ground_truth = np.asarray(
        vtk_to_numpy(slice_mesh.GetPointData().GetArray("temperature")), dtype=np.float32
    )
    return slice_mesh, query_points, ground_truth, "native FEM central slice"


def prepare_bundle(args: argparse.Namespace) -> None:
    from scripts import audit_paired_architecture_sampling as canonical

    spec = TASKS[args.dataset]
    rankings = ranked_cases(args.per_case_metrics, str(spec["selection_metric"]))
    constrained_selection = args.min_channels > 0 or args.waviness_category
    eligible_ids: list[int] | None = None
    if constrained_selection:
        if args.dataset != "heat_exchanger":
            raise ValueError("Geometry-constrained selection is currently defined only for heat exchange")
        manifest = json.loads((args.data_root / "preprocessed_manifest.json").read_text())
        eligible_ids = []
        candidate_ids = list(manifest["validation_ids"])
        if args.candidate_split == "all":
            candidate_ids += list(manifest["train_ids"])
        for case_id in candidate_ids:
            parameters = json.loads(
                (args.data_root / f"case_{int(case_id):05d}" / "case_metadata.json").read_text()
            )["parameters"]
            if len(parameters["channels"]) < args.min_channels:
                continue
            if args.waviness_category and parameters["waviness_category"] != args.waviness_category:
                continue
            eligible_ids.append(int(case_id))
        if args.run_id not in eligible_ids:
            raise ValueError(f"Case {args.run_id} does not satisfy the geometry constraints")
    elif not rankings or args.run_id not in {
        int(row["case_id"]) for row in rankings[: max(1, args.balanced_top_k)]
    }:
        leaders = rankings[: max(1, args.balanced_top_k)]
        raise ValueError(
            f"Case {args.run_id} is not in the allowed balanced low-error/large-gap pool: {leaders}"
        )
    cfg, dataset, item, _ = dataset_context(args.config, args.data_root, args.run_id, args.seed)
    geometry, _, _, _, _, _, density = item
    budget = input_budget(cfg)
    lower, span = bounds(dataset)
    view_seed = args.seed + args.run_id * 10_007
    names: list[str] = []
    encoders: list[np.ndarray] = []
    sources: list[str] = []
    for name, axis, offset in (("sine_x", 0, 101), ("sine_y", 1, 211)):
        sampled, _, _ = sample_geometry_view(
            geometry.unsqueeze(0),
            density.unsqueeze(0),
            budget,
            "sinusoidal_axis_mixture_wor",
            0.0,
            0.0,
            view_seed + offset,
            sinusoidal_axis=axis,
            sinusoidal_mix_fraction=1.0,
        )
        names.append(name)
        encoders.append(sampled[0].cpu().numpy().astype(np.float32))
        sources.append("native geometry")
    for method in spec["methods"]:
        for factor in (5, 10):
            path = canonical.remesh_path(args.dataset, args.remesh_root, args.run_id, method, factor)
            physical = canonical.remesh_geometry(
                path, budget, view_seed + factor * 10 + len(method), args.dataset
            )
            names.append(f"remesh_{method}_div{factor}")
            encoders.append(np.ascontiguousarray((physical - lower) / span, dtype=np.float32))
            sources.append(str(path.resolve()))
    mesh, query_points, ground_truth, query_source = query_data(args)
    if not np.isfinite(query_points).all() or not np.isfinite(ground_truth).all():
        raise ValueError("Native query mesh contains non-finite values")
    write_polydata(args.query_mesh, mesh, ground_truth)
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.bundle,
        condition_names=np.asarray(names),
        encoder_views=np.stack(encoders),
        query_points=query_points,
        ground_truth=ground_truth.astype(np.float32),
    )
    manifest = {
        "dataset": args.dataset,
        "run_id": args.run_id,
        "selection": {
            "rule": (
                "geometry-constrained candidate pool whose generated native-query relative-L2 outputs "
                "are ranked jointly for low DeAL error and a large Base-minus-DeAL gap"
                if constrained_selection
                else (
                    "native-slice-readable selection from the top balanced candidates ranked by normalized "
                    "distance to low DeAL relative error and large Base-minus-DeAL relative-error gap"
                    if args.balanced_top_k > 1
                    else "minimum normalized distance to low DeAL relative error and large Base-minus-DeAL relative-error gap"
                )
            ),
            "metric": spec["selection_metric"],
            "source": str(args.per_case_metrics.resolve()),
            "source_sha256": sha256(args.per_case_metrics),
            "selected": (
                None
                if constrained_selection
                else next(row for row in rankings if int(row["case_id"]) == args.run_id)
            ),
            "rankings": rankings,
            "balanced_top_k": max(1, args.balanced_top_k),
            "eligible_case_ids": eligible_ids,
            "constraints": {
                "minimum_channels": args.min_channels,
                "waviness_category": args.waviness_category,
                "candidate_split": args.candidate_split,
            },
        },
        "conditions": [
            {"name": name, "encoder_source": source} for name, source in zip(names, sources)
        ],
        "encoder_budget": budget,
        "query_kind": spec["query_kind"],
        "query_source": query_source,
        "query_points": int(query_points.shape[0]),
        "field": spec["field"],
        "pump_slice_z": float(args.pump_slice_z) if args.dataset == "pump" else None,
        "pump_slice_fraction": float(args.pump_slice_fraction) if args.dataset == "pump" else None,
        "heat_slice_fraction": (
            float(args.heat_slice_fraction) if args.dataset == "heat_exchanger" else None
        ),
        "bundle": str(args.bundle.resolve()),
        "query_mesh": str(args.query_mesh.resolve()),
    }
    args.bundle.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Prepared {args.dataset} case {args.run_id}: {len(names)} shifts, "
        f"{query_points.shape[0]:,} native queries -> {args.bundle}",
        flush=True,
    )


def scalar_field(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "first":
        return values[:, 0]
    if mode == "velocity_magnitude":
        return np.linalg.norm(values[:, 1:4], axis=1)
    if mode == "vector_magnitude":
        return np.linalg.norm(values, axis=1)
    raise ValueError(mode)


@torch.inference_mode()
def predict_scalar(
    model,
    device: torch.device,
    geometry: np.ndarray,
    queries: np.ndarray,
    params: torch.Tensor | None,
    query_kind: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    scalar_mode: str,
    chunk_size: int,
    seed: int,
) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    geometry_tensor = torch.from_numpy(np.ascontiguousarray(geometry)).unsqueeze(0).to(device)
    params_tensor = None if params is None else params.unsqueeze(0).to(device)
    output = np.empty((queries.shape[0], mean.numel()), dtype=np.float32)
    with torch.cuda.device(device), torch.autocast(device_type="cuda", dtype=torch.float16):
        encoded, latent_positions = model.encode(geometry_tensor, params_tensor)
        for start in range(0, queries.shape[0], chunk_size):
            stop = min(start + chunk_size, queries.shape[0])
            query = torch.from_numpy(np.ascontiguousarray(queries[start:stop])).unsqueeze(0).to(device)
            normalized = model.decode(encoded, latent_positions, params_tensor, query)[0]
            if query_kind == "surface":
                normalized = normalized[:, : model.surface_channels]
            else:
                normalized = normalized[:, model.surface_channels :]
            output[start:stop] = normalized.float().cpu().numpy()
    torch.cuda.synchronize(device)
    physical = output * std.numpy().reshape(1, -1) + mean.numpy().reshape(1, -1)
    return scalar_field(physical, scalar_mode).astype(np.float32)


def infer_bundle(args: argparse.Namespace) -> None:
    spec = TASKS[args.dataset]
    base_cfg, dataset, item, channels = dataset_context(
        args.base_config, args.data_root, args.run_id, args.seed
    )
    deal_cfg = compose_config(args.deal_config)
    _, _, _, _, _, params, _ = item
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This exporter requires CUDA inference")
    with np.load(args.bundle) as bundle:
        condition_names = [str(value) for value in bundle["condition_names"]]
        encoder_views = np.asarray(bundle["encoder_views"], dtype=np.float32)
        query_points = np.asarray(bundle["query_points"], dtype=np.float32)
        ground_truth = np.asarray(bundle["ground_truth"], dtype=np.float32)
    lower, span = bounds(dataset)
    normalized_queries = np.ascontiguousarray((query_points - lower) / span, dtype=np.float32)
    mean = dataset.mean_surf_data.float() if spec["query_kind"] == "surface" else dataset.mean_vol_data.float()
    std = dataset.std_surf_data.float() if spec["query_kind"] == "surface" else dataset.std_vol_data.float()
    sums = {"base": np.zeros_like(ground_truth, dtype=np.float64), "deal": np.zeros_like(ground_truth, dtype=np.float64)}
    raw: dict[str, np.ndarray] = {}
    checkpoint_records = {}
    for variant, cfg, checkpoint in (
        ("base", base_cfg, args.base_checkpoint),
        ("deal", deal_cfg, args.deal_checkpoint),
    ):
        model = build_smart(cfg, checkpoint, device, channels)
        checkpoint_records[variant] = checkpoint_metadata(checkpoint)
        for index, (condition, geometry) in enumerate(zip(condition_names, encoder_views), start=1):
            inference_seed = stable_seed(args.seed, args.run_id, condition, 0)
            prediction = predict_scalar(
                model,
                device,
                geometry,
                normalized_queries,
                params,
                str(spec["query_kind"]),
                mean,
                std,
                str(spec["scalar_mode"]),
                args.query_chunk_size,
                inference_seed,
            )
            error = np.abs(prediction - ground_truth)
            if not np.isfinite(error).all():
                raise FloatingPointError(f"{variant}/{condition} contains non-finite errors")
            sums[variant] += error
            raw[f"{variant}_absolute_error__{condition}"] = error.astype(np.float32)
            print(f"{variant} [{index}/{len(condition_names)}] {condition}", flush=True)
        del model
        torch.cuda.empty_cache()
    count = float(len(condition_names))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ground_truth=ground_truth,
        base_mean_absolute_error=(sums["base"] / count).astype(np.float32),
        deal_mean_absolute_error=(sums["deal"] / count).astype(np.float32),
        **raw,
    )
    manifest = json.loads(args.bundle.with_suffix(".json").read_text())
    manifest.update(
        {
            "aggregation": "equal-weight pointwise mean absolute error over every displayed shifted condition",
            "base_checkpoint": checkpoint_records["base"],
            "deal_checkpoint": checkpoint_records["deal"],
            "output": str(args.output.resolve()),
        }
    )
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_bundle(args)
    else:
        infer_bundle(args)


if __name__ == "__main__":
    main()
