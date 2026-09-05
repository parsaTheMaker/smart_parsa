#!/usr/bin/env python3
"""Export direct native-surface SMART and DeAL predictions for paper figures."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy


SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2  # noqa: E402
from data.pump_dataset import PumpDataset  # noqa: E402
from data.toy_heat_exchange_dataset import ToyHeatExchangeDataset  # noqa: E402
from scripts.compare_drivaerml_sampling_invariance import (  # noqa: E402
    build_model as build_drivaerml_model,
    load_cfg as load_drivaerml_config,
)
from scripts.compare_shift_endpoint_strategies import load_vtp_points, make_model  # noqa: E402


DEFAULTS = {
    "drivaerml": {
        "data_root": Path("/mnt/ssdraid/parsa/drivaerml_preprocessed"),
        "surface_template": "/mnt/ssdraid/parsa/drivaerml_surface_vtp/run_{run_id}/drivaer_{run_id}.vtp",
        "native_template": "/mnt/ssdraid/parsa/drivaerml_native_fields/run_{run_id}/boundary_{run_id}.vtp",
        "qem_template": (
            "/mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/quadric/run_{run_id}/"
            "drivaer_{run_id}_faces_div10.vtp"
        ),
        "base_config": "drivaerml",
        "deal_config": "drivaerml_satloss7_range100",
        "field": "pressure",
    },
    "pump": {
        "data_root": Path("/mnt/data/parsa/shift_pump_random1400_preprocessed"),
        "surface_template": "/mnt/data/parsa/shift_pump_raw_random1400/sample_{run_id:06d}/merged_surfaces.vtp",
        "native_template": "/mnt/data/parsa/shift_pump_raw_random1400/sample_{run_id:06d}/merged_surfaces.vtp",
        "qem_template": (
            "/mnt/data/parsa/shift_pump_random1400_surface_vtp_remesh_v4/quadric/"
            "sample_{run_id:06d}/merged_surfaces_faces_div10.vtp"
        ),
        "base_config": "pump",
        "deal_config": "pump_deal_from_smart_full",
        "field": "wall_shear_magnitude",
    },
    "heat_exchanger": {
        "data_root": Path("/mnt/ssdraid/parsa/toy_heat_exchange_fem_v1"),
        "surface_template": (
            "/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp/case_{run_id:05d}/"
            "heat_exchange_case_{run_id:05d}_surface.vtp"
        ),
        "native_template": (
            "/mnt/ssdraid/parsa/toy_heat_exchange_fem_v1/case_{run_id:05d}/"
            "surface_fem_face_flux.npy"
        ),
        "qem_template": (
            "/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_remesh_v4/quadric/"
            "case_{run_id:05d}/heat_exchange_case_{run_id:05d}_surface_faces_div10.vtp"
        ),
        "base_config": "toy_heat_exchange",
        "deal_config": "toy_heat_exchange_satloss7",
        "field": "outward_heat_flux",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DEFAULTS), required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--surface-mesh", type=Path)
    parser.add_argument("--native-field-source", type=Path)
    parser.add_argument("--qem-input", type=Path)
    parser.add_argument("--base-config")
    parser.add_argument("--deal-config")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--deal-checkpoint", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--query-chunk-size", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_polydata(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    output = vtk.vtkPolyData()
    output.ShallowCopy(reader.GetOutput())
    if output.GetNumberOfPoints() == 0:
        raise ValueError(f"No points found in {path}")
    return output


def point_values(poly: vtk.vtkPolyData, name: str) -> np.ndarray:
    array = poly.GetPointData().GetArray(name)
    if array is None:
        raise KeyError(f"Point array {name!r} is absent")
    return np.asarray(vtk_to_numpy(array), dtype=np.float32)


def cell_values_at_points(poly: vtk.vtkPolyData, name: str) -> np.ndarray:
    if poly.GetCellData().GetArray(name) is None:
        raise KeyError(f"Cell array {name!r} is absent")
    conversion = vtk.vtkCellDataToPointData()
    conversion.SetInputData(poly)
    conversion.PassCellDataOff()
    conversion.Update()
    return point_values(conversion.GetOutput(), name).copy()


def heat_flux_at_mesh_points(surface: vtk.vtkPolyData, face_flux_path: Path) -> np.ndarray:
    flux = np.asarray(np.load(face_flux_path, mmap_mode="r"), dtype=np.float32).reshape(-1)
    if flux.shape[0] != surface.GetNumberOfPolys():
        raise ValueError(
            f"Heat-flux/face mismatch: {flux.shape[0]:,} values for "
            f"{surface.GetNumberOfPolys():,} polygons"
        )
    result = np.zeros(surface.GetNumberOfPoints(), dtype=np.float64)
    counts = np.zeros(surface.GetNumberOfPoints(), dtype=np.int64)
    connectivity = surface.GetPolys().GetConnectivityArray()
    offsets = surface.GetPolys().GetOffsetsArray()
    cells = np.asarray(vtk_to_numpy(connectivity), dtype=np.int64)
    starts = np.asarray(vtk_to_numpy(offsets), dtype=np.int64)
    if not np.all(np.diff(starts) == 3):
        raise ValueError("Heat Exchanger display surface is not triangular.")
    faces = cells.reshape(-1, 3)
    for corner in range(3):
        np.add.at(result, faces[:, corner], flux)
        np.add.at(counts, faces[:, corner], 1)
    valid = counts > 0
    result[valid] /= counts[valid]
    return result.astype(np.float32)


def dataset_for(name: str, root: Path):
    kwargs = dict(
        saved_folder=str(root),
        if_test=True,
        geometry_points=0,
        surface_points=0,
        volume_points=0,
        scale_positions=False,
    )
    if name == "drivaerml":
        return AhmedMLDatasetV2(require_preprocessed=True, **kwargs)
    if name == "pump":
        return PumpDataset(**kwargs)
    return ToyHeatExchangeDataset(**kwargs)


def parse_devices(value: str) -> tuple[torch.device, torch.device]:
    devices = [torch.device(item.strip()) for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one inference device is required.")
    if len(devices) == 1:
        devices *= 2
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return devices[0], devices[1]


def make_models(
    dataset_name: str,
    base_config: str,
    deal_config: str,
    base_checkpoint: Path,
    deal_checkpoint: Path,
    base_device: torch.device,
    deal_device: torch.device,
    chunk_size: int,
):
    if dataset_name == "drivaerml":
        base = build_drivaerml_model(
            load_drivaerml_config(base_config), str(base_checkpoint), base_device, chunk_size
        ).to(base_device)
        deal = build_drivaerml_model(
            load_drivaerml_config(deal_config), str(deal_checkpoint), deal_device, chunk_size
        ).to(deal_device)
        return base, deal, 131072
    base, _base_cfg, base_budget = make_model(
        dataset_name, base_config, base_checkpoint, base_device, chunk_size
    )
    deal, _deal_cfg, deal_budget = make_model(
        dataset_name, deal_config, deal_checkpoint, deal_device, chunk_size
    )
    if base_budget != deal_budget:
        raise ValueError(f"Base and DeAL encoder budgets differ: {base_budget} vs {deal_budget}")
    return base, deal, base_budget


@torch.inference_mode()
def decode_surface(
    model: torch.nn.Module,
    device: torch.device,
    geometry: np.ndarray,
    queries: np.ndarray,
    params: np.ndarray | None,
    minimum: np.ndarray,
    span: np.ndarray,
    surface_mean: np.ndarray,
    surface_std: np.ndarray,
    chunk_size: int,
    seed: int,
) -> np.ndarray:
    geometry_norm = torch.from_numpy(
        np.ascontiguousarray((geometry - minimum[None, :]) / span[None, :], dtype=np.float32)
    )
    params_tensor = None if params is None else torch.from_numpy(np.ascontiguousarray(params)).unsqueeze(0)
    output = np.empty((queries.shape[0], surface_mean.shape[0]), dtype=np.float32)
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        geometry_tensor = geometry_norm.unsqueeze(0).to(device, non_blocking=True)
        if params_tensor is not None:
            params_tensor = params_tensor.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            encoded, latent_positions = model.encode(geometry_tensor, params_tensor)
            for start in range(0, queries.shape[0], chunk_size):
                stop = min(start + chunk_size, queries.shape[0])
                normalized = np.ascontiguousarray(
                    (queries[start:stop] - minimum[None, :]) / span[None, :], dtype=np.float32
                )
                query = torch.from_numpy(normalized).unsqueeze(0).to(device, non_blocking=True)
                prediction = model.decode(encoded, latent_positions, params_tensor, query)
                output[start:stop] = prediction[0, :, : surface_mean.shape[0]].float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return output * surface_std[None, :] + surface_mean[None, :]


def main() -> None:
    args = parse_args()
    defaults = DEFAULTS[args.dataset]
    data_root = args.data_root or defaults["data_root"]
    surface_path = args.surface_mesh or Path(str(defaults["surface_template"]).format(run_id=args.run_id))
    native_path = args.native_field_source or Path(
        str(defaults["native_template"]).format(run_id=args.run_id)
    )
    qem_path = args.qem_input or Path(str(defaults["qem_template"]).format(run_id=args.run_id))
    base_config = args.base_config or str(defaults["base_config"])
    deal_config = args.deal_config or str(defaults["deal_config"])
    for path in (data_root, surface_path, native_path, qem_path, args.base_checkpoint, args.deal_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    surface = read_polydata(surface_path)
    queries = np.asarray(vtk_to_numpy(surface.GetPoints().GetData()), dtype=np.float32)
    qem_points = load_vtp_points(qem_path)
    if args.dataset == "drivaerml":
        native = read_polydata(native_path)
        native_points = np.asarray(vtk_to_numpy(native.GetPoints().GetData()), dtype=np.float32)
        if queries.shape != native_points.shape or not np.array_equal(queries, native_points):
            raise ValueError("DrivAerML display and native CFD surfaces do not share point ordering.")
        ground_truth = cell_values_at_points(native, "pMeanTrim")
    elif args.dataset == "pump":
        ground_truth = np.linalg.norm(point_values(surface, "Wall Shear Stress (N/m²)"), axis=1)
    else:
        ground_truth = heat_flux_at_mesh_points(surface, native_path)

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
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, args.run_id, 901]))
    if qem_points.shape[0] > encoder_budget:
        indices = np.sort(rng.choice(qem_points.shape[0], size=encoder_budget, replace=False))
        geometry = np.ascontiguousarray(qem_points[indices], dtype=np.float32)
    else:
        geometry = np.ascontiguousarray(qem_points, dtype=np.float32)
    minimum = dataset.min_pos.detach().cpu().numpy().astype(np.float32)
    span = (
        dataset.position_span.detach().cpu().numpy().astype(np.float32)
        if hasattr(dataset, "position_span")
        else (dataset.max_pos - dataset.min_pos).detach().cpu().numpy().astype(np.float32)
    )
    surface_mean = dataset.mean_surf_data.detach().cpu().numpy().astype(np.float32)
    surface_std = dataset.std_surf_data.detach().cpu().numpy().astype(np.float32)
    params = dataset.get_case_params(args.run_id) if hasattr(dataset, "get_case_params") else None
    print(
        f"[{args.dataset}] QEM-10 input={geometry.shape[0]:,}; native queries={queries.shape[0]:,}; "
        f"Base->{base_device}, DeAL->{deal_device}",
        flush=True,
    )
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
                args.seed + 101,
            )
            for model, device in ((base_model, base_device), (deal_model, deal_device))
        ]
        base_full, deal_full = (future.result() for future in futures)

    if args.dataset == "pump":
        base = np.linalg.norm(base_full[:, 4:7], axis=1)
        deal = np.linalg.norm(deal_full[:, 4:7], axis=1)
    else:
        base = base_full[:, 0]
        deal = deal_full[:, 0]
    valid = np.zeros(queries.shape[0], dtype=bool)
    connectivity = np.asarray(vtk_to_numpy(surface.GetPolys().GetConnectivityArray()), dtype=np.int64)
    valid[np.unique(connectivity)] = True
    if not all(np.isfinite(values[valid]).all() for values in (ground_truth, base, deal)):
        raise RuntimeError("Native-surface export contains non-finite rendered values.")

    field = str(defaults["field"])
    arrays = {
        f"ground_truth_{field}": ground_truth,
        f"base_{field}": base,
        f"deal_{field}": deal,
        f"base_absolute_error_{field}": np.abs(base - ground_truth),
        f"deal_absolute_error_{field}": np.abs(deal - ground_truth),
        "rendered_vertex_mask": valid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    summary = {
        "dataset": args.dataset,
        "run_id": args.run_id,
        "condition": "QEM 10x",
        "field": field,
        "encoder_points": int(geometry.shape[0]),
        "native_surface_queries": int(queries.shape[0]),
        "rendered_surface_vertices": int(valid.sum()),
        "surface_mesh": str(surface_path.resolve()),
        "native_field_source": str(native_path.resolve()),
        "qem_input": str(qem_path.resolve()),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "deal_checkpoint": str(args.deal_checkpoint.resolve()),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
