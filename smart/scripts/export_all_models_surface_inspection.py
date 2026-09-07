#!/usr/bin/env python3
"""Export one surface-native QEM-10 inspection VTK per benchmark."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from omegaconf import OmegaConf

from data.datasets import get_dataset
from scripts.audit_new_architecture_base_sampling import bounds, compose_config, unpack_item
from scripts.audit_paired_architecture_sampling import input_budget, remesh_geometry, remesh_path
from scripts.export_all_models_field_inspection import DEFAULTS, MODEL_SPECS, ModelSpec, load_model, safe_name


ROOT = Path(__file__).resolve().parents[2]
SCALARS = {
    "drivaerml": {"pressure": 0},
    "pump": {"pressure": 0},
    "heat_exchanger": {"outward_heat_flux": 0},
    "c_core": {},
}
VECTORS = {
    "drivaerml": {"normal": (1, 2, 3), "wall_shear": (4, 5, 6)},
    "pump": {"velocity": (1, 2, 3), "wall_shear": (4, 5, 6)},
    "heat_exchanger": {},
    "c_core": {"B": (0, 1, 2)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--case-id", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--remesh-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--models", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--joint-query-chunk", type=int, default=65536)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "final" / "all_models_qem10_field_inspection")
    return parser.parse_args()


def heat_surface(root: Path, case_id: int) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    directory = root / f"case_{case_id:05d}"
    points = np.asarray(np.load(directory / "surface_mesh_points.npy", mmap_mode="r"), dtype=np.float32)
    faces = np.asarray(np.load(directory / "surface_mesh_faces.npy", mmap_mode="r"), dtype=np.int64)
    face_flux = np.asarray(np.load(directory / "surface_fem_face_flux.npy", mmap_mode="r"), dtype=np.float32)
    used = np.unique(faces)
    reindex = np.full(len(points), -1, dtype=np.int64)
    reindex[used] = np.arange(len(used), dtype=np.int64)
    points = points[used]
    faces = reindex[faces]
    triangles = points[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    weighted = np.zeros(len(points), dtype=np.float64)
    weights = np.zeros(len(points), dtype=np.float64)
    for column in range(3):
        np.add.at(weighted, faces[:, column], face_flux * areas)
        np.add.at(weights, faces[:, column], areas)
    if np.any(weights <= 0):
        raise RuntimeError("Extracted Heat Exchanger boundary contains isolated vertices")
    target = (weighted / weights).astype(np.float32)[:, None]
    vtk_faces = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel()
    return pv.PolyData(points, vtk_faces), points, target


def matching_mesh(path: Path, points: np.ndarray) -> pv.PolyData:
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if mesh.n_points != len(points) or not np.allclose(mesh.points, points, rtol=0.0, atol=2.0e-6):
        raise RuntimeError(f"Surface topology does not align with preprocessed point fields: {path}")
    mesh = mesh.copy()
    mesh.clear_data()
    return mesh


def full_surface(dataset: str, root: Path, case_id: int) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    if dataset == "heat_exchanger":
        return heat_surface(root, case_id)
    if dataset == "drivaerml":
        directory = root / f"run_{case_id}"
        points = np.asarray(np.load(directory / "surface_coords.npy", mmap_mode="r"), dtype=np.float32)
        target = np.column_stack(
            (
                np.load(directory / "surface_pMeanTrim.npy", mmap_mode="r"),
                np.load(directory / "surface_normals.npy", mmap_mode="r"),
                np.load(directory / "surface_wallShearStressMeanTrim_x.npy", mmap_mode="r"),
                np.load(directory / "surface_wallShearStressMeanTrim_y.npy", mmap_mode="r"),
                np.load(directory / "surface_wallShearStressMeanTrim_z.npy", mmap_mode="r"),
            )
        ).astype(np.float32, copy=False)
        return pv.PolyData(points), points, target
    directory = root / (f"run_{case_id}" if dataset == "pump" else f"case_{case_id:05d}")
    points = np.asarray(np.load(directory / "surface_coords.npy", mmap_mode="r"), dtype=np.float32)
    target = np.asarray(np.load(directory / "surface_data.npy", mmap_mode="r"), dtype=np.float32)
    if dataset == "pump":
        path = Path("/mnt/data/parsa/shift_pump_raw_random1400") / f"sample_{case_id:06d}" / "merged_surfaces.vtp"
    else:
        path = Path("/mnt/data/parsa/c_core_magnetic_fem_v1_raw") / f"case_{case_id:05d}" / f"case_{case_id:05d}_solid_surface.vtp"
    return matching_mesh(path, points), points, target


def add_target(mesh: pv.PolyData, dataset: str, target: np.ndarray) -> None:
    for name, channel in SCALARS[dataset].items():
        mesh.point_data[f"gt_{name}"] = target[:, channel]
    for name, channels in VECTORS[dataset].items():
        mesh.point_data[f"gt_{name}"] = target[:, channels]


def add_prediction(mesh: pv.PolyData, dataset: str, prediction: np.ndarray, target: np.ndarray, variant: str, model: str) -> None:
    prefix = f"{variant}_{safe_name(model)}"
    for name, channel in SCALARS[dataset].items():
        mesh.point_data[f"{prefix}_{name}"] = prediction[:, channel]
        mesh.point_data[f"abs_error_{prefix}_{name}"] = np.abs(prediction[:, channel] - target[:, channel])
    for name, channels in VECTORS[dataset].items():
        mesh.point_data[f"{prefix}_{name}"] = prediction[:, channels]
        mesh.point_data[f"abs_error_{prefix}_{name}_magnitude"] = np.linalg.vector_norm(
            prediction[:, channels] - target[:, channels], axis=1
        )


def infer_surface(
    model,
    model_slug: str,
    device: torch.device,
    geometry: torch.Tensor,
    surface_queries: torch.Tensor,
    volume_dummy: torch.Tensor,
    params: torch.Tensor | None,
    seed: int,
    joint_query_chunk: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    geometry_gpu = geometry.unsqueeze(0).to(device, non_blocking=True)
    params_gpu = None if params is None else params.unsqueeze(0).to(device, non_blocking=True)
    volume_gpu = volume_dummy.unsqueeze(0).to(device, non_blocking=True)
    chunks = []
    # Transolver++ and MSPT jointly process geometry and queries, so their
    # established inference methods need an outer query chunk for native clouds.
    step = joint_query_chunk if model_slug in {"transolverpp", "mspt", "ab_upt"} else len(surface_queries)
    for start in range(0, len(surface_queries), step):
        query_gpu = surface_queries[start : start + step].unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda" and model_slug != "lno",
        ):
            pred_surface, _ = model.inference(geometry_gpu, query_gpu, volume_gpu, params_gpu)
        chunks.append(pred_surface[0].float().cpu())
        del query_gpu, pred_surface
    prediction = torch.cat(chunks, dim=0).numpy()
    if not np.isfinite(prediction).all():
        raise FloatingPointError(f"{model_slug} produced non-finite surface predictions")
    return prediction


def main() -> None:
    args = parse_args()
    default_case, default_data, default_remesh = DEFAULTS[args.dataset]
    case_id = default_case if args.case_id is None else args.case_id
    data_root = default_data if args.data_root is None else args.data_root
    remesh_root = default_remesh if args.remesh_root is None else args.remesh_root
    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    specs = [spec for spec in MODEL_SPECS[args.dataset] if not selected or spec.slug in selected]
    usable: list[ModelSpec] = []
    for spec in specs:
        base = args.checkpoint_root / spec.base_checkpoint
        deal = args.checkpoint_root / spec.deal_checkpoint
        if base.is_file() and deal.is_file():
            usable.append(spec)
        elif not args.allow_missing:
            raise FileNotFoundError(f"Incomplete {spec.slug} pair: base={base.is_file()}, DeAL={deal.is_file()}")
        else:
            print(f"[skip] {spec.slug}: incomplete base/DeAL pair", flush=True)
    if not usable:
        raise RuntimeError("No complete base/DeAL checkpoint pairs")

    canonical = compose_config(usable[0].deal_config)
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(canonical, resolve=True))
    dataset_cfg.experiment.data_path = str(data_root.resolve())
    dataset_cfg.experiment.num_body_points = 0
    dataset_cfg.experiment.model_name = "DEAL_SURFACE_INSPECTION"
    _, dataset, _, spatial_dim, surface_channels, volume_channels, parameter_channels, _ = get_dataset(dataset_cfg.experiment)
    if spatial_dim != 3:
        raise ValueError("Only 3D datasets are supported")
    dataset.set_epoch(0)
    index = dataset.data.index(case_id)
    full_geometry, _, _, volume_queries, _, params, _ = unpack_item(dataset[index], parameter_channels)
    lower, span = bounds(dataset)
    mesh, surface_points, surface_target = full_surface(args.dataset, data_root, case_id)
    if surface_target.shape[1] != surface_channels:
        raise ValueError(f"Surface target channels are {surface_target.shape[1]}, expected {surface_channels}")
    normalized_surface = torch.from_numpy(np.ascontiguousarray((surface_points - lower) / span, dtype=np.float32))
    volume_dummy = volume_queries[:1].contiguous()
    add_target(mesh, args.dataset, surface_target)

    qem_path = remesh_path(args.dataset, remesh_root, case_id, "quadric", 10)
    geometry_cache: dict[int, torch.Tensor] = {}
    device = torch.device(args.device)
    channels = (surface_channels, volume_channels, parameter_channels)
    for spec_index, spec in enumerate(usable):
        base_cfg, deal_cfg = compose_config(spec.base_config), compose_config(spec.deal_config)
        declared_base, declared_deal = input_budget(base_cfg), input_budget(deal_cfg)
        budget = spec.encoder_budget or declared_base
        if spec.encoder_budget is None and declared_base != declared_deal:
            raise ValueError(f"{spec.slug}: unmatched encoder budgets {declared_base}/{declared_deal}")
        if budget not in geometry_cache:
            sampled = remesh_geometry(qem_path, budget, args.seed + case_id * 1009 + budget * 7, args.dataset)
            geometry_cache[budget] = torch.from_numpy(np.ascontiguousarray((sampled - lower) / span, dtype=np.float32))
        shared_seed = args.seed + case_id * 65537 + spec_index * 4099
        for variant, cfg, name in (
            ("base", base_cfg, spec.base_checkpoint),
            ("deal", deal_cfg, spec.deal_checkpoint),
        ):
            checkpoint = args.checkpoint_root / name
            print(f"[{args.dataset}] {spec.slug}/{variant}: QEM-10 input -> {len(surface_points)} native surface queries", flush=True)
            model = load_model(spec, cfg, checkpoint, device, channels)
            prediction_std = infer_surface(
                model,
                spec.slug,
                device,
                geometry_cache[budget],
                normalized_surface,
                volume_dummy,
                params,
                shared_seed,
                args.joint_query_chunk,
            )
            prediction = prediction_std * dataset.std_surf_data.numpy()[None, :] + dataset.mean_surf_data.numpy()[None, :]
            add_prediction(mesh, args.dataset, prediction.astype(np.float32), surface_target, variant, spec.slug)
            del model, prediction_std, prediction
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    mesh.field_data["case_id"] = np.asarray([case_id], dtype=np.int32)
    mesh.field_data["qem_reduction_factor"] = np.asarray([10], dtype=np.int32)
    mesh.field_data["complete_model_pairs"] = np.asarray([len(usable)], dtype=np.int32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.dataset}_case_{case_id}_qem10_input_native_surface_all_models_base_deal.vtk"
    mesh.save(output, binary=True)
    check = pv.read(output)
    if check.n_points != len(surface_points) or check.n_cells != mesh.n_cells:
        raise RuntimeError("VTK read-back changed surface topology")
    if set(check.point_data) != set(mesh.point_data):
        raise RuntimeError("VTK read-back lost field arrays")
    for name in check.point_data:
        if not np.isfinite(np.asarray(check.point_data[name])).all():
            raise FloatingPointError(f"Non-finite VTK array: {name}")
    print(f"[saved] {output}: points={check.n_points}, cells={check.n_cells}, arrays={len(check.point_data)}", flush=True)


if __name__ == "__main__":
    main()
