#!/usr/bin/env python3
"""Export paired base/DeAL fields for original and QEM-div10 inputs.

Each output is a ParaView-readable VTM containing a shared surface-query block,
a shared volume-query block, and the encoder source geometry. Ground truth,
physical-unit predictions, and pointwise absolute errors are stored together.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
from omegaconf import OmegaConf

from data.datasets import get_dataset
from models.ab_upt import ABUPT
from models.geo_fno import GeoFNO
from models.lno import LNO
from models.mspt import MSPT
from models.point_transformer_v3 import PointTransformerV3
from models.pointnet2_ssg import PointNet2SSG
from models.smart.smart import SMART
from models.transolverpp import TransolverPP
from scripts.audit_new_architecture_base_sampling import bounds, checkpoint_state, compose_config, unpack_item
from scripts.audit_paired_architecture_sampling import input_budget, remesh_geometry, remesh_path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    base_config: str
    deal_config: str
    base_checkpoint: str
    deal_checkpoint: str
    encoder_budget: int | None = None


CONSTRUCTORS = {
    "smart": SMART,
    "ab_upt": ABUPT,
    "geo_fno": GeoFNO,
    "pointnet2_ssg": PointNet2SSG,
    "lno": LNO,
    "mspt": MSPT,
    "transolverpp": TransolverPP,
    "point_transformer_v3": PointTransformerV3,
}


MODEL_SPECS: dict[str, tuple[ModelSpec, ...]] = {
    "drivaerml": (
        ModelSpec("smart", "drivaerml", "drivaerml_satloss7_range100", "smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt", "smart-satloss7-range100-smart-satloss7-range100-from-smart-150ep-drivaerml-s42_best.pt"),
        ModelSpec("transolverpp", "drivaerml_transolverpp", "drivaerml_transolverpp_satloss7", "transolverpp-transolverpp-drivaerml-uniform-epochseeded-gpu0-200ep-drivaerml-s42_best.pt", "transolverpp-satloss7-transolverpp-satloss7-drivaerml-65k-drivaerml-s42_best.pt"),
        ModelSpec("pointnet2_ssg", "drivaerml_pointnet2_ssg", "drivaerml_pointnet2_ssg_satloss7", "pointnet2-ssg-pointnet2-ssg-drivaerml-65k-v2-drivaerml-s42_best.pt", "pointnet2-ssg-satloss7-pointnet2-ssg-satloss7-drivaerml-65k-drivaerml-s42_best.pt"),
        ModelSpec("lno", "drivaerml_lno", "drivaerml_lno_satloss7", "lno-lno-drivaerml-65k-drivaerml-s42_best.pt", "lno-satloss7-lno-satloss7-drivaerml-65k-drivaerml-s42_best.pt"),
        ModelSpec("mspt", "drivaerml_mspt", "drivaerml_mspt_satloss7", "mspt-mspt-drivaerml-uniform-epochseeded-gpu6-200ep-drivaerml-s42_best.pt", "mspt-satloss7-mspt-satloss7-drivaerml-65k-drivaerml-s42_best.pt"),
        # This DeAL checkpoint predates the density-sensitive PTv3 runtime
        # configuration, so retain its recorded standard PTv3 inference setup.
        ModelSpec("point_transformer_v3", "drivaerml_point_transformer_v3_density_sensitive", "drivaerml_point_transformer_v3_satloss7", "point-transformer-v3-ptv3-density-sensitive-drivaerml-drivaerml-s42_best.pt", "point-transformer-v3-satloss7-ptv3-satloss7-density-sensitive-drivaerml-131k-drivaerml-s42_best.pt"),
        ModelSpec("ab_upt", "drivaerml_ab_upt", "drivaerml_ab_upt_deal_from_base", "ab-upt-expanded-v3-drivaerml-s42_best.pt", "ab-upt-deal-from-base-150ep-drivaerml-s42_best.pt"),
        ModelSpec("geo_fno", "drivaerml_geo_fno", "drivaerml_geo_fno_deal_from_base", "geofno-medium-v2-raw65k-drivaerml-s42_best.pt", "geofno-deal-medium-v2-from-base-150ep-drivaerml-s42_best.pt", 65536),
    ),
    "heat_exchanger": (
        ModelSpec("smart", "toy_heat_exchange", "toy_heat_exchange_satloss7", "smart-toy-heat-exchange-heat-exchange-base-ratio-aligned-toyheatexchange-s42_best.pt", "smart-toy-heat-exchange-satloss7-heat-exchange-satloss-ratio-aligned-toyheatexchange-s42_best.pt"),
        ModelSpec("transolverpp", "toy_heat_exchange_transolverpp", "toy_heat_exchange_transolverpp_satloss7", "transolverpp-toy-heat-exchange-transolverpp-base-toyheatexchange-s42_best.pt", "transolverpp-toy-heat-exchange-satloss7-transolverpp-satloss-from-base-toyheatexchange-s42_best.pt"),
        ModelSpec("pointnet2_ssg", "toy_heat_exchange_pointnet2_ssg", "toy_heat_exchange_pointnet2_ssg_satloss7", "pointnet2-ssg-toy-heat-exchange-pointnet2-ssg-base-toyheatexchange-s42_best.pt", "pointnet2-ssg-toy-heat-exchange-satloss7-pointnet2-ssg-satloss7-from-base-toyheatexchange-s42_best.pt"),
        ModelSpec("lno", "toy_heat_exchange_lno", "toy_heat_exchange_lno_satloss7", "lno-toy-heat-exchange-lno-base-toyheatexchange-s42_best.pt", "lno-toy-heat-exchange-satloss7-lno-satloss7-from-base-toyheatexchange-s42_best.pt"),
        ModelSpec("mspt", "toy_heat_exchange_mspt", "toy_heat_exchange_mspt_satloss7", "mspt-toy-heat-exchange-mspt-base-toyheatexchange-s42_best.pt", "mspt-toy-heat-exchange-satloss7-mspt-satloss-from-base-toyheatexchange-s42_best.pt"),
        ModelSpec("point_transformer_v3", "toy_heat_exchange_point_transformer_v3", "toy_heat_exchange_point_transformer_v3_satloss7", "point-transformer-v3-toy-heat-exchange-point-transformer-v3-base-toyheatexchange-s42_best.pt", "point-transformer-v3-toy-heat-exchange-satloss7-point-transformer-v3-satloss7-from-base-toyheatexchange-s42_best.pt"),
        ModelSpec("ab_upt", "toy_heat_exchange_ab_upt", "toy_heat_exchange_ab_upt_deal_from_base", "ab-upt-toy-heat-exchange-expanded-v3-toyheatexchange-s42_best.pt", "ab-upt-heat-exchange-deal-from-base-150ep-toyheatexchange-s42_best.pt"),
        ModelSpec("geo_fno", "toy_heat_exchange_geo_fno", "toy_heat_exchange_geo_fno_deal_from_base", "geofno-heat-exchanger-medium-v2-raw65k-toyheatexchange-s42_best.pt", "geofno-heat-exchanger-deal-medium-v2-from-base-150ep-toyheatexchange-s42_best.pt"),
    ),
    "pump": (
        ModelSpec("smart", "pump", "pump_deal_from_smart_full", "smart-pump-random1400-base-16k-pump-s42_best.pt", "smart-pump-deal-random1400-from-smart-150ep-pump-s42_best.pt"),
        ModelSpec("transolverpp", "pump_transolverpp", "pump_transolverpp_deal_from_base", "transolverpp-pump-servus06-base-v1-pump-s42_best.pt", "transolverpp-pump-deal-from-base-150ep-pump-s42_best.pt"),
        ModelSpec("pointnet2_ssg", "pump_pointnet2_ssg", "pump_pointnet2_ssg_deal_from_base", "pointnet2-ssg-pump-servus06-base-v1-pump-s42_best.pt", "pointnet2-ssg-pump-deal-from-base-150ep-pump-s42_best.pt"),
        ModelSpec("lno", "pump_lno", "pump_lno_deal_from_base", "lno-pump-servus06-base-v1-pump-s42_best.pt", "lno-pump-deal-from-base-150ep-pump-s42_best.pt"),
        ModelSpec("mspt", "pump_mspt", "pump_mspt_deal_from_base", "mspt-pump-servus06-base-v1-pump-s42_best.pt", "mspt-pump-deal-from-base-150ep-pump-s42_best.pt"),
        ModelSpec("point_transformer_v3", "pump_point_transformer_v3", "pump_point_transformer_v3_deal_from_base", "point-transformer-v3-pump-servus06-base-v1-pump-s42_best.pt", "point-transformer-v3-pump-deal-from-base-150ep-pump-s42_best.pt"),
        ModelSpec("ab_upt", "pump_ab_upt", "pump_ab_upt_deal_from_base", "ab-upt-pump-servus06-base-v1-pump-s42_best.pt", "ab-upt-pump-deal-from-base-150ep-pump-s42_best.pt"),
        ModelSpec("geo_fno", "pump_geo_fno", "pump_geo_fno_deal_from_base", "geofno-pump-medium-v2-raw16k-pump-s42_best.pt", "geofno-pump-deal-medium-v2-from-base-150ep-pump-s42_best.pt"),
    ),
    "c_core": (
        ModelSpec("smart", "c_core_magnetic_smart", "c_core_magnetic_smart_deal_from_base", "smart-c-core-magnetic-ccoremagnetic-s42_best.pt", "smart-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt"),
        ModelSpec("transolverpp", "c_core_magnetic_transolverpp", "c_core_magnetic_transolverpp_deal_from_base", "transolverpp-c-core-magnetic-ccoremagnetic-s42_best.pt", "transolverpp-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt"),
        ModelSpec("pointnet2_ssg", "c_core_magnetic_pointnet2_ssg", "c_core_magnetic_pointnet2_ssg_deal_from_base", "pointnet2-ssg-c-core-magnetic-ccoremagnetic-s42_best.pt", "pointnet2-ssg-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt"),
        ModelSpec("lno", "c_core_magnetic_lno", "c_core_magnetic_lno_deal_from_base", "lno-c-core-magnetic-ccoremagnetic-s42_best.pt", "lno-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt"),
        ModelSpec("mspt", "c_core_magnetic_mspt", "c_core_magnetic_mspt_deal_from_base", "mspt-c-core-magnetic-ccoremagnetic-s42_best.pt", "mspt-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt"),
        ModelSpec("point_transformer_v3", "c_core_magnetic_point_transformer_v3", "c_core_magnetic_point_transformer_v3_deal_from_base", "point-transformer-v3-c-core-magnetic-ccoremagnetic-s42_best.pt", "point-transformer-v3-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt"),
        ModelSpec("ab_upt", "c_core_magnetic_ab_upt", "c_core_magnetic_ab_upt_deal_from_base", "ab-upt-c-core-magnetic-ccoremagnetic-s42_best.pt", "ab-upt-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt"),
        ModelSpec("geo_fno", "c_core_magnetic_geo_fno", "c_core_magnetic_geo_fno_deal_from_base", "geofno-c-core-magnetic-ccoremagnetic-s42_best.pt", "geofno-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt"),
    ),
}


DEFAULTS = {
    "drivaerml": (1, Path("/mnt/ssdraid/parsa/drivaerml_preprocessed"), Path("/mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4")),
    "heat_exchanger": (256, Path("/mnt/ssdraid/parsa/toy_heat_exchange_fem_v1"), Path("/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_remesh_v4")),
    "pump": (33, Path("/mnt/data/parsa/shift_pump_random1400_preprocessed"), Path("/mnt/data/parsa/shift_pump_random1400_surface_vtp_remesh_v4")),
    "c_core": (256, Path("/mnt/data/parsa/c_core_magnetic_fem_v1"), Path("/mnt/data/parsa/c_core_magnetic_surface_vtp_remesh_v4")),
}


VECTOR_GROUPS = {
    "drivaerml": {"normal": (1, 2, 3), "wall_shear": (4, 5, 6)},
    "pump": {"velocity": (1, 2, 3), "wall_shear": (4, 5, 6)},
    "c_core": {"B": (0, 1, 2)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--case-id", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--remesh-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument(
        "--checkpoint-policy",
        choices=("best", "last"),
        default="best",
        help="Select best-validation or final-epoch checkpoints for paired field export.",
    )
    parser.add_argument("--models", default="", help="Optional comma-separated architecture slugs.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "final" / "all_models_qem10_field_inspection")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "epoch": int(payload.get("epoch", -1)) if isinstance(payload, dict) else -1,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def checkpoint_for_policy(checkpoint_root: Path, checkpoint_name: str, policy: str) -> Path:
    """Resolve a paired checkpoint without changing the model/config pairing."""
    if policy == "last":
        checkpoint_name = checkpoint_name.replace("_best.pt", "_last.pt")
    return checkpoint_root / checkpoint_name


def load_model(spec: ModelSpec, cfg, checkpoint: Path, device: torch.device, channels: tuple[int, int, int]):
    architecture = OmegaConf.to_container(cfg.experiment.architecture, resolve=True)
    if not isinstance(architecture, dict):
        raise TypeError(f"{spec.slug}: experiment.architecture is not a mapping")
    if spec.slug == "transolverpp":
        valid = set(inspect.signature(TransolverPP.__mro__[1].__init__).parameters) - {"self"}
        architecture = {key: value for key, value in architecture.items() if key in valid}
    elif spec.slug == "mspt":
        valid = set(inspect.signature(MSPT.__init__).parameters) - {"self"}
        architecture = {key: value for key, value in architecture.items() if key in valid}
    elif spec.slug == "point_transformer_v3" and bool(architecture.get("enable_flash", False)):
        # The installed FlashAttention build targets an older TorchDynamo API;
        # SDPA is the production evaluator's equivalent inference backend.
        architecture["enable_flash"] = False
    model = CONSTRUCTORS[spec.slug](
        spatial_dim=3,
        surface_channels=channels[0],
        volume_channels=channels[1],
        parameter_channels=channels[2],
        **architecture,
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint_state(checkpoint), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{checkpoint.name} is incompatible with {spec.slug}/{cfg.experiment.model_name}: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return model.eval()


def sample_original(points: torch.Tensor, budget: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = len(points)
    if count >= budget:
        indices = torch.randperm(count, generator=generator)[:budget]
    else:
        indices = torch.randint(count, (budget,), generator=generator)
    return points[indices].contiguous()


def infer(model, model_slug: str, device: torch.device, geometry: torch.Tensor, surf_q: torch.Tensor, vol_q: torch.Tensor, params: torch.Tensor | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    geometry = geometry.unsqueeze(0).to(device, non_blocking=True)
    surf_q = surf_q.unsqueeze(0).to(device, non_blocking=True)
    vol_q = vol_q.unsqueeze(0).to(device, non_blocking=True)
    params = None if params is None else params.unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda" and model_slug != "lno",
    ):
        pred_s, pred_v = model.inference(geometry, surf_q, vol_q, params)
    pred_s = pred_s[0].float().cpu().numpy()
    pred_v = pred_v[0].float().cpu().numpy()
    if not np.isfinite(pred_s).all() or not np.isfinite(pred_v).all():
        raise FloatingPointError(f"{model_slug} produced a non-finite prediction")
    return pred_s, pred_v


def add_ground_truth(block: pv.PolyData, values: np.ndarray, field_names: list[str], groups: dict[str, tuple[int, ...]]) -> None:
    for channel, field in enumerate(field_names):
        block.point_data[f"gt_{safe_name(field)}"] = values[:, channel]
    for group, channels in groups.items():
        if max(channels) < values.shape[1]:
            block.point_data[f"gt_{safe_name(group)}"] = values[:, channels]


def add_prediction(block: pv.PolyData, prediction: np.ndarray, target: np.ndarray, field_names: list[str], groups: dict[str, tuple[int, ...]], variant: str, model_slug: str) -> None:
    prefix = f"{variant}_{model_slug}"
    error = np.abs(prediction - target)
    for channel, field in enumerate(field_names):
        name = safe_name(field)
        block.point_data[f"{prefix}_{name}"] = prediction[:, channel]
        block.point_data[f"abs_error_{prefix}_{name}"] = error[:, channel]
    block.point_data[f"abs_error_{prefix}_all_fields_l2"] = np.linalg.vector_norm(prediction - target, axis=1)
    for group, channels in groups.items():
        if max(channels) >= prediction.shape[1]:
            continue
        group_name = safe_name(group)
        block.point_data[f"{prefix}_{group_name}"] = prediction[:, channels]
        block.point_data[f"abs_error_{prefix}_{group_name}_magnitude"] = np.linalg.vector_norm(
            prediction[:, channels] - target[:, channels], axis=1
        )


def physical(values: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    return np.asarray(values * std.cpu().numpy()[None, :] + mean.cpu().numpy()[None, :], dtype=np.float32)


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(target.astype(np.float64))), 1.0e-12)
    return float(np.linalg.norm((prediction - target).astype(np.float64)) / denominator)


def source_block(condition: str, full_geometry: torch.Tensor, lower: np.ndarray, span: np.ndarray, qem_path: Path) -> pv.PolyData:
    if condition == "qem_div10":
        mesh = pv.read(qem_path)
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()
        mesh = mesh.copy()
        mesh.clear_data()
        return mesh
    points = full_geometry.cpu().numpy() * span[None, :] + lower[None, :]
    return pv.PolyData(np.ascontiguousarray(points, dtype=np.float32))


def validate_vtm(path: Path, expected_arrays: dict[str, set[str]]) -> dict[str, Any]:
    loaded = pv.read(path)
    if not isinstance(loaded, pv.MultiBlock):
        raise TypeError(f"{path} did not read back as vtkMultiBlockDataSet")
    report: dict[str, Any] = {"path": str(path.resolve()), "blocks": {}}
    for block_name, arrays in expected_arrays.items():
        block = loaded[block_name]
        if block is None:
            raise RuntimeError(f"{path}: missing block {block_name}")
        present = set(block.point_data.keys())
        missing = arrays - present
        if missing:
            raise RuntimeError(f"{path}/{block_name}: missing arrays {sorted(missing)}")
        for name in arrays:
            if not np.isfinite(np.asarray(block.point_data[name])).all():
                raise FloatingPointError(f"{path}/{block_name}/{name} contains non-finite values")
        report["blocks"][block_name] = {"points": int(block.n_points), "arrays": len(present)}
    return report


def main() -> None:
    args = parse_args()
    default_case, default_data, default_remesh = DEFAULTS[args.dataset]
    case_id = default_case if args.case_id is None else args.case_id
    data_root = default_data if args.data_root is None else args.data_root
    remesh_root = default_remesh if args.remesh_root is None else args.remesh_root
    selected_models = {item.strip() for item in args.models.split(",") if item.strip()}
    specs = [spec for spec in MODEL_SPECS[args.dataset] if not selected_models or spec.slug in selected_models]
    unknown = selected_models - {spec.slug for spec in MODEL_SPECS[args.dataset]}
    if unknown:
        raise ValueError(f"Unknown models for {args.dataset}: {sorted(unknown)}")
    if not specs:
        raise ValueError("No model specifications selected")

    availability: dict[str, dict[str, Any]] = {}
    usable_specs: list[ModelSpec] = []
    for spec in specs:
        paths = {
            "base": checkpoint_for_policy(args.checkpoint_root, spec.base_checkpoint, args.checkpoint_policy),
            "deal": checkpoint_for_policy(args.checkpoint_root, spec.deal_checkpoint, args.checkpoint_policy),
        }
        availability[spec.slug] = {variant: str(path.resolve()) if path.is_file() else None for variant, path in paths.items()}
        if all(path.is_file() for path in paths.values()):
            usable_specs.append(spec)
        elif not args.allow_missing:
            missing = [str(path) for path in paths.values() if not path.is_file()]
            raise FileNotFoundError(f"{spec.slug} is missing checkpoint(s): {missing}")
        else:
            print(f"[skip] {spec.slug}: incomplete base/DeAL pair", flush=True)
    if not usable_specs:
        raise RuntimeError("No complete base/DeAL checkpoint pairs are available")

    canonical_cfg = compose_config(usable_specs[0].deal_config)
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(canonical_cfg, resolve=True))
    dataset_cfg.experiment.data_path = str(data_root.resolve())
    dataset_cfg.experiment.num_body_points = 0
    dataset_cfg.experiment.model_name = "DEAL_FIELD_INSPECTION"
    _, dataset, _, spatial_dim, surface_channels, volume_channels, parameter_channels, fields = get_dataset(dataset_cfg.experiment)
    if spatial_dim != 3:
        raise ValueError("Only three-dimensional datasets are supported")
    dataset.set_epoch(0)
    try:
        case_index = dataset.data.index(case_id)
    except ValueError as exc:
        raise ValueError(f"case {case_id} is not in the evaluation split: {dataset.data[:8]}...") from exc

    full_geo, surf_q, surf_y, vol_q, vol_y, params, _ = unpack_item(dataset[case_index], parameter_channels)
    lower, span = bounds(dataset)
    surface_points = surf_q.numpy() * span[None, :] + lower[None, :]
    volume_points = vol_q.numpy() * span[None, :] + lower[None, :]
    target_s = physical(surf_y.numpy(), dataset.mean_surf_data.float(), dataset.std_surf_data.float())
    target_v = physical(vol_y.numpy(), dataset.mean_vol_data.float(), dataset.std_vol_data.float())
    surface_fields = list(fields["surface"])
    volume_fields = list(fields["volume"])
    surface_groups = VECTOR_GROUPS.get(args.dataset, {})
    volume_groups = {"velocity": (1, 2, 3)} if args.dataset in {"drivaerml", "pump"} else VECTOR_GROUPS.get(args.dataset, {})
    qem_path = remesh_path(args.dataset, remesh_root, case_id, "quadric", 10)
    if not qem_path.is_file():
        raise FileNotFoundError(qem_path)

    output_dir = args.output_dir / args.dataset / f"case_{case_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    channels = (surface_channels, volume_channels, parameter_channels)
    condition_blocks: dict[str, tuple[pv.PolyData, pv.PolyData]] = {}
    expected: dict[str, dict[str, set[str]]] = {}
    for condition in ("unshifted", "qem_div10"):
        surface = pv.PolyData(np.ascontiguousarray(surface_points, dtype=np.float32))
        volume = pv.PolyData(np.ascontiguousarray(volume_points, dtype=np.float32))
        add_ground_truth(surface, target_s, surface_fields, surface_groups)
        add_ground_truth(volume, target_v, volume_fields, volume_groups)
        condition_blocks[condition] = (surface, volume)
        expected[condition] = {
            "surface_queries": set(surface.point_data.keys()),
            "volume_queries": set(volume.point_data.keys()),
        }

    manifest: dict[str, Any] = {
        "dataset": args.dataset,
        "case_id": case_id,
        "data_root": str(data_root.resolve()),
        "remesh_root": str(remesh_root.resolve()),
        "checkpoint_policy": args.checkpoint_policy,
        "qem_div10_source": str(qem_path.resolve()),
        "query_points": {"surface": len(surf_q), "volume": len(vol_q)},
        "physical_fields": {"surface": surface_fields, "volume": volume_fields},
        "availability": availability,
        "models": {},
        "conditions": ["unshifted", "qem_div10"],
        "metrics": [],
    }

    geometry_cache: dict[tuple[str, int], torch.Tensor] = {}
    for spec_index, spec in enumerate(usable_specs):
        base_cfg, deal_cfg = compose_config(spec.base_config), compose_config(spec.deal_config)
        declared_base_budget, declared_deal_budget = input_budget(base_cfg), input_budget(deal_cfg)
        base_budget = spec.encoder_budget or declared_base_budget
        deal_budget = spec.encoder_budget or declared_deal_budget
        if base_budget != deal_budget:
            raise ValueError(f"{spec.slug}: base budget {base_budget} != DeAL budget {deal_budget}")
        budget = base_budget
        for condition_index, condition in enumerate(("unshifted", "qem_div10")):
            key = (condition, budget)
            if key not in geometry_cache:
                sample_seed = args.seed + case_id * 1009 + budget * 7 + condition_index * 104729
                if condition == "unshifted":
                    geometry_cache[key] = sample_original(full_geo, budget, sample_seed)
                else:
                    points = remesh_geometry(qem_path, budget, sample_seed, args.dataset)
                    geometry_cache[key] = torch.from_numpy((points - lower[None, :]) / span[None, :]).float()

        model_record: dict[str, Any] = {
            "encoder_budget": budget,
            "declared_config_budgets": {"base": declared_base_budget, "deal": declared_deal_budget},
            "variants": {},
        }
        shared_inference_seed = args.seed + case_id * 65537 + spec_index * 4099
        for variant, cfg, checkpoint_name in (
            ("base", base_cfg, spec.base_checkpoint),
            ("deal", deal_cfg, spec.deal_checkpoint),
        ):
            checkpoint = checkpoint_for_policy(args.checkpoint_root, checkpoint_name, args.checkpoint_policy)
            print(f"[{args.dataset}] {spec.slug}/{variant}: loading {checkpoint.name}", flush=True)
            model = load_model(spec, cfg, checkpoint, device, channels)
            model_record["variants"][variant] = {
                "config": spec.base_config if variant == "base" else spec.deal_config,
                "checkpoint": checkpoint_metadata(checkpoint),
            }
            for condition_index, condition in enumerate(("unshifted", "qem_div10")):
                pred_s_std, pred_v_std = infer(
                    model,
                    spec.slug,
                    device,
                    geometry_cache[(condition, budget)],
                    surf_q,
                    vol_q,
                    params,
                    shared_inference_seed + condition_index,
                )
                pred_s = physical(pred_s_std, dataset.mean_surf_data.float(), dataset.std_surf_data.float())
                pred_v = physical(pred_v_std, dataset.mean_vol_data.float(), dataset.std_vol_data.float())
                surface, volume = condition_blocks[condition]
                add_prediction(surface, pred_s, target_s, surface_fields, surface_groups, variant, spec.slug)
                add_prediction(volume, pred_v, target_v, volume_fields, volume_groups, variant, spec.slug)
                expected[condition]["surface_queries"].update(surface.point_data.keys())
                expected[condition]["volume_queries"].update(volume.point_data.keys())
                surf_error, vol_error = relative_l2(pred_s, target_s), relative_l2(pred_v, target_v)
                manifest["metrics"].append({
                    "model": spec.slug,
                    "variant": variant,
                    "condition": condition,
                    "surface_physical_rel_l2": surf_error,
                    "volume_physical_rel_l2": vol_error,
                    "combined_physical_rel_l2": 0.5 * (surf_error + vol_error),
                })
                print(
                    f"[{args.dataset}] {spec.slug}/{variant}/{condition}: "
                    f"surface={surf_error:.6f} volume={vol_error:.6f}",
                    flush=True,
                )
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        manifest["models"][spec.slug] = model_record

    validation = []
    for condition, (surface, volume) in condition_blocks.items():
        multiblock = pv.MultiBlock()
        multiblock["encoder_source"] = source_block(condition, full_geo, lower, span, qem_path)
        multiblock["surface_queries"] = surface
        multiblock["volume_queries"] = volume
        path = output_dir / f"{args.dataset}_case_{case_id}_{condition}_all_models_base_deal.vtm"
        multiblock.save(path, binary=True)
        validation.append(validate_vtm(path, expected[condition]))
        print(f"[saved] {path}", flush=True)
    manifest["vtk_validation"] = validation
    manifest_path = output_dir / "field_inspection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[saved] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
