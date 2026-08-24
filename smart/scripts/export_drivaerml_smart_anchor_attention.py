#!/usr/bin/env python3
"""Export SMART encoder-anchor attention concentration on a DrivAerML surface.

For every requested encoder-input condition, this diagnostic runs the six SMART
geometry cross-attention blocks for the base and SATLOSS7 checkpoints.  An
attention distribution is not itself a scalar anchor field: each latent anchor
has one probability distribution over geometry keys.  The exported score is
therefore the mean, across heads and encoder blocks, of one minus the
normalised attention entropy.  A value near one means a focused anchor-to-geo
attention distribution; a value near zero means diffuse attention.

The score is computed from the exact projected Q/K logits and RoPE used by the
model.  Log-sum-exp reduction is streamed over key chunks, avoiding materialising
the otherwise multi-gigabyte attention tensor.  Scores at the 4,096 learned
anchor locations are spread to the full preprocessed surface cloud using local
affine (linear) k-neighbour interpolation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2  # noqa: E402
from models.smart.smart import SMART  # noqa: E402
from train_consistency_common import sample_geometry_view  # noqa: E402


DEFAULT_SMART_CHECKPOINT = Path(
    "/home/parsa/smart_parsa/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt"
)
DEFAULT_SATLOSS_CHECKPOINT = Path(
    "/home/parsa/smart_parsa/checkpoints/smart-satloss7-smart-satloss7-drivaerml-131k-drivaerml-s42_best.pt"
)


def parse_csv(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def load_experiment_config(name: str, stack: tuple[str, ...] = ()):  # noqa: ANN201
    """Resolve this repository's local Hydra defaults without starting Hydra."""
    if name in stack:
        raise ValueError(f"Circular config defaults: {' -> '.join((*stack, name))}")
    path = SMART_ROOT / "config" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    root = OmegaConf.load(path)
    merged = OmegaConf.create()
    for default in root.get("defaults", []):
        if not isinstance(default, str) or default == "_self_" or default.startswith("override "):
            continue
        merged = OmegaConf.merge(merged, load_experiment_config(default.rsplit("/", 1)[-1], (*stack, name)))
    return OmegaConf.merge(merged, root.get("experiment", OmegaConf.create()))


def build_model(config_name: str, checkpoint_path: Path, device: torch.device) -> tuple[SMART, Any]:
    cfg = load_experiment_config(config_name)
    arch = cfg.architecture
    model = SMART(
        spatial_dim=3,
        surface_channels=7,
        volume_channels=4,
        parameter_channels=0,
        latent_dim=int(arch.latent_dim),
        latent_geometry_points=int(arch.latent_geometry_points),
        subsampled_geometry_points=int(arch.subsampled_geometry_points),
        subsampled_geometry_with_replacement=bool(arch.subsampled_geometry_with_replacement),
        num_encoder_decoder_blocks=int(arch.num_encoder_decoder_blocks),
        pos_scale_factor=float(arch.pos_scale_factor),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def sample_model_indices(count: int, budget: int, replacement: bool, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    if budget <= 0 or budget >= count and not replacement:
        return torch.arange(count, device=device, dtype=torch.long)
    if replacement:
        return torch.randint(count, (budget,), generator=generator, device=device, dtype=torch.long)
    return torch.randperm(count, generator=generator, device=device, dtype=torch.long)[:budget]


@torch.inference_mode()
def streamed_anchor_attention_concentration(
    attention_module,
    query_tokens: torch.Tensor,
    key_value_tokens: torch.Tensor,
    query_positions: torch.Tensor,
    key_value_positions: torch.Tensor,
    key_chunk_size: int,
) -> torch.Tensor:
    """Return 1-normalised entropy for every query anchor without a dense map."""
    batch, query_count, _ = query_tokens.shape
    key_count = int(key_value_tokens.shape[1])
    if key_count <= 1:
        return query_tokens.new_ones(batch, query_count, dtype=torch.float32)

    q = attention_module.q(attention_module.norm_q(query_tokens))
    q = q.view(batch, query_count, attention_module.num_heads, attention_module.head_dim).permute(0, 2, 1, 3)
    q = attention_module.rope(q, query_positions).float()
    running_max = None
    running_exp = None
    running_weighted_logits = None

    for start in range(0, key_count, int(key_chunk_size)):
        end = min(start + int(key_chunk_size), key_count)
        kv = attention_module.kv(attention_module.norm_kv(key_value_tokens[:, start:end]))
        kv = kv.view(batch, end - start, 2 * attention_module.num_heads, attention_module.head_dim).permute(0, 2, 1, 3)
        key = attention_module.rope(kv[:, : attention_module.num_heads], key_value_positions[:, start:end]).float()
        logits = torch.matmul(q, key.transpose(-1, -2)) / math.sqrt(float(attention_module.head_dim))
        block_max = logits.amax(dim=-1)
        if running_max is None:
            running_max = block_max
            shifted = torch.exp(logits - running_max.unsqueeze(-1))
            running_exp = shifted.sum(dim=-1)
            running_weighted_logits = (shifted * logits).sum(dim=-1)
            continue

        updated_max = torch.maximum(running_max, block_max)
        old_scale = torch.exp(running_max - updated_max)
        shifted = torch.exp(logits - updated_max.unsqueeze(-1))
        running_exp = running_exp * old_scale + shifted.sum(dim=-1)
        running_weighted_logits = running_weighted_logits * old_scale + (shifted * logits).sum(dim=-1)
        running_max = updated_max

    assert running_max is not None and running_exp is not None and running_weighted_logits is not None
    log_partition = running_max + torch.log(running_exp.clamp_min(torch.finfo(running_exp.dtype).tiny))
    entropy = log_partition - running_weighted_logits / running_exp.clamp_min(torch.finfo(running_exp.dtype).tiny)
    normalised_entropy = entropy / math.log(float(key_count))
    return (1.0 - normalised_entropy).clamp(0.0, 1.0).mean(dim=1)


@torch.inference_mode()
def extract_anchor_scores(
    model: SMART,
    geometry: torch.Tensor,
    seed: int,
    key_chunk_size: int,
    encoder_layer_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate SMART's encoder and record its anchor-to-geometry attention."""
    device = next(model.parameters()).device
    geo = geometry.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    geo_scaled = geo * float(model.pos_scale_factor)
    latent_idx = sample_model_indices(
        int(geo_scaled.shape[1]), int(model.num_geo), False, generator, device
    ).unsqueeze(0)
    latent_pos = torch.gather(geo_scaled, 1, latent_idx.unsqueeze(-1).expand(-1, -1, 3))
    latent = model.pos_encoder(latent_pos)
    layer_scores = []
    max_layers = len(model.encoder_blocks) if int(encoder_layer_count) <= 0 else min(int(encoder_layer_count), len(model.encoder_blocks))
    for block in model.encoder_blocks[:max_layers]:
        sub_idx = sample_model_indices(
            int(geo_scaled.shape[1]),
            int(model.subsampled_geometry_points),
            bool(model.subsampled_geometry_with_replacement),
            generator,
            device,
        ).unsqueeze(0)
        sub_pos = torch.gather(geo_scaled, 1, sub_idx.unsqueeze(-1).expand(-1, -1, 3))
        sub_tokens = model.pos_encoder(sub_pos)
        layer_scores.append(
            streamed_anchor_attention_concentration(
                block.geo_attn,
                latent,
                sub_tokens,
                latent_pos,
                sub_pos,
                key_chunk_size,
            )
        )
        latent, _cross_attended = block(
            latent,
            sub_tokens,
            None,
            latent_geometry_pos=latent_pos,
            subsampled_geometry_pos=sub_pos,
        )
    score = torch.stack(layer_scores, dim=0).mean(dim=0)[0]
    per_layer = torch.stack(layer_scores, dim=0)[:, 0]
    return (
        (latent_pos[0] / float(model.pos_scale_factor)).detach().cpu().numpy().astype(np.float32),
        score.detach().cpu().numpy().astype(np.float32),
        per_layer.detach().cpu().numpy().astype(np.float32),
    )


def locally_affine_interpolate(
    anchors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    neighbors: int,
    chunk_size: int,
    workers: int,
) -> np.ndarray:
    """Locally affine, distance-weighted interpolation from anchor scores to a surface cloud."""
    anchor_xyz = np.ascontiguousarray(anchors, dtype=np.float64)
    target_xyz = np.ascontiguousarray(targets, dtype=np.float64)
    source_values = np.asarray(values, dtype=np.float64).reshape(-1)
    if anchor_xyz.shape[0] != source_values.shape[0]:
        raise ValueError("Anchor coordinates and anchor attention scores have inconsistent lengths.")
    k = min(max(4, int(neighbors)), anchor_xyz.shape[0])
    tree = cKDTree(anchor_xyz)
    result = np.empty(target_xyz.shape[0], dtype=np.float32)
    identity = np.eye(4, dtype=np.float64)
    identity[0, 0] = 1.0e-10
    value_low, value_high = float(source_values.min()), float(source_values.max())
    for start in tqdm(range(0, target_xyz.shape[0], int(chunk_size)), desc="Linear anchor interpolation", leave=False):
        end = min(start + int(chunk_size), target_xyz.shape[0])
        points = target_xyz[start:end]
        distance, indices = tree.query(points, k=k, workers=int(workers))
        if k == 1:
            distance, indices = distance[:, None], indices[:, None]
        selected_xyz = anchor_xyz[indices]
        selected_values = source_values[indices]
        local_scale = np.maximum(distance[:, -1], 1.0e-8)
        offsets = (selected_xyz - points[:, None, :]) / local_scale[:, None, None]
        design = np.concatenate([np.ones((points.shape[0], k, 1), dtype=np.float64), offsets], axis=-1)
        weights = 1.0 / np.maximum(distance, 1.0e-8) ** 2
        gram = np.einsum("nki,nk,nkj->nij", design, weights, design, optimize=True)
        rhs = np.einsum("nki,nk,nk->ni", design, weights, selected_values, optimize=True)
        solution = np.linalg.solve(gram + 1.0e-6 * identity[None], rhs[..., None])[..., 0]
        interpolated = solution[:, 0]
        exact = distance[:, 0] <= 1.0e-10
        interpolated[exact] = selected_values[exact, 0]
        # Concentration is a bounded function of an attention distribution.
        result[start:end] = np.clip(interpolated, value_low, value_high).astype(np.float32)
    return result


def read_vtp_points(path: Path) -> np.ndarray:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetPoints() is None or poly.GetNumberOfPoints() <= 0:
        raise RuntimeError(f"Invalid VTP geometry source: {path}")
    points = np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RuntimeError(f"Invalid VTP point coordinates: {path}")
    return np.ascontiguousarray(points)


def write_point_vtp(path: Path, points: np.ndarray, fields: dict[str, np.ndarray]) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.ascontiguousarray(points, dtype=np.float32)
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(xyz, deep=True))
    poly.SetPoints(vtk_points)
    count = int(xyz.shape[0])
    offsets = np.arange(count + 1, dtype=np.int64)
    connectivity = np.arange(count, dtype=np.int64)
    vertices = vtk.vtkCellArray()
    vertices.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(connectivity, deep=True))
    poly.SetVerts(vertices)
    for name, value in fields.items():
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(count, -1))
        vtk_array = numpy_to_vtk(array, deep=True)
        vtk_array.SetName(str(name))
        poly.GetPointData().AddArray(vtk_array)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    # Use inline ASCII rather than appended binary/base64.  These full-surface
    # files are larger, but this is the most portable VTP encoding for the
    # cluster's ParaView builds and avoids XML parser failures in AppendedData.
    writer.SetDataModeToAscii()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTP: {path}")


def sample_condition(
    name: str,
    full_geometry: torch.Tensor,
    full_density: torch.Tensor,
    input_budget: int,
    seed: int,
    iso_points_normalized: torch.Tensor | None,
) -> torch.Tensor:
    if name == "aligned":
        view, _density, _mode = sample_geometry_view(
            full_geometry.unsqueeze(0), None, input_budget, "uniform_wor", 0.0, 1.0, seed
        )
        return view[0].contiguous()
    if name == "beta1":
        view, _density, _mode = sample_geometry_view(
            full_geometry.unsqueeze(0), full_density.unsqueeze(0), input_budget, "inverse_density_wor", 1.0, 1.0, seed
        )
        return view[0].contiguous()
    if name in {"sine_x1", "sine_y1"}:
        axis = 0 if name == "sine_x1" else 1
        view, _density, _mode = sample_geometry_view(
            full_geometry.unsqueeze(0),
            full_density.unsqueeze(0),
            input_budget,
            "sinusoidal_axis_mixture_wor",
            0.0,
            1.0,
            seed,
            sinusoidal_axis=axis,
            sinusoidal_mix_fraction=1.0,
        )
        return view[0].contiguous()
    if name == "isotropic_div10":
        if iso_points_normalized is None:
            raise ValueError("isotropic_div10 was requested but no isotropic VTP source was loaded.")
        if int(iso_points_normalized.shape[0]) < input_budget:
            raise ValueError("The isotropic div10 VTP has fewer points than the required encoder input budget.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        idx = torch.randperm(int(iso_points_normalized.shape[0]), generator=generator)[:input_budget]
        return iso_points_normalized.index_select(0, idx).contiguous()
    raise ValueError(f"Unknown condition: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssdraid/parsa/drivaerml_preprocessed"))
    parser.add_argument("--run-id", type=int, default=34)
    parser.add_argument("--smart-config", default="drivaerml")
    parser.add_argument("--satloss7-config", default="drivaerml_satloss7")
    parser.add_argument("--smart-checkpoint", type=Path, default=DEFAULT_SMART_CHECKPOINT)
    parser.add_argument("--satloss7-checkpoint", type=Path, default=DEFAULT_SATLOSS_CHECKPOINT)
    parser.add_argument("--smart-device", default="cuda:0")
    parser.add_argument("--satloss7-device", default="cuda:1")
    parser.add_argument("--input-points", type=int, default=131072)
    parser.add_argument("--density-knn-k", type=int, default=16)
    parser.add_argument("--density-estimator", default="kde", choices=("kde", "rk2", "tangent_cov"))
    parser.add_argument("--isotropic-decimated-vtp-dir", type=Path, default=Path("/mnt/ssdraid/parsa/drivaerml_surface_vtp_isotropic_gpu"))
    parser.add_argument("--conditions", default="aligned,isotropic_div10,beta1,sine_x1,sine_y1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-key-chunk-size", type=int, default=512)
    parser.add_argument("--encoder-layer-count", type=int, default=0, help="0 uses all encoder blocks; positive values are smoke-test only.")
    parser.add_argument("--interpolation-neighbors", type=int, default=12)
    parser.add_argument("--interpolation-chunk-size", type=int, default=65536)
    parser.add_argument("--interpolation-workers", type=int, default=8)
    parser.add_argument("--output-point-limit", type=int, default=0, help="0 exports the complete preprocessed surface cloud.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = parse_csv(args.conditions)
    valid_conditions = {"aligned", "isotropic_div10", "beta1", "sine_x1", "sine_y1"}
    invalid = sorted(set(conditions) - valid_conditions)
    if invalid:
        raise ValueError(f"Unknown --conditions: {invalid}; valid={sorted(valid_conditions)}")
    if not conditions:
        raise ValueError("At least one condition is required.")
    if args.input_points <= 0 or args.attention_key_chunk_size <= 0 or args.interpolation_chunk_size <= 0:
        raise ValueError("Input and chunk sizes must be positive.")
    if args.interpolation_workers == 0:
        raise ValueError("--interpolation-workers cannot be zero.")
    for checkpoint in (args.smart_checkpoint, args.satloss7_checkpoint):
        if not checkpoint.expanduser().is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    smart_device = torch.device(args.smart_device)
    satloss_device = torch.device(args.satloss7_device)
    if smart_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were requested but CUDA is unavailable.")
    smart_model, smart_cfg = build_model(args.smart_config, args.smart_checkpoint.expanduser().resolve(), smart_device)
    satloss_model, satloss_cfg = build_model(args.satloss7_config, args.satloss7_checkpoint.expanduser().resolve(), satloss_device)
    if smart_model.num_geo != satloss_model.num_geo or smart_model.subsampled_geometry_points != satloss_model.subsampled_geometry_points:
        raise RuntimeError("SMART and SATLOSS7 have incompatible anchor or encoder subsampling budgets.")
    if int(args.input_points) != 131072:
        raise ValueError("This export is fixed to the 131072-point DrivAerML training encoder budget.")

    dataset = AhmedMLDatasetV2(
        saved_folder=str(args.data_root.expanduser().resolve()),
        if_test=True,
        geometry_points=0,
        surface_points=1,
        volume_points=1,
        require_preprocessed=True,
        return_geometry_density=True,
        geometry_density_knn_k=int(args.density_knn_k),
        geometry_density_estimator=str(args.density_estimator),
        geometry_density_cache_dtype="float16",
        geometry_epoch_seeded_sampling=False,
    )
    if int(args.run_id) not in set(dataset.all_ids):
        raise ValueError(f"run_{args.run_id} is absent from {args.data_root}.")
    run_dir = args.data_root.expanduser().resolve() / f"run_{int(args.run_id)}"
    full_surface_raw = np.array(np.load(run_dir / "surface_coords.npy", mmap_mode="r"), dtype=np.float32, copy=True)
    if full_surface_raw.ndim != 2 or full_surface_raw.shape[1] != 3 or not np.isfinite(full_surface_raw).all():
        raise RuntimeError(f"run_{args.run_id} has invalid full surface coordinates.")
    full_geometry = (torch.from_numpy(full_surface_raw) - dataset.min_pos) / torch.clamp(dataset.max_pos - dataset.min_pos, min=1.0e-12)
    full_density = dataset._load_or_compute_full_geometry_density(int(args.run_id), expected_n=int(full_geometry.shape[0])).float()
    if int(full_density.shape[0]) != int(full_geometry.shape[0]):
        raise RuntimeError("Full KDE density cache does not align with the full preprocessed surface cloud.")

    iso_normalized = None
    if "isotropic_div10" in conditions:
        iso_path = args.isotropic_decimated_vtp_dir.expanduser().resolve() / f"run_{int(args.run_id)}" / f"drivaer_{int(args.run_id)}_faces_div10.vtp"
        if not iso_path.is_file():
            raise FileNotFoundError(f"Missing isotropic div10 source: {iso_path}")
        iso_raw = read_vtp_points(iso_path)
        bbox_delta = float(np.max(np.abs(np.concatenate([iso_raw.min(0), iso_raw.max(0)]) - np.concatenate([full_surface_raw.min(0), full_surface_raw.max(0)]))))
        if bbox_delta > 1.0e-3:
            raise ValueError(f"Isotropic VTP and preprocessed surface bounding boxes differ by {bbox_delta:.6g}; refusing to map mismatched coordinates.")
        iso_normalized = (torch.from_numpy(iso_raw) - dataset.min_pos) / torch.clamp(dataset.max_pos - dataset.min_pos, min=1.0e-12)
        print(f"isotropic_div10: {iso_raw.shape[0]} source vertices, bbox delta={bbox_delta:.3g}")

    output_raw = full_surface_raw
    if args.output_point_limit > 0 and int(args.output_point_limit) < output_raw.shape[0]:
        rng = np.random.default_rng(args.seed)
        output_raw = output_raw[np.sort(rng.choice(output_raw.shape[0], size=int(args.output_point_limit), replace=False))]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    min_pos, span = dataset.min_pos.numpy(), (dataset.max_pos - dataset.min_pos).numpy()
    summary: dict[str, Any] = {
        "run_id": int(args.run_id),
        "input_points": int(args.input_points),
        "anchor_points": int(smart_model.num_geo),
        "encoder_key_points_per_layer": int(smart_model.subsampled_geometry_points),
        "encoder_layers": len(smart_model.encoder_blocks) if args.encoder_layer_count <= 0 else int(args.encoder_layer_count),
        "attention_score": "mean_over_encoder_blocks_and_heads(1 - entropy(attention)/log(number_of_keys))",
        "interpolation": "local affine k-neighbour interpolation, clipped only to the observed anchor-score range",
        "output_surface_points": int(output_raw.shape[0]),
        "conditions": {},
    }
    print(f"Full output surface: {output_raw.shape[0]} points; anchors: {smart_model.num_geo}; conditions: {', '.join(conditions)}")
    for condition_index, condition in enumerate(conditions):
        condition_seed = int(args.seed + 100003 * int(args.run_id) + 1009 * condition_index)
        view = sample_condition(condition, full_geometry, full_density, int(args.input_points), condition_seed, iso_normalized)
        with ThreadPoolExecutor(max_workers=2) as executor:
            smart_future = executor.submit(extract_anchor_scores, smart_model, view, condition_seed, args.attention_key_chunk_size, args.encoder_layer_count)
            satloss_future = executor.submit(extract_anchor_scores, satloss_model, view, condition_seed, args.attention_key_chunk_size, args.encoder_layer_count)
            smart_anchor_norm, smart_score, smart_per_layer = smart_future.result()
            satloss_anchor_norm, satloss_score, satloss_per_layer = satloss_future.result()
        smart_anchor_raw = smart_anchor_norm * span + min_pos
        satloss_anchor_raw = satloss_anchor_norm * span + min_pos
        smart_full = locally_affine_interpolate(smart_anchor_raw, smart_score, output_raw, args.interpolation_neighbors, args.interpolation_chunk_size, args.interpolation_workers)
        satloss_full = locally_affine_interpolate(satloss_anchor_raw, satloss_score, output_raw, args.interpolation_neighbors, args.interpolation_chunk_size, args.interpolation_workers)
        if not np.isfinite(smart_full).all() or not np.isfinite(satloss_full).all():
            raise FloatingPointError(f"Non-finite interpolated attention field for {condition}.")
        stem = f"drivaerml_run_{int(args.run_id)}_{condition}_anchor_attention"
        write_point_vtp(
            output_dir / f"{stem}.vtp",
            output_raw,
            {"smart_attention": smart_full, "satloss7_attention": satloss_full},
        )
        np.savez_compressed(
            output_dir / f"{stem}_anchors.npz",
            smart_anchor_points=smart_anchor_raw.astype(np.float32),
            smart_attention=smart_score.astype(np.float32),
            smart_attention_per_layer=smart_per_layer.astype(np.float32),
            satloss7_anchor_points=satloss_anchor_raw.astype(np.float32),
            satloss7_attention=satloss_score.astype(np.float32),
            satloss7_attention_per_layer=satloss_per_layer.astype(np.float32),
        )
        summary["conditions"][condition] = {
            "seed": condition_seed,
            "output_vtp": f"{stem}.vtp",
            "anchors_npz": f"{stem}_anchors.npz",
            "smart_anchor_score_range": [float(smart_score.min()), float(smart_score.max())],
            "satloss7_anchor_score_range": [float(satloss_score.min()), float(satloss_score.max())],
            "smart_interpolated_score_range": [float(smart_full.min()), float(smart_full.max())],
            "satloss7_interpolated_score_range": [float(satloss_full.min()), float(satloss_full.max())],
        }
        print(f"Exported {condition}: {output_dir / f'{stem}.vtp'}", flush=True)
    (output_dir / "attention_export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {output_dir / 'attention_export_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
