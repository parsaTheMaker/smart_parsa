#!/usr/bin/env python3
"""Export pressure-conditioned SMART attention-rollout maps for DrivAerML.

This is deliberately different from anchor-attention entropy.  For each
decoder/encoder block pair, it routes a surface-pressure relevance weight from
surface query -> latent anchor -> encoder key point:

    R_l(i) = sum_q w(q) sum_a A_decoder,l(q, a) A_encoder,l(a, i)

where w(q) is the predicted pressure-contrast contribution at query q.  The
result is a query-conditioned routing diagnostic, not a claim that attention
alone is causal feature importance.  It is nevertheless much more meaningful
than attention entropy because it is tied to the predicted pressure field and
is mapped from actual encoder keys rather than randomly selected anchors.
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
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_drivaerml_smart_anchor_attention import (  # noqa: E402
    DEFAULT_SATLOSS_CHECKPOINT,
    DEFAULT_SMART_CHECKPOINT,
    build_model,
    parse_csv,
    read_vtp_points,
    sample_condition,
    sample_model_indices,
    write_point_vtp,
)

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2  # noqa: E402
def _attention_components(attention, query, key_value, query_pos, key_value_pos):
    """Return exact eval-mode cross-attention output and mean-head weights."""
    batch, query_count, _ = query.shape
    key_count = key_value.shape[1]
    q = attention.q(attention.norm_q(query))
    kv = attention.kv(attention.norm_kv(key_value))
    q = q.view(batch, query_count, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
    kv = kv.view(batch, key_count, 2 * attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
    key, value = kv[:, : attention.num_heads], kv[:, attention.num_heads :]
    q = attention.rope(q, query_pos)
    key = attention.rope(key, key_value_pos)
    logits = torch.matmul(q, key.transpose(-1, -2)) / math.sqrt(float(attention.head_dim))
    weights = torch.softmax(logits.float(), dim=-1).to(dtype=query.dtype)
    output = torch.matmul(weights, value)
    output = output.permute(0, 2, 1, 3).reshape(batch, query_count, -1)
    return attention.out_proj(output), weights.mean(dim=1)


@torch.inference_mode()
def _encode_with_seed(model, geometry: torch.Tensor, seed: int):
    """Replicate SMART.encode while fixing all sampled point indices."""
    device = next(model.parameters()).device
    geo = geometry.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    geo = geo * float(model.pos_scale_factor)
    latent_idx = sample_model_indices(geo.shape[1], model.num_geo, False, generator, device).unsqueeze(0)
    latent_pos = torch.gather(geo, 1, latent_idx.unsqueeze(-1).expand(-1, -1, 3))
    latent = model.pos_encoder(latent_pos)
    intermediate = []
    for block in model.encoder_blocks:
        sub_idx = sample_model_indices(
            geo.shape[1], model.subsampled_geometry_points, model.subsampled_geometry_with_replacement, generator, device
        ).unsqueeze(0)
        sub_pos = torch.gather(geo, 1, sub_idx.unsqueeze(-1).expand(-1, -1, 3))
        sub_tokens = model.pos_encoder(sub_pos)
        latent, cross_attended = block(
            latent, sub_tokens, None, latent_geometry_pos=latent_pos, subsampled_geometry_pos=sub_pos
        )
        intermediate.append(cross_attended)
    return intermediate, latent_pos


@torch.inference_mode()
def _decode_surface_pressure(model, intermediate, latent_pos, queries: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Decode pressure in bounded chunks using the model's native decoder."""
    device = next(model.parameters()).device
    values = []
    for start in range(0, queries.shape[0], int(chunk_size)):
        query_pos = queries[start : start + int(chunk_size)].unsqueeze(0).to(device=device, dtype=torch.float32)
        query_tokens = model.pos_encoder(query_pos * float(model.pos_scale_factor))
        for encoded, block in zip(intermediate, model.decoder_blocks):
            query_tokens = block(
                query_tokens,
                encoded,
                None,
                queries_pos=query_pos * float(model.pos_scale_factor),
                latent_geometry_pos=latent_pos,
            )
        values.append(model.mlp(query_tokens)[0, :, 0].float().cpu())
    return torch.cat(values, dim=0)


@torch.inference_mode()
def _verify_manual_attention(model, intermediate, latent_pos, queries: torch.Tensor) -> None:
    """Fail rather than export if manual attention differs from SMART's attention."""
    device = next(model.parameters()).device
    query_pos = queries[: min(64, queries.shape[0])].unsqueeze(0).to(device=device, dtype=torch.float32)
    scaled = query_pos * float(model.pos_scale_factor)
    tokens = model.pos_encoder(scaled)
    for encoded, block in zip(intermediate, model.decoder_blocks):
        expected = block.attn(tokens, encoded, q_pos=scaled, kv_pos=latent_pos)
        actual, _weights = _attention_components(block.attn, tokens, encoded, scaled, latent_pos)
        error = float((actual - expected).abs().max().item())
        if error > 3.0e-5:
            raise RuntimeError(f"Manual attention mismatch ({error:.3g}); refusing to export attribution.")
        tokens = block(tokens, encoded, None, queries_pos=scaled, latent_geometry_pos=latent_pos)


@torch.inference_mode()
def _decoder_anchor_relevance(model, intermediate, latent_pos, queries: torch.Tensor, weights: torch.Tensor, chunk_size: int):
    """Accumulate pressure-weighted decoder attention for every latent anchor."""
    device = next(model.parameters()).device
    masses = [torch.zeros(model.num_geo, device=device, dtype=torch.float32) for _ in model.decoder_blocks]
    total_weight = weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    for start in tqdm(range(0, queries.shape[0], int(chunk_size)), desc="Pressure-conditioned decoder rollout", leave=False):
        end = min(start + int(chunk_size), queries.shape[0])
        query_pos = queries[start:end].unsqueeze(0).to(device=device, dtype=torch.float32)
        scaled = query_pos * float(model.pos_scale_factor)
        tokens = model.pos_encoder(scaled)
        query_weight = weights[start:end].to(device=device, dtype=torch.float32)
        for layer, (encoded, block) in enumerate(zip(intermediate, model.decoder_blocks)):
            attended, attention_mean = _attention_components(block.attn, tokens, encoded, scaled, latent_pos)
            masses[layer] += torch.matmul(query_weight.unsqueeze(0), attention_mean[0]).squeeze(0)
            tokens = tokens + block.residual_update_scale * block.attn_dropout(attended)
            tokens = tokens + block.residual_update_scale * block.mlp(tokens, None)
            tokens = block.output_norm(tokens)
    return [mass / total_weight for mass in masses]


@torch.inference_mode()
def _encoder_key_relevance(
    model,
    geometry: torch.Tensor,
    seed: int,
    anchor_masses: list[torch.Tensor],
    key_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Route decoder relevance through each encoder attention matrix to keys."""
    device = next(model.parameters()).device
    geo = geometry.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    geo = geo * float(model.pos_scale_factor)
    latent_idx = sample_model_indices(geo.shape[1], model.num_geo, False, generator, device).unsqueeze(0)
    latent_pos = torch.gather(geo, 1, latent_idx.unsqueeze(-1).expand(-1, -1, 3))
    latent = model.pos_encoder(latent_pos)
    all_positions, all_scores, layer_mass_checks = [], [], []
    for layer, block in enumerate(model.encoder_blocks):
        sub_idx = sample_model_indices(
            geo.shape[1], model.subsampled_geometry_points, model.subsampled_geometry_with_replacement, generator, device
        ).unsqueeze(0)
        sub_pos = torch.gather(geo, 1, sub_idx.unsqueeze(-1).expand(-1, -1, 3))
        sub_tokens = model.pos_encoder(sub_pos)
        attention = block.geo_attn
        q = attention.q(attention.norm_q(latent))
        q = q.view(1, model.num_geo, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
        q = attention.rope(q, latent_pos).float()
        running_max = None
        running_exp = None
        for start in range(0, sub_tokens.shape[1], int(key_chunk_size)):
            end = min(start + int(key_chunk_size), sub_tokens.shape[1])
            kv = attention.kv(attention.norm_kv(sub_tokens[:, start:end]))
            kv = kv.view(1, end - start, 2 * attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
            key = attention.rope(kv[:, : attention.num_heads], sub_pos[:, start:end]).float()
            logits = torch.matmul(q, key.transpose(-1, -2)) / math.sqrt(float(attention.head_dim))
            block_max = logits.amax(dim=-1)
            if running_max is None:
                running_max = block_max
                running_exp = torch.exp(logits - block_max.unsqueeze(-1)).sum(dim=-1)
            else:
                merged_max = torch.maximum(running_max, block_max)
                running_exp = running_exp * torch.exp(running_max - merged_max) + torch.exp(logits - merged_max.unsqueeze(-1)).sum(dim=-1)
                running_max = merged_max
        scores = []
        anchor_mass = anchor_masses[layer].to(device=device, dtype=torch.float32)
        for start in range(0, sub_tokens.shape[1], int(key_chunk_size)):
            end = min(start + int(key_chunk_size), sub_tokens.shape[1])
            kv = attention.kv(attention.norm_kv(sub_tokens[:, start:end]))
            kv = kv.view(1, end - start, 2 * attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
            key = attention.rope(kv[:, : attention.num_heads], sub_pos[:, start:end]).float()
            logits = torch.matmul(q, key.transpose(-1, -2)) / math.sqrt(float(attention.head_dim))
            probs = torch.exp(logits - running_max.unsqueeze(-1)) / running_exp.unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
            scores.append(torch.matmul(anchor_mass.unsqueeze(0), probs.mean(dim=1)[0]).squeeze(0).cpu())
        score = torch.cat(scores, dim=0)
        all_positions.append((sub_pos[0] / float(model.pos_scale_factor)).cpu())
        all_scores.append(score)
        layer_mass_checks.append(float(score.sum().item()))
        latent, _cross_attended = block(latent, sub_tokens, None, latent_geometry_pos=latent_pos, subsampled_geometry_pos=sub_pos)
    return (
        torch.cat(all_positions, dim=0).numpy().astype(np.float32),
        torch.cat(all_scores, dim=0).numpy().astype(np.float32),
        layer_mass_checks,
    )


def _positive_idw_interpolate(points: np.ndarray, values: np.ndarray, targets: np.ndarray, neighbors: int, chunk_size: int, workers: int) -> np.ndarray:
    """Nonnegative inverse-distance interpolation for a nonnegative relevance field."""
    source = np.ascontiguousarray(points, dtype=np.float64)
    target = np.ascontiguousarray(targets, dtype=np.float64)
    value = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    if source.shape[0] != value.shape[0] or np.any(value < 0.0) or not np.isfinite(value).all():
        raise ValueError("Invalid nonnegative key relevance support.")
    k = min(max(4, int(neighbors)), source.shape[0])
    tree = cKDTree(source)
    output = np.empty(target.shape[0], dtype=np.float32)
    for start in tqdm(range(0, target.shape[0], int(chunk_size)), desc="Key relevance interpolation", leave=False):
        end = min(start + int(chunk_size), target.shape[0])
        distance, indices = tree.query(target[start:end], k=k, workers=int(workers))
        if k == 1:
            distance, indices = distance[:, None], indices[:, None]
        exact = distance[:, 0] <= 1.0e-12
        weight = 1.0 / np.maximum(distance, 1.0e-12) ** 2
        interpolated = (weight * value[indices]).sum(axis=1) / weight.sum(axis=1)
        interpolated[exact] = value[indices[exact, 0]]
        output[start:end] = interpolated.astype(np.float32)
    return output


def _joint_display_normalization(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    values = np.concatenate([first, second]).astype(np.float64, copy=False)
    low, high = np.quantile(values, [0.01, 0.995])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = low + 1.0e-12
    return (
        np.clip((first - low) / (high - low), 0.0, 1.0).astype(np.float32),
        np.clip((second - low) / (high - low), 0.0, 1.0).astype(np.float32),
        (float(low), float(high)),
    )


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument("--semantic-query-points", type=int, default=65536)
    parser.add_argument("--decoder-query-chunk-size", type=int, default=512)
    parser.add_argument("--encoder-key-chunk-size", type=int, default=512)
    parser.add_argument("--density-knn-k", type=int, default=16)
    parser.add_argument("--density-estimator", default="kde", choices=("kde", "rk2", "tangent_cov"))
    parser.add_argument("--isotropic-decimated-vtp-dir", type=Path, default=Path("/mnt/ssdraid/parsa/drivaerml_surface_vtp_isotropic_gpu"))
    parser.add_argument("--conditions", default="aligned,isotropic_div10,beta1,sine_x1,sine_y1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interpolation-neighbors", type=int, default=16)
    parser.add_argument("--interpolation-chunk-size", type=int, default=65536)
    parser.add_argument("--interpolation-workers", type=int, default=8)
    parser.add_argument("--output-point-limit", type=int, default=0, help="0 exports the full preprocessed surface cloud.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _model_semantic_relevance(model, view, query_positions, seed, decoder_chunk, key_chunk):
    intermediate, latent_pos = _encode_with_seed(model, view, seed)
    _verify_manual_attention(model, intermediate, latent_pos, query_positions)
    pressure = _decode_surface_pressure(model, intermediate, latent_pos, query_positions, decoder_chunk)
    pressure_weight = (pressure - pressure.mean()).square()
    pressure_weight = pressure_weight / pressure_weight.sum().clamp_min(torch.finfo(pressure_weight.dtype).tiny)
    anchor_mass = _decoder_anchor_relevance(model, intermediate, latent_pos, query_positions, pressure_weight, decoder_chunk)
    support, score, mass_check = _encoder_key_relevance(model, view, seed, anchor_mass, key_chunk)
    return support, score, pressure, mass_check


def main() -> None:
    args = _parse_args()
    conditions = parse_csv(args.conditions)
    valid = {"aligned", "isotropic_div10", "beta1", "sine_x1", "sine_y1"}
    if not conditions or set(conditions) - valid:
        raise ValueError(f"--conditions must be nonempty and chosen from {sorted(valid)}")
    if args.input_points != 131072:
        raise ValueError("DrivAerML semantic export is fixed to the 131072-point training encoder budget.")
    if min(args.semantic_query_points, args.decoder_query_chunk_size, args.encoder_key_chunk_size, args.interpolation_chunk_size) <= 0:
        raise ValueError("All point and chunk budgets must be positive.")
    for checkpoint in (args.smart_checkpoint, args.satloss7_checkpoint):
        if not checkpoint.expanduser().is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    smart, _smart_cfg = build_model(args.smart_config, args.smart_checkpoint.expanduser().resolve(), torch.device(args.smart_device))
    satloss, _satloss_cfg = build_model(args.satloss7_config, args.satloss7_checkpoint.expanduser().resolve(), torch.device(args.satloss7_device))
    if (smart.num_geo, smart.subsampled_geometry_points, len(smart.encoder_blocks)) != (
        satloss.num_geo,
        satloss.subsampled_geometry_points,
        len(satloss.encoder_blocks),
    ):
        raise RuntimeError("SMART and SATLOSS7 architecture budgets must match for paired attention rollout.")

    dataset = AhmedMLDatasetV2(
        saved_folder=str(args.data_root.expanduser().resolve()), if_test=True, geometry_points=0, surface_points=1, volume_points=1,
        require_preprocessed=True, return_geometry_density=True, geometry_density_knn_k=args.density_knn_k,
        geometry_density_estimator=args.density_estimator, geometry_density_cache_dtype="float16", geometry_epoch_seeded_sampling=False,
    )
    run_dir = args.data_root.expanduser().resolve() / f"run_{args.run_id}"
    raw_surface = np.array(np.load(run_dir / "surface_coords.npy", mmap_mode="r"), dtype=np.float32, copy=True)
    if raw_surface.ndim != 2 or raw_surface.shape[1] != 3 or not np.isfinite(raw_surface).all():
        raise RuntimeError(f"Invalid surface coordinates in {run_dir}")
    span_tensor = torch.clamp(dataset.max_pos - dataset.min_pos, min=1.0e-12)
    full_geometry = (torch.from_numpy(raw_surface) - dataset.min_pos) / span_tensor
    full_density = dataset._load_or_compute_full_geometry_density(args.run_id, expected_n=full_geometry.shape[0]).float()
    if full_density.shape[0] != full_geometry.shape[0]:
        raise RuntimeError("The surface density cache does not align with the full surface.")

    iso_normalized = None
    if "isotropic_div10" in conditions:
        iso_path = args.isotropic_decimated_vtp_dir.expanduser().resolve() / f"run_{args.run_id}" / f"drivaer_{args.run_id}_faces_div10.vtp"
        iso_raw = read_vtp_points(iso_path)
        bbox_delta = float(np.max(np.abs(np.r_[iso_raw.min(0), iso_raw.max(0)] - np.r_[raw_surface.min(0), raw_surface.max(0)])))
        if bbox_delta > 1.0e-3:
            raise ValueError(f"Isotropic VTP bbox mismatch ({bbox_delta:.6g}); refusing to map coordinates.")
        iso_normalized = (torch.from_numpy(iso_raw) - dataset.min_pos) / span_tensor

    query_generator = torch.Generator(device="cpu")
    query_generator.manual_seed(args.seed + 8093 * args.run_id)
    query_count = min(args.semantic_query_points, full_geometry.shape[0])
    query_idx = torch.randperm(full_geometry.shape[0], generator=query_generator)[:query_count]
    query_positions = full_geometry.index_select(0, query_idx).contiguous()
    output_raw = raw_surface
    if 0 < args.output_point_limit < raw_surface.shape[0]:
        output_raw = raw_surface[np.sort(np.random.default_rng(args.seed).choice(raw_surface.shape[0], args.output_point_limit, replace=False))]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "semantic_method": "predicted_pressure_variance_weighted_decoder_encoder_attention_rollout",
        "equation": "R_l(i)=sum_q w(q) sum_a A_decoder,l(q,a) A_encoder,l(a,i), w(q)=(p(q)-mean(p))^2/sum_q(...)",
        "query_count": int(query_count),
        "encoder_key_points_per_layer": int(smart.subsampled_geometry_points),
        "encoder_layers": len(smart.encoder_blocks),
        "interpolation": "nonnegative inverse-distance interpolation from actual encoder key locations",
        "display_normalization": "paired 1st to 99.5th percentile over SMART and SATLOSS7; raw fields are also saved",
        "conditions": {},
    }
    print(f"Semantic pressure rollout: run={args.run_id}, queries={query_count}, output_points={output_raw.shape[0]}", flush=True)
    for condition_index, condition in enumerate(conditions):
        seed = int(args.seed + 100003 * args.run_id + 1009 * condition_index)
        view = sample_condition(condition, full_geometry, full_density, args.input_points, seed, iso_normalized)
        with ThreadPoolExecutor(max_workers=2) as executor:
            smart_future = executor.submit(_model_semantic_relevance, smart, view, query_positions, seed, args.decoder_query_chunk_size, args.encoder_key_chunk_size)
            satloss_future = executor.submit(_model_semantic_relevance, satloss, view, query_positions, seed, args.decoder_query_chunk_size, args.encoder_key_chunk_size)
            smart_support, smart_key_score, smart_pressure, smart_mass_check = smart_future.result()
            satloss_support, satloss_key_score, satloss_pressure, satloss_mass_check = satloss_future.result()
        if any(abs(value - 1.0) > 2.0e-4 for value in (*smart_mass_check, *satloss_mass_check)):
            raise RuntimeError("Attention rollout mass is not conserved across an encoder layer.")
        smart_raw = _positive_idw_interpolate(smart_support, smart_key_score, output_raw, args.interpolation_neighbors, args.interpolation_chunk_size, args.interpolation_workers)
        satloss_raw = _positive_idw_interpolate(satloss_support, satloss_key_score, output_raw, args.interpolation_neighbors, args.interpolation_chunk_size, args.interpolation_workers)
        smart_display, satloss_display, display_range = _joint_display_normalization(smart_raw, satloss_raw)
        stem = f"drivaerml_run_{args.run_id}_{condition}_semantic_pressure_attention"
        write_point_vtp(
            output_dir / f"{stem}.vtp",
            output_raw,
            {
                "smart_attention": smart_display,
                "satloss7_attention": satloss_display,
                "smart_attention_raw": smart_raw,
                "satloss7_attention_raw": satloss_raw,
            },
        )
        np.savez_compressed(
            output_dir / f"{stem}_support.npz",
            smart_encoder_key_points=smart_support,
            smart_encoder_key_relevance=smart_key_score,
            satloss7_encoder_key_points=satloss_support,
            satloss7_encoder_key_relevance=satloss_key_score,
            query_positions=query_positions.numpy().astype(np.float32),
            smart_query_pressure=smart_pressure.numpy().astype(np.float32),
            satloss7_query_pressure=satloss_pressure.numpy().astype(np.float32),
        )
        summary["conditions"][condition] = {
            "seed": seed,
            "output_vtp": f"{stem}.vtp",
            "support_npz": f"{stem}_support.npz",
            "smart_raw_range": [float(smart_raw.min()), float(smart_raw.max())],
            "satloss7_raw_range": [float(satloss_raw.min()), float(satloss_raw.max())],
            "shared_display_percentile_range": list(display_range),
            "smart_layer_mass": smart_mass_check,
            "satloss7_layer_mass": satloss_mass_check,
        }
        print(f"Exported {condition}: {output_dir / f'{stem}.vtp'}", flush=True)
    (output_dir / "semantic_attention_export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
