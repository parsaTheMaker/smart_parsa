# MSPT Pipeline: Complete Embedded Backup

This document is a standalone source-level backup of the current MSPT
DrivAerML pipeline. It includes the effective vanilla and SATLOSS6
configurations, preprocessing contract, loader and normalization behavior,
targets, model implementation, training wrappers, losses, optimizers,
validation, checkpointing, AMP/CUDA prefetch, and distributed execution. The
source modules are embedded below in dependency order; this document is not an
index that requires opening another repository file.

## Scope

The vanilla path trains the model with one uniformly sampled geometry view and
the shared surface-plus-volume Relative-L2 objective. The SATLOSS6 path uses
the same model and targets but the shared two-view consistency trainer: each
batch samples two density-shifted geometry views from the full source cloud,
uses the same query coordinates and targets for both, and combines supervised
view losses with prediction consistency and the configured task-weighting
backend.

## Configuration Resolution

The vanilla and SATLOSS6 configuration inheritance is embedded explicitly:

~~~text
vanilla base:       embedded canonical base configuration
vanilla model:      embedded model-specific vanilla configuration
SATLOSS6 base:      embedded canonical SATLOSS6 base configuration
SATLOSS6 model:     embedded model-specific SATLOSS6 configuration
~~~

The sections below contain the exact source bodies used by these paths.

## Embedded Source Archive

### Vanilla base configuration


~~~yaml
defaults:
  - _self_
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

hydra:
  output_subdir: null
  run:
    dir: .

wandb:
  project: "smart"
  entity: "parsa-vatani99-technical-university-of-munich"

experiment:
  name: "DrivAerML_SMART"
  model_name: "SMART"
  model_tag: ""
  random_seed: 42
  manifest_variant: "full"

  architecture:
    latent_dim: 256
    latent_geometry_points: 4096
    subsampled_geometry_points: 16384
    subsampled_geometry_with_replacement: True
    num_encoder_decoder_blocks: 6
    pos_scale_factor: 100

  optimizer: "adamw"
  loss_fn: "rel_l2"
  scheduler: "cosine"
  scheduler_warumup_fraction: 0.2
  batch_size: 1
  epochs: 300
  learning_rate: 2.e-4
  gradient_norm: null
  amp: True
  precision: "float16"
  log_every_n_steps: 10
  inspect_first_sample: False

  dataset: "DrivAerML"
  data_path: "/mnt/ssdraid/parsa/drivaerml_preprocessed"
  num_workers: 8
  prefetch_factor: 4
  pin_memory: True
  geometry_epoch_seeded_sampling: True
  num_body_points: 131072
  num_surface_points: 65536
  num_volume_points: 65536
  scale_positions: False

~~~

### Model vanilla configuration


~~~yaml
defaults:
  - _self_
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

hydra:
  output_subdir: null
  run:
    dir: .

wandb:
  project: "smart_drivaerml"
  entity: "parsa-vatani99-technical-university-of-munich"

experiment:
  name: "DrivAerML_MSPT"
  model_name: "MSPT"
  model_tag: ""
  random_seed: 42
  manifest_variant: "full"

  architecture:
    # SMART-matched capacity and official MSPT defaults where applicable.
    num_blocks: 4
    n_hidden: 192
    num_heads: 4
    dropout: 0.1
    activation: "gelu"
    mlp_ratio: 1
    V: 128
    Q: 1
    attn_pool: "mean"
    chunking_mode: "balltree"
    use_rope: False
    rope_base: 10000.0
    use_flash_attn: False
    use_checkpoint: True

  optimizer: "adamw"
  loss_fn: "rel_l2"
  scheduler: "cosine"
  scheduler_warmup_fraction: 0.2
  batch_size: 1
  epochs: 300
  learning_rate: 2.e-4
  gradient_norm: null
  amp: True
  precision: "float16"
  log_every_n_steps: 10
  wandb_watch_model: False
  init_ckpt: ""
  resume_ckpt: ""
  resume_full_state: False

  dataset: "DrivAerML"
  data_path: "/mnt/ssdraid/parsa/drivaerml_preprocessed"
  num_workers: 8
  prefetch_factor: 4
  pin_memory: True
  cuda_batch_prefetch: True
  # Match SMART vanilla: uniform, epoch-varying geometry views.
  geometry_epoch_seeded_sampling: True
  num_body_points: 65536
  num_surface_points: 65536
  num_volume_points: 65536
  scale_positions: False

~~~

### SATLOSS6 base configuration


~~~yaml
defaults:
  - _self_
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

hydra:
  output_subdir: null
  run:
    dir: .

wandb:
  project: "smart"
  entity: "parsa-vatani99-technical-university-of-munich"

experiment:
  name: "DrivAerML_SMART_SATLOSS6"
  model_name: "SMART_SATLOSS6"
  model_tag: ""
  random_seed: 42
  manifest_variant: "full"

  architecture:
    latent_dim: 256
    latent_geometry_points: 4096
    subsampled_geometry_points: 16384
    subsampled_geometry_with_replacement: True
    num_encoder_decoder_blocks: 6
    pos_scale_factor: 100

  optimizer: "adamw"
  loss_fn: "rel_l2"
  scheduler: "cosine"
  scheduler_warmup_fraction: 0.2
  batch_size: 1
  epochs: 300
  learning_rate: 2.e-4
  gradient_norm: null
  amp: True
  precision: "float16"
  log_every_n_steps: 5
  inspect_first_sample: False
  init_ckpt: "/home/parsa/smart_parsa/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt"
  resume_ckpt: ""
  resume_full_state: False
  wandb_watch_model: False
  multi_gpu_strategy: "auto"

  dataset: "DrivAerML"
  data_path: "/mnt/ssdraid/parsa/drivaerml_preprocessed"
  num_workers: 4
  prefetch_factor: 2
  pin_memory: True
  cuda_batch_prefetch: True
  geometry_density_cache_dtype: "float16"
  density_knn_k: 16
  density_neighbor_hops: 1
  density_estimator: "kde"
  geometry_epoch_seeded_sampling: False
  num_body_points: 0
  num_surface_points: 65536
  num_volume_points: 65536
  scale_positions: False

  view_geometry_points: 131072
  eval_view_geometry_points: 131072
  train_primary_sampling_mode: "inverse_density_wor"
  train_secondary_sampling_mode: "inverse_density_wor"
  eval_aligned_sampling_mode: "uniform_wor"
  eval_shifted_sampling_mode: "inverse_density_wor"
  inverse_density_beta: 1.0
  randomize_primary_inverse_density_beta: True
  primary_inverse_density_beta_min: 0.0
  primary_inverse_density_beta_max: 0.5
  randomize_secondary_inverse_density_beta: True
  secondary_inverse_density_beta_min: 0.0
  secondary_inverse_density_beta_max: 0.5
  mixed_inverse_density_prob: 1.0
  use_prediction_consistency: True
  prediction_consistency_weight: 1.0
  prediction_consistency_smooth_l1_beta: 0.1
  use_latent_consistency: False
  consistency_warmup_epochs: 20
  fuse_consistency_views: True

  use_learned_task_weighting: True
  task_weight_lr: 5.e-4
  task_weight_init_logits: [0.0, 0.0, 0.0]
  task_weight_logit_min: -3.0
  task_weight_logit_max: 3.0
  task_weight_min_weights: [0.1, 0.1, 0.1]
  task_weight_base_weights: [0.15, 0.15, 0.70]
  task_weight_warmup_epochs: 20
  soft_worst_case_tau: 0.1

~~~

### Model SATLOSS6 configuration


~~~yaml
defaults:
  - _self_
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

hydra:
  output_subdir: null
  run:
    dir: .

wandb:
  project: "smart_drivaerml"
  entity: "parsa-vatani99-technical-university-of-munich"

experiment:
  name: "DrivAerML_MSPT_SATLOSS6"
  model_name: "MSPT_SATLOSS6"
  model_tag: ""
  random_seed: 42
  manifest_variant: "full"

  architecture:
    num_blocks: 4
    n_hidden: 192
    num_heads: 4
    dropout: 0.1
    activation: "gelu"
    mlp_ratio: 1
    V: 128
    Q: 1
    attn_pool: "mean"
    chunking_mode: "balltree"
    use_rope: False
    rope_base: 10000.0
    use_flash_attn: False
    use_checkpoint: True

  optimizer: "adamw"
  loss_fn: "rel_l2"
  scheduler: "cosine"
  scheduler_warmup_fraction: 0.2
  batch_size: 1
  epochs: 300
  learning_rate: 2.e-4
  gradient_norm: null
  amp: True
  precision: "float16"
  log_every_n_steps: 5
  inspect_first_sample: False
  init_ckpt: "/home/parsa/smart_parsa/checkpoints/mspt-mspt-drivaerml-smart-fair-gpu6-drivaerml-s42_best.pt"
  resume_ckpt: ""
  resume_full_state: False
  wandb_watch_model: False
  multi_gpu_strategy: "single"

  dataset: "DrivAerML"
  data_path: "/mnt/ssdraid/parsa/drivaerml_preprocessed"
  num_workers: 4
  prefetch_factor: 2
  pin_memory: True
  cuda_batch_prefetch: True
  geometry_density_cache_dtype: "float16"
  density_knn_k: 16
  density_neighbor_hops: 1
  density_estimator: "kde"
  geometry_epoch_seeded_sampling: False
  # Load the full cloud, then draw two real 65k density-shifted views.
  num_body_points: 0
  num_surface_points: 65536
  num_volume_points: 65536
  scale_positions: False

  view_geometry_points: 65536
  eval_view_geometry_points: 65536
  train_primary_sampling_mode: "inverse_density_wor"
  train_secondary_sampling_mode: "inverse_density_wor"
  eval_aligned_sampling_mode: "uniform_wor"
  eval_shifted_sampling_mode: "inverse_density_wor"
  inverse_density_beta: 1.0
  randomize_primary_inverse_density_beta: True
  primary_inverse_density_beta_min: 0.0
  primary_inverse_density_beta_max: 0.5
  randomize_secondary_inverse_density_beta: True
  secondary_inverse_density_beta_min: 0.0
  secondary_inverse_density_beta_max: 0.5
  mixed_inverse_density_prob: 1.0
  use_prediction_consistency: True
  prediction_consistency_weight: 1.0
  prediction_consistency_smooth_l1_beta: 0.1
  use_latent_consistency: False
  consistency_warmup_epochs: 20
  # Match the fused two-view SATLOSS6 protocol used by the other baselines.
  fuse_consistency_views: True

  use_gradnorm: False
  use_learned_task_weighting: True
  task_weight_lr: 5.e-4
  task_weight_init_logits: [0.0, 0.0, 0.0]
  task_weight_logit_min: -3.0
  task_weight_logit_max: 3.0
  task_weight_min_weights: [0.1, 0.1, 0.1]
  task_weight_base_weights: [0.15, 0.15, 0.70]
  task_weight_warmup_epochs: 20
  soft_worst_case_tau: 0.1

~~~

### DrivAerML preprocessor


~~~python
#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import signal
import tempfile
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def _true_random_sample_rows(hf, dataset_names, n_rows, k, seed, chunk_rows=1_000_000):
    """True uniform random sample without replacement using sequential chunk scan.

    We assign each row an i.i.d. random key and keep the k smallest keys.
    This is exact (uniform) and disk-friendly (sequential H5 reads).
    """
    if n_rows <= 0:
        idx = np.empty((0,), dtype=np.int64)
        arrays = [np.empty((0,), dtype=np.float32) for _ in dataset_names]
        return idx, arrays, False

    if k <= 0:
        idx = np.arange(n_rows, dtype=np.int64)
        arrays = [np.asarray(hf[name][:], dtype=np.float32) for name in dataset_names]
        return idx, arrays, False

    if k >= n_rows:
        idx = np.arange(n_rows, dtype=np.int64)
        arrays = [np.asarray(hf[name][:], dtype=np.float32) for name in dataset_names]
        if k > n_rows:
            rep = np.arange(k - n_rows, dtype=np.int64) % n_rows
            idx = np.concatenate([idx, rep], axis=0)
            arrays = [np.concatenate([a, a[rep]], axis=0) for a in arrays]
        return idx, arrays, bool(k > n_rows)

    rng = np.random.default_rng(seed)
    res_keys = np.empty((0,), dtype=np.float64)
    res_idx = np.empty((0,), dtype=np.int64)
    res_fields = [np.empty((0,) + hf[name].shape[1:], dtype=np.float32) for name in dataset_names]

    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        m = end - start
        chunk_keys = rng.random(m, dtype=np.float64)
        chunk_idx = np.arange(start, end, dtype=np.int64)
        chunk_fields = [np.asarray(hf[name][start:end], dtype=np.float32) for name in dataset_names]

        if res_keys.size == 0:
            if m <= k:
                res_keys = chunk_keys
                res_idx = chunk_idx
                res_fields = chunk_fields
            else:
                sel = np.argpartition(chunk_keys, k - 1)[:k]
                res_keys = chunk_keys[sel]
                res_idx = chunk_idx[sel]
                res_fields = [cf[sel] for cf in chunk_fields]
            continue

        all_keys = np.concatenate([res_keys, chunk_keys], axis=0)
        sel = np.argpartition(all_keys, k - 1)[:k]
        res_keys = all_keys[sel]
        all_idx = np.concatenate([res_idx, chunk_idx], axis=0)
        res_idx = all_idx[sel]
        for i in range(len(dataset_names)):
            all_field = np.concatenate([res_fields[i], chunk_fields[i]], axis=0)
            res_fields[i] = all_field[sel]

    return res_idx, res_fields, False


def _save_npy(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".tmp.",
        suffix=".npy",
        delete=False,
    ) as tf:
        tmp_name = tf.name
    try:
        np.save(tmp_name, arr)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def process_run(args):
    run_id, input_root, output_root, surface_k, volume_k, seed, overwrite, chunk_rows_surface, chunk_rows_volume = args
    in_dir = Path(input_root) / f"run_{run_id}"
    out_dir = Path(output_root) / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_probe = out_dir / "surface_coords.npy"
    if out_probe.is_file() and not overwrite:
        return run_id, "skipped"

    bfile = in_dir / f"boundary_{run_id}.h5"
    vfile = in_dir / f"volume_{run_id}_filtered.h5"
    if not bfile.is_file() or not vfile.is_file():
        return run_id, "missing"

    with h5py.File(bfile, "r") as hb, h5py.File(vfile, "r") as hv:
        ns = int(hb["coords"].shape[0])
        nv = int(hv["coords"].shape[0])

        sidx, s_fields, srep = _true_random_sample_rows(
            hb,
            [
                "coords",
                "areas",
                "normals",
                "pMeanTrim",
                "pPrime2MeanTrim",
                "wallShearStressMeanTrim_x",
                "wallShearStressMeanTrim_y",
                "wallShearStressMeanTrim_z",
            ],
            ns,
            surface_k,
            seed + run_id * 17,
            chunk_rows=chunk_rows_surface,
        )
        (
            s_coords,
            s_areas,
            s_normals,
            s_p,
            s_pp2,
            s_wx,
            s_wy,
            s_wz,
        ) = s_fields

        vidx, v_fields, vrep = _true_random_sample_rows(
            hv,
            ["coords", "UMeanTrim", "pMeanTrim"],
            nv,
            volume_k,
            seed + run_id * 31,
            chunk_rows=chunk_rows_volume,
        )
        v_coords, v_u, v_p = v_fields
        mask_len = int(hv["mask"].shape[0]) if "mask" in hv else -1

    _save_npy(out_dir / "surface_indices.npy", sidx)
    _save_npy(out_dir / "surface_coords.npy", s_coords)
    _save_npy(out_dir / "surface_areas.npy", s_areas)
    _save_npy(out_dir / "surface_normals.npy", s_normals)
    _save_npy(out_dir / "surface_pMeanTrim.npy", s_p)
    _save_npy(out_dir / "surface_pPrime2MeanTrim.npy", s_pp2)
    _save_npy(out_dir / "surface_wallShearStressMeanTrim_x.npy", s_wx)
    _save_npy(out_dir / "surface_wallShearStressMeanTrim_y.npy", s_wy)
    _save_npy(out_dir / "surface_wallShearStressMeanTrim_z.npy", s_wz)

    _save_npy(out_dir / "volume_indices.npy", vidx)
    _save_npy(out_dir / "volume_coords.npy", v_coords)
    _save_npy(out_dir / "volume_UMeanTrim.npy", v_u)
    _save_npy(out_dir / "volume_pMeanTrim.npy", v_p)

    meta = {
        "run_id": run_id,
        "surface_points": int(s_coords.shape[0]),
        "volume_points": int(v_coords.shape[0]),
        "volume_mask_length_original": int(mask_len),
        "surface_sample_replace": srep,
        "volume_sample_replace": vrep,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_id, "ok"


def compute_stats(output_root, train_ids):
    surf_sum = np.zeros((1,), dtype=np.float64)
    surf_sq = np.zeros((1,), dtype=np.float64)
    surf_n = 0
    vol_sum = np.zeros((3,), dtype=np.float64)
    vol_sq = np.zeros((3,), dtype=np.float64)
    vol_n = 0
    min_pos = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    max_pos = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for rid in tqdm(train_ids, desc="Computing stats"):
        run_dir = Path(output_root) / f"run_{rid}"
        sc = np.load(run_dir / "surface_coords.npy", mmap_mode="r")
        sp = np.load(run_dir / "surface_pMeanTrim.npy", mmap_mode="r")
        vc = np.load(run_dir / "volume_coords.npy", mmap_mode="r")
        vu = np.load(run_dir / "volume_UMeanTrim.npy", mmap_mode="r")

        surf_sum += sp.astype(np.float64).reshape(-1, 1).sum(axis=0)
        surf_sq += (sp.astype(np.float64).reshape(-1, 1) ** 2).sum(axis=0)
        surf_n += int(sp.shape[0])

        vol_sum += vu.astype(np.float64).sum(axis=0)
        vol_sq += (vu.astype(np.float64) ** 2).sum(axis=0)
        vol_n += int(vu.shape[0])

        min_pos = np.minimum(min_pos, np.minimum(sc.min(axis=0), vc.min(axis=0)))
        max_pos = np.maximum(max_pos, np.maximum(sc.max(axis=0), vc.max(axis=0)))

    surf_mean = (surf_sum / max(surf_n, 1)).astype(np.float32)
    vol_mean = (vol_sum / max(vol_n, 1)).astype(np.float32)
    surf_var = (surf_sq - (surf_sum ** 2) / max(surf_n, 1)) / max(surf_n - 1, 1)
    vol_var = (vol_sq - (vol_sum ** 2) / max(vol_n, 1)) / max(vol_n - 1, 1)
    surf_std = np.sqrt(np.clip(surf_var, 1e-12, None)).astype(np.float32)
    vol_std = np.sqrt(np.clip(vol_var, 1e-12, None)).astype(np.float32)

    np.save(Path(output_root) / "surface_stats_v2_h5.npy", np.stack([surf_mean, surf_std]))
    np.save(Path(output_root) / "volume_stats_v2_h5.npy", np.stack([vol_mean, vol_std]))
    np.save(Path(output_root) / "position_stats_v2_h5.npy", np.stack([min_pos.astype(np.float32), max_pos.astype(np.float32)]))


def main():
    ap = argparse.ArgumentParser(description="Preprocess DrivAerML H5 into fixed-size fast NPY layout.")
    ap.add_argument("--input-root", default="/mnt/ssdraid/drivaer_data")
    ap.add_argument("--output-root", default="/mnt/ssdraid/parsa/drivaerml_preprocessed")
    ap.add_argument("--surface-points", type=int, default=131072)  # 128k
    ap.add_argument("--volume-points", type=int, default=262144)  # nearest power-of-two to 250k
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk-rows-surface", type=int, default=1_000_000)
    ap.add_argument("--chunk-rows-volume", type=int, default=1_000_000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    def _hard_interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _hard_interrupt)
    signal.signal(signal.SIGTERM, _hard_interrupt)

    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    run_ids = sorted(
        int(p.name.split("_")[1])
        for p in in_root.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    )
    jobs = [
        (
            rid,
            str(in_root),
            str(out_root),
            args.surface_points,
            args.volume_points,
            args.seed,
            args.overwrite,
            args.chunk_rows_surface,
            args.chunk_rows_volume,
        )
        for rid in run_ids
    ]

    ok = 0
    status_counts = {"ok": 0, "skipped": 0, "missing": 0}
    try:
        if int(args.workers) <= 1:
            for j in tqdm(jobs, total=len(jobs), desc="Preprocessing runs"):
                rid, status = process_run(j)
                status_counts[status] = status_counts.get(status, 0) + 1
                if status == "ok":
                    ok += 1
        else:
            ctx = mp.get_context("spawn")
            with cf.ProcessPoolExecutor(max_workers=int(args.workers), mp_context=ctx) as ex:
                futs = [ex.submit(process_run, j) for j in jobs]
                for fut in tqdm(cf.as_completed(futs), total=len(futs), desc="Preprocessing runs"):
                    rid, status = fut.result()
                    status_counts[status] = status_counts.get(status, 0) + 1
                    if status == "ok":
                        ok += 1
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating immediately.")
        try:
            ex.shutdown(wait=False, cancel_futures=True)  # type: ignore[name-defined]
        except Exception:
            pass
        # Hard-exit so there is no lingering work.
        os._exit(130)

    # deterministic split consistent with loader logic
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(run_ids))
    n_test = max(1, int(round(0.2 * len(run_ids))))
    test_set = set(perm[:n_test].tolist())
    test_ids = [run_ids[i] for i in range(len(run_ids)) if i in test_set]
    train_ids = [run_ids[i] for i in range(len(run_ids)) if i not in test_set]

    compute_stats(str(out_root), train_ids)

    manifest = {
        "format": "drivaerml_preprocessed_v1",
        "input_root": str(in_root),
        "surface_points": int(args.surface_points),
        "volume_points": int(args.volume_points),
        "seed": int(args.seed),
        "num_runs_total": len(run_ids),
        "num_runs_ok": ok,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "files_per_run": [
            "surface_indices.npy",
            "surface_coords.npy",
            "surface_areas.npy",
            "surface_normals.npy",
            "surface_pMeanTrim.npy",
            "surface_pPrime2MeanTrim.npy",
            "surface_wallShearStressMeanTrim_x.npy",
            "surface_wallShearStressMeanTrim_y.npy",
            "surface_wallShearStressMeanTrim_z.npy",
            "volume_indices.npy",
            "volume_coords.npy",
            "volume_UMeanTrim.npy",
            "volume_pMeanTrim.npy",
            "meta.json",
        ],
        "status_counts": status_counts,
    }
    (out_root / "preprocessed_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done. Preprocessed data at: {out_root}")


if __name__ == "__main__":
    main()

~~~

### Dataset registry and factory


~~~python
from data.shapenetcar_dataset import ShapeNetCarDataset
from data.ahmedml_dataset import AhmedMLDataset
from data.ahmedml_dataset_v2 import AhmedMLDatasetV2, DrivAerMLDataset
from data.shiftsuv_dataset import ShiftSUVDataset
from data.shiftwing_dataset import ShiftWingDataset
from data.shift_submarine_dataset import ShiftSubmarineDataset
from data.pump_dataset import PumpDataset
from data.naca4_dataset import NACA4Dataset


# Mapping of dataset names to their corresponding classes and properties
datasets = {"ShapeNetCar": {"dataset": ShapeNetCarDataset, "spatial_dim": 3, "surf_channels": 1, "vol_channels": 3, "params_dim": 0, "fields": {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}},
            "AhmedML": {"dataset": AhmedMLDataset, "spatial_dim": 3, "surf_channels": 1, "vol_channels": 3, "params_dim": 0, "fields": {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}},
            "AhmedMLV2": {"dataset": AhmedMLDatasetV2, "spatial_dim": 3, "surf_channels": 7, "vol_channels": 4, "params_dim": 0, "fields": {"surface": ["pressure", "normal_x", "normal_y", "normal_z", "wall_shear_x", "wall_shear_y", "wall_shear_z"], "volume": ["pressure", "velocity_x", "velocity_y", "velocity_z"]}},
            "DrivAerML": {"dataset": DrivAerMLDataset, "spatial_dim": 3, "surf_channels": 7, "vol_channels": 4, "params_dim": 0, "fields": {"surface": ["pressure", "normal_x", "normal_y", "normal_z", "wall_shear_x", "wall_shear_y", "wall_shear_z"], "volume": ["pressure", "velocity_x", "velocity_y", "velocity_z"]}},
            "ShiftSUV": {"dataset": ShiftSUVDataset, "spatial_dim": 3, "surf_channels": 1, "vol_channels": 3, "params_dim": 0, "fields": {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}},
            "ShiftWing": {"dataset": ShiftWingDataset, "spatial_dim": 3, "surf_channels": 1, "vol_channels": 3, "params_dim": 2, "fields": {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}},
            "ShiftSubmarine": {"dataset": ShiftSubmarineDataset, "spatial_dim": 3, "surf_channels": 4, "vol_channels": 4, "params_dim": 0, "fields": {"surface": ["pressure", "wall_shear_x", "wall_shear_y", "wall_shear_z"], "volume": ["pressure", "velocity_x", "velocity_y", "velocity_z"]}},
            "Pump": {"dataset": PumpDataset, "spatial_dim": 3, "surf_channels": 7, "vol_channels": 4, "params_dim": 13, "fields": {"surface": ["pressure", "velocity_x", "velocity_y", "velocity_z", "wall_shear_x", "wall_shear_y", "wall_shear_z"], "volume": ["pressure", "velocity_x", "velocity_y", "velocity_z"]}},
            "NACA4": {"dataset": NACA4Dataset, "spatial_dim": 2, "surf_channels": 3, "vol_channels": 4, "params_dim": 0, "fields": {"surface": ["pressure", "normal_x", "normal_y"], "volume": ["pressure", "sdf", "velocity_x", "velocity_y"]}}
           }


def _uses_geometry_density(model_name):
    model_name = str(model_name)
    return model_name.startswith("SMART_SATLOSS") or "_SATLOSS" in model_name


def get_dataset(config):
    """Returns the dataset based on the provided configuration.

    Args:
        config: Configuration object containing dataset parameters.

    Returns:
        tuple: A tuple containing:
            - train_data: Training dataset.
            - test_data: Testing dataset.
            - stats: Stats for normalization.
            - spatial_dim: Spatial dimension of the dataset.
            - surf_channels: Number of surface channels.
            - vol_channels: Number of volume channels.
            - params_dim: Number of dimensions of simulation parameters.
    """
    
    dataset = config.dataset
    data_path = config.data_path
    print(f"Using dataset {dataset} stored at {data_path}")

    if dataset in datasets:
        spatial_dim = datasets[dataset]["spatial_dim"]
        surf_channels = datasets[dataset]["surf_channels"]
        vol_channels = datasets[dataset]["vol_channels"]
        params_dim = datasets[dataset]["params_dim"]
        fields = datasets[dataset]["fields"]
        dataset_kwargs = dict(geometry_points=config.num_body_points,
                              surface_points=config.num_surface_points,
                              volume_points=config.num_volume_points,
                              scale_positions=config.scale_positions)
        if dataset == "DrivAerML":
            dataset_kwargs["require_preprocessed"] = True
            dataset_kwargs["geometry_epoch_seeded_sampling"] = bool(getattr(config, "geometry_epoch_seeded_sampling", False))
            domain_split_json = str(getattr(config, "geometry_domain_split_json", "")).strip()
            if domain_split_json:
                dataset_kwargs["domain_split_json"] = domain_split_json
                dataset_kwargs["domain_split_train_cluster"] = int(
                    getattr(config, "geometry_domain_split_train_cluster", 0)
                )
                dataset_kwargs["domain_split_test_cluster"] = int(
                    getattr(config, "geometry_domain_split_test_cluster", 1)
                )
            dataset_kwargs["return_sample_info"] = getattr(config, "model_name", "") == "DARM"
            dataset_kwargs["return_half_precision"] = getattr(config, "model_name", "") == "DARM" and getattr(config, "precision", "") == "float16"
            model_name = getattr(config, "model_name", "")
            if _uses_geometry_density(model_name):
                arch = getattr(config, "architecture", {})
                density_knn_k = int(getattr(config, "density_knn_k", getattr(arch, "density_knn_k", 8)))
                density_neighbor_hops = int(getattr(config, "density_neighbor_hops", getattr(arch, "density_neighbor_hops", 1)))
                density_estimator = getattr(config, "density_estimator", getattr(arch, "density_estimator", "rk2"))
                dataset_kwargs["geometry_density_knn_k"] = density_knn_k
                dataset_kwargs["geometry_density_neighbor_hops"] = density_neighbor_hops
                dataset_kwargs["geometry_density_estimator"] = density_estimator
                dataset_kwargs["geometry_density_cache_dtype"] = getattr(config, "geometry_density_cache_dtype", "float16")
                if _uses_geometry_density(model_name) and model_name != "SMART_SATLOSS":
                    dataset_kwargs["return_geometry_density"] = True
                if model_name == "SMART_SATLOSS":
                    dataset_kwargs["return_surface_density"] = True
        if dataset == "ShiftSubmarine":
            model_name = str(getattr(config, "model_name", ""))
            dataset_kwargs["coordinate_normalization"] = getattr(
                config, "coordinate_normalization", "global_train_bounds"
            )
            dataset_kwargs["geometry_epoch_seeded_sampling"] = bool(
                getattr(config, "geometry_epoch_seeded_sampling", False)
            )
            if _uses_geometry_density(model_name):
                arch = getattr(config, "architecture", {})
                dataset_kwargs["return_geometry_density"] = True
                dataset_kwargs["geometry_density_knn_k"] = int(
                    getattr(config, "density_knn_k", getattr(arch, "density_knn_k", 16))
                )
                dataset_kwargs["geometry_density_neighbor_hops"] = int(
                    getattr(config, "density_neighbor_hops", getattr(arch, "density_neighbor_hops", 1))
                )
                dataset_kwargs["geometry_density_estimator"] = getattr(
                    config, "density_estimator", getattr(arch, "density_estimator", "kde")
                )
                dataset_kwargs["geometry_density_cache_dtype"] = getattr(
                    config, "geometry_density_cache_dtype", "float16"
                )
            dataset_kwargs["split_seed"] = int(getattr(config, "random_seed", 42))
        if dataset == "Pump":
            model_name = str(getattr(config, "model_name", ""))
            dataset_kwargs["coordinate_normalization"] = getattr(
                config, "coordinate_normalization", "global_train_bounds"
            )
            dataset_kwargs["geometry_epoch_seeded_sampling"] = bool(
                getattr(config, "geometry_epoch_seeded_sampling", False)
            )
            if _uses_geometry_density(model_name):
                arch = getattr(config, "architecture", {})
                dataset_kwargs["return_geometry_density"] = True
                dataset_kwargs["geometry_density_knn_k"] = int(
                    getattr(config, "density_knn_k", getattr(arch, "density_knn_k", 16))
                )
                dataset_kwargs["geometry_density_neighbor_hops"] = int(
                    getattr(config, "density_neighbor_hops", getattr(arch, "density_neighbor_hops", 1))
                )
                dataset_kwargs["geometry_density_estimator"] = getattr(
                    config, "density_estimator", getattr(arch, "density_estimator", "kde")
                )
                dataset_kwargs["geometry_density_cache_dtype"] = getattr(
                    config, "geometry_density_cache_dtype", "float16"
                )
            dataset_kwargs["split_seed"] = int(getattr(config, "random_seed", 42))
        if dataset == "NACA4":
            dataset_kwargs["manifest_variant"] = getattr(config, "manifest_variant", "full")
        train_data = datasets[dataset]["dataset"](data_path,
                                                  if_test=False,
                                                  **dataset_kwargs)
        test_data = datasets[dataset]["dataset"](data_path,
                                                 if_test=True,
                                                 **dataset_kwargs)
        stats = [train_data.mean_surf_data, train_data.std_surf_data,
                train_data.mean_vol_data, train_data.std_vol_data]
    else:
        raise ValueError(f"Unknown dataset ({config.dataset}) which is not supported!")
    
    return train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields


def prepare_dataset(config):
    """Prepare the dataset based on the provided configuration. Preparation means storing each sample in a 
    numpy array to speed up data loading during training and computing mean and std for normalization.

    Args:
        config: Configuration object containing dataset parameters.
    """
    
    dataset = config.dataset
    data_path = config.data_path
    print(f"Preparing dataset {dataset} stored at {data_path}")

    if dataset in datasets:
        dataset_kwargs = dict(if_test=False, prepare_data=True, copy_to_node=False)
        if dataset == "NACA4":
            dataset_kwargs["manifest_variant"] = getattr(config, "manifest_variant", "full")
        train_data = datasets[dataset]["dataset"](data_path, **dataset_kwargs)
        print(f"Dataset length: {len(train_data)}")
    else:
        raise ValueError(f"Unknown dataset ({config.dataset}) which is not supported!")

~~~

### DrivAerML loader, statistics, and sampling


~~~python
import os
import json
import shutil
import tempfile
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    from utils.geometry_density import estimate_log_sampling_density
except ImportError:  # pragma: no cover - package-style imports
    from smart.utils.geometry_density import estimate_log_sampling_density


class AhmedMLDatasetV2(Dataset):
    """DrivAerML/AhmedML-v2 dataset reader for run_*/boundary_*.h5 + volume_*_filtered.h5."""

    CACHE_VERSION = "v2_h5"

    def __init__(
        self,
        saved_folder="../data/",
        if_test=False,
        geometry_points=65536,
        surface_points=65536,
        volume_points=65536,
        copy_to_node=False,
        prepare_data=False,
        fast_approx_sampling=True,
        scale_positions=False,
        split_seed=42,
        test_fraction=0.2,
        stats_stride=20,
        stats_max_runs=32,
        io_oversample_factor=4,
        cache_root=None,
        require_preprocessed=False,
        return_geometry_density=False,
        return_surface_density=False,
        geometry_density_knn_k=8,
        geometry_density_neighbor_hops=1,
        geometry_density_estimator="rk2",
        geometry_density_cache_dtype="float16",
        geometry_epoch_seeded_sampling=False,
        return_sample_info=False,
        return_half_precision=False,
        domain_split_json="",
        domain_split_train_cluster=0,
        domain_split_test_cluster=1,
    ):
        geo_label = "all" if int(geometry_points) == 0 else str(int(geometry_points))
        surf_label = "all" if int(surface_points) == 0 else str(int(surface_points))
        vol_label = "all" if int(volume_points) == 0 else str(int(volume_points))
        print(f"Using {geo_label} geometry points, {surf_label} surface points, and {vol_label} volume points.")

        self.geometry_points = int(geometry_points)
        self.surface_points = int(surface_points)
        self.volume_points = int(volume_points)
        self.fast_approx_sampling = bool(fast_approx_sampling)
        self.if_test = bool(if_test)
        self.scale_positions = bool(scale_positions)
        self.stats_stride = max(1, int(stats_stride))
        self.stats_max_runs = max(1, int(stats_max_runs))
        self.io_oversample_factor = max(1, int(io_oversample_factor))

        self.file_path = os.path.abspath(saved_folder)
        self.cache_root = self._resolve_cache_root(cache_root)
        print(f"AhmedMLDatasetV2 cache root: {self.cache_root}")
        self.preprocessed_mode = (Path(self.file_path) / "preprocessed_manifest.json").is_file()
        self.require_preprocessed = bool(require_preprocessed)
        self.return_geometry_density = bool(return_geometry_density)
        self.return_surface_density = bool(return_surface_density)
        self.geometry_density_knn_k = max(1, int(geometry_density_knn_k))
        self.geometry_density_neighbor_hops = max(0, int(geometry_density_neighbor_hops))
        self.geometry_density_estimator = str(geometry_density_estimator)
        self.geometry_density_cache_dtype = str(geometry_density_cache_dtype)
        self.geometry_epoch_seeded_sampling = bool(geometry_epoch_seeded_sampling)
        self.return_sample_info = bool(return_sample_info)
        self.return_half_precision = bool(return_half_precision)
        self.domain_split_json = str(domain_split_json or "").strip()
        self.domain_split_train_cluster = int(domain_split_train_cluster)
        self.domain_split_test_cluster = int(domain_split_test_cluster)
        self._shared_epoch = mp.Value("i", 0, lock=False)
        self._geometry_density_ram_cache = OrderedDict()
        self._preprocessed_memmap_cache = OrderedDict()
        # Keep a materially larger in-memory cache so repeated epochs do not
        # keep reloading density tensors from disk for the same runs.
        self._geometry_density_ram_cache_max_entries = 512
        self._preprocessed_memmap_cache_max_entries = 16
        if self.require_preprocessed and not self.preprocessed_mode:
            raise FileNotFoundError(
                "DrivAerML is configured to use preprocessed-only mode, but "
                f"`preprocessed_manifest.json` was not found in {self.file_path}."
            )
        if self.preprocessed_mode:
            print("AhmedMLDatasetV2: detected preprocessed DrivAerML layout.")

        # Conservative defaults; overwritten by stats when available.
        if scale_positions:
            self.min_pos = torch.tensor([-4.0, -4.0, -4.0], dtype=torch.float32)
            self.max_pos = torch.tensor([6.0, 6.0, 6.0], dtype=torch.float32)
        else:
            self.min_pos = torch.tensor([-4.0, -2.0, -2.0], dtype=torch.float32)
            self.max_pos = torch.tensor([6.0, 2.0, 2.0], dtype=torch.float32)

        self.surface_field_names = ["pressure", "normal_x", "normal_y", "normal_z", "wall_shear_x", "wall_shear_y", "wall_shear_z"]
        self.volume_field_names = ["pressure", "velocity_x", "velocity_y", "velocity_z"]

        self.all_ids = self._discover_ids()
        self._point_counts = self._load_point_counts_cache()
        self.training_ids, self.test_ids = self._resolve_split_ids(split_seed, test_fraction)
        self.data = self.test_ids if self.if_test else self.training_ids

        if prepare_data:
            print("Precompute numpy arrays...")
            self.precompute_numpy_arrays()
            print("Computing stats...")
            self.compute_stats()

        if copy_to_node:
            user = os.getenv("USER", "user")
            self.copy_data_to_node(f"/data/scratch/{user}/data/ahmedml_v2")

        self.load_stats()

    def _resolve_cache_root(self, cache_root):
        # Prefer user-provided cache root; otherwise use dataset dir only if writable.
        if cache_root:
            root = Path(cache_root).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            return root

        data_root = Path(self.file_path)
        if os.access(data_root, os.W_OK):
            return data_root

        fallback = Path.home() / "smart_parsa" / ".cache" / "ahmedml_v2"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _discover_ids(self):
        if self.preprocessed_mode:
            ids = []
            for entry in os.scandir(self.file_path):
                if not entry.is_dir() or not entry.name.startswith("run_"):
                    continue
                try:
                    rid = int(entry.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                run_dir = Path(entry.path)
                required = [
                    run_dir / "surface_coords.npy",
                    run_dir / "surface_pMeanTrim.npy",
                    run_dir / "volume_coords.npy",
                    run_dir / "volume_UMeanTrim.npy",
                ]
                if all(p.is_file() for p in required):
                    ids.append(rid)
            ids = sorted(set(ids))
            if not ids:
                raise FileNotFoundError(f"No preprocessed run_* folders found in {self.file_path}")
            print(f"Found {len(ids)} valid preprocessed run folders.")
            return ids

        ids = []
        for entry in os.scandir(self.file_path):
            if not entry.is_dir() or not entry.name.startswith("run_"):
                continue
            try:
                rid = int(entry.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if self._boundary_h5_path(rid).is_file() and self._volume_h5_path(rid).is_file():
                ids.append(rid)
        ids = sorted(set(ids))
        if not ids:
            raise FileNotFoundError(f"No valid run_* folders with boundary_*.h5 and volume_*_filtered.h5 found in {self.file_path}")
        print(f"Found {len(ids)} valid run folders.")
        return ids

    def _point_counts_cache_path(self):
        return self.cache_root / f"point_counts_{self.CACHE_VERSION}.json"

    def _load_point_counts_cache(self):
        path = self._point_counts_cache_path()
        if not path.is_file():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            out = {}
            for k, v in raw.items():
                rid = int(k)
                if isinstance(v, dict) and "surface" in v and "volume" in v:
                    out[rid] = {"surface": int(v["surface"]), "volume": int(v["volume"])}
            return out
        except Exception:
            return {}

    def _save_point_counts_cache(self):
        path = self._point_counts_cache_path()
        tmp = path.with_suffix(path.suffix + ".tmp")
        serializable = {str(k): v for k, v in self._point_counts.items()}
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        os.replace(tmp, path)

    def _get_point_counts(self, run_id):
        if self.preprocessed_mode:
            run_dir = self._run_dir(run_id)
            ns = int(np.load(run_dir / "surface_coords.npy", mmap_mode="r").shape[0])
            nv = int(np.load(run_dir / "volume_coords.npy", mmap_mode="r").shape[0])
            return ns, nv
        if run_id in self._point_counts:
            return self._point_counts[run_id]["surface"], self._point_counts[run_id]["volume"]
        with h5py.File(self._boundary_h5_path(run_id), "r") as hb:
            ns = int(hb["coords"].shape[0])
        with h5py.File(self._volume_h5_path(run_id), "r") as hv:
            nv = int(hv["coords"].shape[0])
        self._point_counts[run_id] = {"surface": ns, "volume": nv}
        # Persist incrementally so train/test dataset instances share this quickly.
        try:
            self._save_point_counts_cache()
        except Exception:
            pass
        return ns, nv

    @staticmethod
    def _split_ids(ids, seed=42, test_fraction=0.2):
        ids = list(ids)
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(len(ids))
        n_test = max(1, int(round(len(ids) * float(test_fraction))))
        test_idx = set(perm[:n_test].tolist())
        test_ids = [ids[i] for i in range(len(ids)) if i in test_idx]
        train_ids = [ids[i] for i in range(len(ids)) if i not in test_idx]
        return train_ids, test_ids

    def _resolve_split_ids(self, split_seed, test_fraction):
        if self.domain_split_json:
            split_path = Path(self.domain_split_json).expanduser()
            if not split_path.is_absolute():
                split_path = Path.cwd() / split_path
            if not split_path.is_file():
                raise FileNotFoundError(f"Geometry domain split JSON not found: {split_path}")
            with split_path.open("r", encoding="utf-8") as handle:
                split = json.load(handle)
            train_cluster = self.domain_split_train_cluster
            test_cluster = self.domain_split_test_cluster
            direction_key = f"train_cluster_{train_cluster}_test_cluster_{test_cluster}"
            direction = split.get(direction_key)
            if not isinstance(direction, dict):
                raise ValueError(
                    f"Geometry domain split JSON does not contain `{direction_key}`: {split_path}"
                )
            have = set(self.all_ids)
            train_ids = [int(run_id) for run_id in direction.get("train_ids", [])]
            test_ids = [int(run_id) for run_id in direction.get("test_ids", [])]
            train_ids = [run_id for run_id in train_ids if run_id in have]
            test_ids = [run_id for run_id in test_ids if run_id in have]
            if not train_ids or not test_ids or set(train_ids).intersection(test_ids):
                raise ValueError(
                    f"Invalid geometry domain split `{direction_key}` in {split_path}: "
                    f"train={len(train_ids)}, test={len(test_ids)}, overlap={set(train_ids).intersection(test_ids)}"
                )
            if set(train_ids).union(test_ids) != have:
                raise ValueError(
                    f"Geometry domain split `{direction_key}` does not cover all available runs: "
                    f"split={len(set(train_ids).union(test_ids))}, available={len(have)}"
                )
            print(
                f"[domain split] {direction_key}: train={len(train_ids)} runs, "
                f"test={len(test_ids)} runs; role={'test' if self.if_test else 'train'}"
            )
            return train_ids, test_ids
        if self.preprocessed_mode:
            manifest_file = Path(self.file_path) / "preprocessed_manifest.json"
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                train_ids = [int(x) for x in m.get("train_ids", [])]
                test_ids = [int(x) for x in m.get("test_ids", [])]
                if train_ids and test_ids:
                    have = set(self.all_ids)
                    train_ids = [x for x in train_ids if x in have]
                    test_ids = [x for x in test_ids if x in have]
                    if train_ids and test_ids:
                        return train_ids, test_ids
            except Exception:
                pass
        return self._split_ids(self.all_ids, split_seed, test_fraction)

    def _run_dir(self, run_id):
        return Path(self.file_path) / f"run_{run_id}"

    def _boundary_h5_path(self, run_id):
        return self._run_dir(run_id) / f"boundary_{run_id}.h5"

    def _volume_h5_path(self, run_id):
        return self._run_dir(run_id) / f"volume_{run_id}_filtered.h5"

    def _cache_paths(self, run_id):
        run_dir = self.cache_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "geometry": run_dir / f"geometry_{self.CACHE_VERSION}.npy",
            "surface": run_dir / f"surface_{self.CACHE_VERSION}.npy",
            "surface_targets": run_dir / f"surface_targets_{self.CACHE_VERSION}.npy",
            "volume": run_dir / f"volume_{self.CACHE_VERSION}.npy",
            "volume_targets": run_dir / f"volume_targets_{self.CACHE_VERSION}.npy",
        }

    def _geometry_density_cache_path(self, run_id):
        run_dir = self.cache_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scale_tag = "scaled" if self.scale_positions else "noscale"
        dtype_tag = self.geometry_density_cache_dtype
        estimator_tag = self.geometry_density_estimator
        if estimator_tag == "kde":
            estimator_tag = "kde_mean64"
        return run_dir / (
            f"geometry_log_density_{self.CACHE_VERSION}_{scale_tag}"
            f"_{estimator_tag}"
            f"_k{self.geometry_density_knn_k}"
            f"_h{self.geometry_density_neighbor_hops}"
            f"_{dtype_tag}.npy"
        )

    def set_epoch(self, epoch):
        self._shared_epoch.value = int(epoch)

    def get_epoch(self):
        return int(self._shared_epoch.value)

    def _make_epoch_rng(self, run_id, stream_id):
        # Mix run/stream into the epoch seed so worker ordering does not affect samples.
        seed = np.random.SeedSequence([self.get_epoch(), int(run_id), int(stream_id)])
        return np.random.default_rng(seed)

    def _load_full_geometry_mesh(self, run_id):
        if self.preprocessed_mode:
            surf_coords = np.load(self._run_dir(run_id) / "surface_coords.npy", mmap_mode="r")
            geo_mesh = torch.from_numpy(np.array(surf_coords, dtype=np.float32, copy=True))
            return self._normalize_pos(geo_mesh, self.min_pos, self.max_pos)

        with h5py.File(self._boundary_h5_path(run_id), "r") as hb:
            bcoords = np.asarray(hb["coords"], dtype=np.float32)
        geo_mesh = torch.tensor(bcoords, dtype=torch.float32)
        geo_mask = self._finite_mask(geo_mesh)
        geo_mesh = geo_mesh[geo_mask]
        if geo_mesh.shape[0] == 0:
            raise ValueError(f"Run {run_id} has empty geometry after finite filtering.")
        return self._normalize_pos(geo_mesh, self.min_pos, self.max_pos)

    def _get_preprocessed_arrays(self, run_id):
        cache_key = int(run_id)
        cached = self._preprocessed_memmap_cache.get(cache_key)
        if cached is not None:
            self._preprocessed_memmap_cache.move_to_end(cache_key)
            return cached

        run_dir = self._run_dir(run_id)
        arrays = {
            "surf_coords": np.load(run_dir / "surface_coords.npy", mmap_mode="r"),
            "surf_p": np.load(run_dir / "surface_pMeanTrim.npy", mmap_mode="r"),
            "surf_n": np.load(run_dir / "surface_normals.npy", mmap_mode="r"),
            "surf_wx": np.load(run_dir / "surface_wallShearStressMeanTrim_x.npy", mmap_mode="r"),
            "surf_wy": np.load(run_dir / "surface_wallShearStressMeanTrim_y.npy", mmap_mode="r"),
            "surf_wz": np.load(run_dir / "surface_wallShearStressMeanTrim_z.npy", mmap_mode="r"),
            "vol_coords": np.load(run_dir / "volume_coords.npy", mmap_mode="r"),
            "vol_u": np.load(run_dir / "volume_UMeanTrim.npy", mmap_mode="r"),
            "vol_p": np.load(run_dir / "volume_pMeanTrim.npy", mmap_mode="r"),
        }
        self._preprocessed_memmap_cache[cache_key] = arrays
        self._preprocessed_memmap_cache.move_to_end(cache_key)
        while len(self._preprocessed_memmap_cache) > self._preprocessed_memmap_cache_max_entries:
            self._preprocessed_memmap_cache.popitem(last=False)
        return arrays

    def _load_or_compute_full_geometry_density(self, run_id, expected_n=None):
        cache_path = self._geometry_density_cache_path(run_id)
        ram_key = str(cache_path)
        cached = self._geometry_density_ram_cache.get(ram_key)
        if cached is not None and (expected_n is None or int(cached.shape[0]) == int(expected_n)):
            self._geometry_density_ram_cache.move_to_end(ram_key)
            return cached

        if cache_path.is_file():
            try:
                arr = np.load(cache_path)
                if expected_n is None or int(arr.shape[0]) == int(expected_n):
                    tensor = torch.from_numpy(np.asarray(arr))
                    self._remember_geometry_density(ram_key, tensor)
                    return tensor
            except Exception:
                pass

        full_geo_mesh = self._load_full_geometry_mesh(run_id)
        log_density = estimate_log_sampling_density(
            full_geo_mesh.unsqueeze(0),
            knn_k=self.geometry_density_knn_k,
            neighbor_hops=self.geometry_density_neighbor_hops,
            estimator=self.geometry_density_estimator,
        ).squeeze(0).cpu()

        remembered = log_density
        if self.geometry_density_cache_dtype == "float16":
            cache_arr = log_density.numpy().astype(np.float16, copy=False)
            remembered = torch.from_numpy(cache_arr)
        else:
            cache_arr = log_density.numpy().astype(np.float32, copy=False)
            remembered = torch.from_numpy(cache_arr)
        try:
            self._atomic_save_npy(cache_path, cache_arr)
        except Exception:
            pass
        self._remember_geometry_density(ram_key, remembered)
        return remembered

    def _load_or_compute_geometry_density(self, run_id, geo_mesh, can_cache):
        if can_cache:
            cached = self._load_or_compute_full_geometry_density(run_id, expected_n=int(geo_mesh.shape[0]))
            if int(cached.shape[0]) == int(geo_mesh.shape[0]):
                return cached

        log_density = estimate_log_sampling_density(
            geo_mesh.unsqueeze(0),
            knn_k=self.geometry_density_knn_k,
            neighbor_hops=self.geometry_density_neighbor_hops,
            estimator=self.geometry_density_estimator,
        ).squeeze(0).cpu()

        remembered = log_density
        if can_cache:
            if self.geometry_density_cache_dtype == "float16":
                cache_arr = log_density.numpy().astype(np.float16, copy=False)
                remembered = torch.from_numpy(cache_arr)
            else:
                cache_arr = log_density.numpy().astype(np.float32, copy=False)
                remembered = torch.from_numpy(cache_arr)
            try:
                self._atomic_save_npy(cache_path, cache_arr)
            except Exception:
                pass
        self._remember_geometry_density(ram_key, remembered)

        return remembered

    def _remember_geometry_density(self, ram_key, tensor):
        self._geometry_density_ram_cache[ram_key] = tensor
        self._geometry_density_ram_cache.move_to_end(ram_key)
        while len(self._geometry_density_ram_cache) > self._geometry_density_ram_cache_max_entries:
            self._geometry_density_ram_cache.popitem(last=False)

    def _try_load_surface_density_subset_from_cache(self, run_id, surf_idx, expected_n):
        cache_path = self._geometry_density_cache_path(run_id)
        if not cache_path.is_file():
            return None
        try:
            arr = np.load(cache_path, mmap_mode="r")
            if int(arr.shape[0]) != int(expected_n):
                return None
            surf_idx_np = surf_idx.detach().cpu().numpy().astype(np.int64, copy=False)
            subset = np.asarray(arr[surf_idx_np], dtype=np.float32)
            return torch.from_numpy(subset)
        except Exception:
            return None

    @staticmethod
    def _finite_mask(*arrays):
        mask = None
        for arr in arrays:
            cur = torch.isfinite(arr).all(dim=-1) if arr.ndim > 1 else torch.isfinite(arr)
            mask = cur if mask is None else (mask & cur)
        return mask

    @staticmethod
    def _atomic_save_npy(path: Path, array: np.ndarray):
        """Write npy atomically to avoid half-written files seen by other workers."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".tmp.", suffix=".npy", delete=False) as tf:
            tmp_name = tf.name
        try:
            np.save(tmp_name, array)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass

    def _load_case_arrays(self, run_id, write_cache=True):
        cache = self._cache_paths(run_id)
        if all(p.is_file() for p in cache.values()):
            try:
                geo_mesh = torch.tensor(np.load(cache["geometry"]), dtype=torch.float32)
                surf_mesh = torch.tensor(np.load(cache["surface"]), dtype=torch.float32)
                surf_data = torch.tensor(np.load(cache["surface_targets"]), dtype=torch.float32)
                vol_mesh = torch.tensor(np.load(cache["volume"]), dtype=torch.float32)
                vol_data = torch.tensor(np.load(cache["volume_targets"]), dtype=torch.float32)
                return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data
            except Exception as exc:
                # Corrupted/truncated cache from interrupted or concurrent writes. Rebuild from H5.
                print(f"Warning: cache for run_{run_id} is invalid ({exc}). Rebuilding from H5.")
                for p in cache.values():
                    try:
                        if p.exists():
                            p.unlink()
                    except OSError:
                        pass

        with h5py.File(self._boundary_h5_path(run_id), "r") as hb:
            bcoords = np.asarray(hb["coords"], dtype=np.float32)
            p_surf = np.asarray(hb["pMeanTrim"], dtype=np.float32).reshape(-1, 1)
            normals = np.asarray(hb["normals"], dtype=np.float32)
            if normals.ndim == 1:
                normals = normals.reshape(-1, 1)
            normals = normals[:, :3]
            wsx = np.asarray(hb["wallShearStressMeanTrim_x"], dtype=np.float32).reshape(-1, 1)
            wsy = np.asarray(hb["wallShearStressMeanTrim_y"], dtype=np.float32).reshape(-1, 1)
            wsz = np.asarray(hb["wallShearStressMeanTrim_z"], dtype=np.float32).reshape(-1, 1)

        with h5py.File(self._volume_h5_path(run_id), "r") as hv:
            vcoords = np.asarray(hv["coords"], dtype=np.float32)
            p_vol = np.asarray(hv["pMeanTrim"], dtype=np.float32).reshape(-1, 1)
            u = np.asarray(hv["UMeanTrim"], dtype=np.float32)
            if u.ndim == 1:
                u = u.reshape(-1, 1)
            u = u[:, :3]

        geo_mesh = torch.tensor(bcoords, dtype=torch.float32)
        surf_mesh = torch.tensor(bcoords, dtype=torch.float32)
        surf_data = torch.tensor(np.concatenate([p_surf, normals, wsx, wsy, wsz], axis=1), dtype=torch.float32)
        vol_mesh = torch.tensor(vcoords, dtype=torch.float32)
        vol_data = torch.tensor(np.concatenate([p_vol, u], axis=1), dtype=torch.float32)

        surf_mask = self._finite_mask(geo_mesh, surf_mesh, surf_data)
        vol_mask = self._finite_mask(vol_mesh, vol_data)
        geo_mesh, surf_mesh, surf_data = geo_mesh[surf_mask], surf_mesh[surf_mask], surf_data[surf_mask]
        vol_mesh, vol_data = vol_mesh[vol_mask], vol_data[vol_mask]
        if geo_mesh.shape[0] == 0 or surf_mesh.shape[0] == 0 or vol_mesh.shape[0] == 0:
            raise ValueError(f"Run {run_id} has empty arrays after finite filtering.")

        if write_cache:
            self._atomic_save_npy(cache["geometry"], geo_mesh.numpy())
            self._atomic_save_npy(cache["surface"], surf_mesh.numpy())
            self._atomic_save_npy(cache["surface_targets"], surf_data.numpy())
            self._atomic_save_npy(cache["volume"], vol_mesh.numpy())
            self._atomic_save_npy(cache["volume_targets"], vol_data.numpy())

        return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data

    def copy_data_to_node(self, path, force_copy=False):
        if not os.path.exists(path) or force_copy:
            print(f"Creating directory {path}")
            os.makedirs(path, exist_ok=True)
            for run_id in self.all_ids:
                src_dir = self._run_dir(run_id)
                dst_dir = Path(path) / f"run_{run_id}"
                dst_dir.mkdir(parents=True, exist_ok=True)
                for file in src_dir.glob("*.npy"):
                    dst = dst_dir / file.name
                    if not dst.exists():
                        shutil.copy2(file, dst)
            for name in [
                f"volume_stats_{self.CACHE_VERSION}.npy",
                f"surface_stats_{self.CACHE_VERSION}.npy",
                f"position_stats_{self.CACHE_VERSION}.npy",
                f"split_{self.CACHE_VERSION}.json",
            ]:
                src = self.cache_root / name
                dst = Path(path) / name
                if src.is_file() and not dst.exists():
                    shutil.copy2(src, dst)
        else:
            print(f"Data already copied to {path}, skipping copy step.")
        self.file_path = path

    def precompute_numpy_arrays(self):
        for run_id in self.all_ids:
            print(f"Precompute numpy arrays for run_{run_id}")
            self._load_case_arrays(run_id, write_cache=True)

    def _stats_paths(self):
        return (
            self.cache_root / f"volume_stats_{self.CACHE_VERSION}.npy",
            self.cache_root / f"surface_stats_{self.CACHE_VERSION}.npy",
            self.cache_root / f"position_stats_{self.CACHE_VERSION}.npy",
        )

    def load_stats(self):
        vol_file, surf_file, pos_file = self._stats_paths()
        if not (vol_file.is_file() and surf_file.is_file() and pos_file.is_file()):
            if self.if_test:
                raise FileNotFoundError(
                    f"Stats files missing ({self.CACHE_VERSION}). Run training split once or `python3 smart/prepare.py --config-name=drivaerml`."
                )
            print(
                "Stats files not found. Computing FAST sampled statistics from training split "
                f"(max_runs={self.stats_max_runs}, stride={self.stats_stride})..."
            )
            if self.preprocessed_mode:
                self.compute_stats_from_preprocessed()
            else:
                self.compute_stats_fast()

        print("Loading stats")
        surf = np.load(surf_file)
        vol = np.load(vol_file)
        pos = np.load(pos_file)
        if surf.shape[-1] != len(self.surface_field_names) or vol.shape[-1] != len(self.volume_field_names):
            if self.if_test:
                raise ValueError(
                    f"Stats channel mismatch. Found surface={surf.shape[-1]}, volume={vol.shape[-1]} "
                    f"but expected surface={len(self.surface_field_names)}, volume={len(self.volume_field_names)}. "
                    "Recompute stats with the training split."
                )
            print("Stats shape mismatch for current target channels, recomputing fast stats...")
            if self.preprocessed_mode:
                self.compute_stats_from_preprocessed()
            else:
                self.compute_stats_fast()
            surf = np.load(surf_file)
            vol = np.load(vol_file)
            pos = np.load(pos_file)
        self.mean_surf_data = torch.tensor(surf[0], dtype=torch.float32)
        self.std_surf_data = torch.tensor(surf[1], dtype=torch.float32)
        self.mean_vol_data = torch.tensor(vol[0], dtype=torch.float32)
        self.std_vol_data = torch.tensor(vol[1], dtype=torch.float32)
        self.min_pos = torch.tensor(pos[0], dtype=torch.float32)
        self.max_pos = torch.tensor(pos[1], dtype=torch.float32)

    @staticmethod
    def _safe_std(sum_, sq_sum, count):
        if count <= 1:
            return torch.ones_like(sum_)
        var = (sq_sum - (sum_ ** 2) / count) / (count - 1)
        return torch.sqrt(torch.clamp(var, min=1e-12))

    def compute_stats(self):
        min_pos = torch.full((3,), np.inf, dtype=torch.float32)
        max_pos = torch.full((3,), -np.inf, dtype=torch.float32)

        surf_c = len(self.surface_field_names)
        vol_c = len(self.volume_field_names)

        surf_sum = torch.zeros((surf_c,), dtype=torch.float32)
        surf_sq_sum = torch.zeros((surf_c,), dtype=torch.float32)
        surf_count = 0

        vol_sum = torch.zeros((vol_c,), dtype=torch.float32)
        vol_sq_sum = torch.zeros((vol_c,), dtype=torch.float32)
        vol_count = 0

        stride = self.stats_stride
        for run_id in self.training_ids:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = self._load_case_arrays(run_id, write_cache=True)

            geo_s = geo_mesh[::stride]
            surf_s, surf_d = surf_mesh[::stride], surf_data[::stride]
            vol_s, vol_d = vol_mesh[::stride], vol_data[::stride]

            for d in range(3):
                max_pos[d] = max(max_pos[d], geo_s[:, d].max().item(), surf_s[:, d].max().item(), vol_s[:, d].max().item())
                min_pos[d] = min(min_pos[d], geo_s[:, d].min().item(), surf_s[:, d].min().item(), vol_s[:, d].min().item())

            surf_sum += surf_d.sum(dim=0)
            surf_sq_sum += (surf_d ** 2).sum(dim=0)
            surf_count += int(surf_d.shape[0])

            vol_sum += vol_d.sum(dim=0)
            vol_sq_sum += (vol_d ** 2).sum(dim=0)
            vol_count += int(vol_d.shape[0])

        self.mean_surf_data = surf_sum / max(surf_count, 1)
        self.std_surf_data = self._safe_std(surf_sum, surf_sq_sum, surf_count)
        self.mean_vol_data = vol_sum / max(vol_count, 1)
        self.std_vol_data = self._safe_std(vol_sum, vol_sq_sum, vol_count)
        self.min_pos = min_pos
        self.max_pos = max_pos

        vol_file, surf_file, pos_file = self._stats_paths()
        np.save(surf_file, np.stack([self.mean_surf_data.numpy(), self.std_surf_data.numpy()]))
        np.save(vol_file, np.stack([self.mean_vol_data.numpy(), self.std_vol_data.numpy()]))
        np.save(pos_file, np.stack([self.min_pos.numpy(), self.max_pos.numpy()]))

        split_file = self.cache_root / f"split_{self.CACHE_VERSION}.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump({"train_ids": self.training_ids, "test_ids": self.test_ids}, f, indent=2)

    def _load_h5_sample_for_stats(self, run_id, stride):
        with h5py.File(self._boundary_h5_path(run_id), "r") as hb:
            bcoords = np.asarray(hb["coords"][::stride], dtype=np.float32)
            p_surf = np.asarray(hb["pMeanTrim"][::stride], dtype=np.float32).reshape(-1, 1)
            normals = np.asarray(hb["normals"][::stride], dtype=np.float32)
            if normals.ndim == 1:
                normals = normals.reshape(-1, 1)
            normals = normals[:, :3]
            wsx = np.asarray(hb["wallShearStressMeanTrim_x"][::stride], dtype=np.float32).reshape(-1, 1)
            wsy = np.asarray(hb["wallShearStressMeanTrim_y"][::stride], dtype=np.float32).reshape(-1, 1)
            wsz = np.asarray(hb["wallShearStressMeanTrim_z"][::stride], dtype=np.float32).reshape(-1, 1)
        with h5py.File(self._volume_h5_path(run_id), "r") as hv:
            vcoords = np.asarray(hv["coords"][::stride], dtype=np.float32)
            p_vol = np.asarray(hv["pMeanTrim"][::stride], dtype=np.float32).reshape(-1, 1)
            u = np.asarray(hv["UMeanTrim"][::stride], dtype=np.float32)
            if u.ndim == 1:
                u = u.reshape(-1, 1)
            u = u[:, :3]

        geo_mesh = torch.tensor(bcoords, dtype=torch.float32)
        surf_mesh = torch.tensor(bcoords, dtype=torch.float32)
        surf_data = torch.tensor(np.concatenate([p_surf, normals, wsx, wsy, wsz], axis=1), dtype=torch.float32)
        vol_mesh = torch.tensor(vcoords, dtype=torch.float32)
        vol_data = torch.tensor(np.concatenate([p_vol, u], axis=1), dtype=torch.float32)

        surf_mask = self._finite_mask(geo_mesh, surf_mesh, surf_data)
        vol_mask = self._finite_mask(vol_mesh, vol_data)
        return geo_mesh[surf_mask], surf_mesh[surf_mask], surf_data[surf_mask], vol_mesh[vol_mask], vol_data[vol_mask]

    def compute_stats_fast(self):
        """Fast startup stats: sample directly from H5 with coarse stride and limited runs."""
        min_pos = torch.full((3,), np.inf, dtype=torch.float32)
        max_pos = torch.full((3,), -np.inf, dtype=torch.float32)

        surf_c = len(self.surface_field_names)
        vol_c = len(self.volume_field_names)

        surf_sum = torch.zeros((surf_c,), dtype=torch.float32)
        surf_sq_sum = torch.zeros((surf_c,), dtype=torch.float32)
        surf_count = 0

        vol_sum = torch.zeros((vol_c,), dtype=torch.float32)
        vol_sq_sum = torch.zeros((vol_c,), dtype=torch.float32)
        vol_count = 0

        run_ids = self.training_ids[: self.stats_max_runs]
        stride = self.stats_stride
        print(f"Fast stats over {len(run_ids)} runs...")

        for run_id in run_ids:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = self._load_h5_sample_for_stats(run_id, stride)
            if geo_mesh.shape[0] == 0 or surf_mesh.shape[0] == 0 or vol_mesh.shape[0] == 0:
                continue

            for d in range(3):
                max_pos[d] = max(max_pos[d], geo_mesh[:, d].max().item(), surf_mesh[:, d].max().item(), vol_mesh[:, d].max().item())
                min_pos[d] = min(min_pos[d], geo_mesh[:, d].min().item(), surf_mesh[:, d].min().item(), vol_mesh[:, d].min().item())

            surf_sum += surf_data.sum(dim=0)
            surf_sq_sum += (surf_data ** 2).sum(dim=0)
            surf_count += int(surf_data.shape[0])

            vol_sum += vol_data.sum(dim=0)
            vol_sq_sum += (vol_data ** 2).sum(dim=0)
            vol_count += int(vol_data.shape[0])

        self.mean_surf_data = surf_sum / max(surf_count, 1)
        self.std_surf_data = self._safe_std(surf_sum, surf_sq_sum, surf_count)
        self.mean_vol_data = vol_sum / max(vol_count, 1)
        self.std_vol_data = self._safe_std(vol_sum, vol_sq_sum, vol_count)
        self.min_pos = min_pos
        self.max_pos = max_pos

        vol_file, surf_file, pos_file = self._stats_paths()
        np.save(surf_file, np.stack([self.mean_surf_data.numpy(), self.std_surf_data.numpy()]))
        np.save(vol_file, np.stack([self.mean_vol_data.numpy(), self.std_vol_data.numpy()]))
        np.save(pos_file, np.stack([self.min_pos.numpy(), self.max_pos.numpy()]))

        split_file = self.cache_root / f"split_{self.CACHE_VERSION}.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump({"train_ids": self.training_ids, "test_ids": self.test_ids}, f, indent=2)

    def compute_stats_from_preprocessed(self):
        """Compute stats directly from preprocessed NPY files (no H5 access)."""
        surf_sum = np.zeros((len(self.surface_field_names),), dtype=np.float64)
        surf_sq = np.zeros((len(self.surface_field_names),), dtype=np.float64)
        surf_n = 0
        vol_sum = np.zeros((len(self.volume_field_names),), dtype=np.float64)
        vol_sq = np.zeros((len(self.volume_field_names),), dtype=np.float64)
        vol_n = 0
        min_pos = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        max_pos = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

        print(f"Computing stats from preprocessed files over {len(self.training_ids)} runs...")
        for rid in self.training_ids:
            run_dir = self._run_dir(rid)
            sc = np.load(run_dir / "surface_coords.npy", mmap_mode="r")
            sp = np.load(run_dir / "surface_pMeanTrim.npy", mmap_mode="r")
            sn = np.load(run_dir / "surface_normals.npy", mmap_mode="r")
            swx = np.load(run_dir / "surface_wallShearStressMeanTrim_x.npy", mmap_mode="r")
            swy = np.load(run_dir / "surface_wallShearStressMeanTrim_y.npy", mmap_mode="r")
            swz = np.load(run_dir / "surface_wallShearStressMeanTrim_z.npy", mmap_mode="r")
            vc = np.load(run_dir / "volume_coords.npy", mmap_mode="r")
            vp = np.load(run_dir / "volume_pMeanTrim.npy", mmap_mode="r")
            vu = np.load(run_dir / "volume_UMeanTrim.npy", mmap_mode="r")

            surf = np.concatenate(
                [
                    np.asarray(sp, dtype=np.float32).reshape(-1, 1),
                    np.asarray(sn, dtype=np.float32),
                    np.asarray(swx, dtype=np.float32).reshape(-1, 1),
                    np.asarray(swy, dtype=np.float32).reshape(-1, 1),
                    np.asarray(swz, dtype=np.float32).reshape(-1, 1),
                ],
                axis=1,
            ).astype(np.float64, copy=False)
            vol = np.concatenate(
                [
                    np.asarray(vp, dtype=np.float32).reshape(-1, 1),
                    np.asarray(vu, dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float64, copy=False)

            surf_sum += surf.sum(axis=0)
            surf_sq += (surf ** 2).sum(axis=0)
            surf_n += int(surf.shape[0])

            vol_sum += vol.sum(axis=0)
            vol_sq += (vol ** 2).sum(axis=0)
            vol_n += int(vol.shape[0])

            min_pos = np.minimum(min_pos, np.minimum(sc.min(axis=0), vc.min(axis=0)))
            max_pos = np.maximum(max_pos, np.maximum(sc.max(axis=0), vc.max(axis=0)))

        self.mean_surf_data = torch.tensor(surf_sum / max(surf_n, 1), dtype=torch.float32)
        self.std_surf_data = torch.tensor(
            np.sqrt(np.clip((surf_sq - (surf_sum ** 2) / max(surf_n, 1)) / max(surf_n - 1, 1), 1e-12, None)),
            dtype=torch.float32,
        )
        self.mean_vol_data = torch.tensor(vol_sum / max(vol_n, 1), dtype=torch.float32)
        self.std_vol_data = torch.tensor(
            np.sqrt(np.clip((vol_sq - (vol_sum ** 2) / max(vol_n, 1)) / max(vol_n - 1, 1), 1e-12, None)),
            dtype=torch.float32,
        )
        self.min_pos = torch.tensor(min_pos, dtype=torch.float32)
        self.max_pos = torch.tensor(max_pos, dtype=torch.float32)

        vol_file, surf_file, pos_file = self._stats_paths()
        np.save(surf_file, np.stack([self.mean_surf_data.numpy(), self.std_surf_data.numpy()]))
        np.save(vol_file, np.stack([self.mean_vol_data.numpy(), self.std_vol_data.numpy()]))
        np.save(pos_file, np.stack([self.min_pos.numpy(), self.max_pos.numpy()]))

    def _sample_idx(self, n, k, rng=None, replace=None):
        if k <= 0 or k >= n:
            return torch.arange(n, dtype=torch.long)
        if replace is None:
            replace = bool(self.fast_approx_sampling)
        if rng is not None:
            idx = rng.choice(n, size=k, replace=bool(replace))
            return torch.from_numpy(idx.astype(np.int64, copy=False))
        if replace:
            return torch.randint(0, n, (k,), dtype=torch.long)
        # Avoid torch.randperm(n) for huge n.
        idx = np.random.choice(n, size=k, replace=False)
        return torch.from_numpy(idx.astype(np.int64))

    @staticmethod
    def _normalize_pos(pos, min_pos, max_pos):
        denom = torch.clamp(max_pos - min_pos, min=1e-12)
        return (pos - min_pos) / denom

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _read_rows_h5(ds, idx_np):
        # h5py fancy indexing is fastest/most reliable with sorted indices.
        if idx_np.size == 0:
            return np.empty((0,) + ds.shape[1:], dtype=ds.dtype)
        # h5py requires strictly increasing indices (no duplicates).
        # We gather unique sorted rows, then expand back to original order.
        unique_sorted, inverse = np.unique(idx_np, return_inverse=True)
        arr_unique = ds[unique_sorted]
        return arr_unique[inverse]

    def _read_strided_pool(self, ds, n_total, k_target):
        """Read a near-sequential pool from H5 and sample locally.

        This avoids heavy random disk seeks on huge contiguous H5 datasets.
        """
        if k_target <= 0 or k_target >= n_total:
            return np.asarray(ds[:], dtype=np.float32)

        # Read a moderately larger pool, then subsample in memory.
        pool_target = min(n_total, max(k_target, k_target * self.io_oversample_factor))
        stride = max(1, n_total // pool_target)
        offset = int(np.random.randint(0, stride)) if stride > 1 else 0
        pool = ds[offset::stride]
        return np.asarray(pool, dtype=np.float32)

    def _strided_slice_params(self, n_total, k_target):
        if k_target <= 0 or k_target >= n_total:
            return 1, 0
        pool_target = min(n_total, max(k_target, k_target * self.io_oversample_factor))
        stride = max(1, n_total // pool_target)
        offset = int(np.random.randint(0, stride)) if stride > 1 else 0
        return stride, offset

    @staticmethod
    def _sample_local(arr, k):
        n = arr.shape[0]
        if k <= 0 or k >= n:
            return arr
        idx = np.random.choice(n, size=k, replace=False)
        return arr[idx]

    def _load_case_sampled_from_h5(self, run_id, return_sample_info=False):
        ns, nv = self._get_point_counts(run_id)

        with h5py.File(self._boundary_h5_path(run_id), "r") as hb:
            # Two independent pools for geometry/surface to keep stochasticity.
            stride_g, offset_g = self._strided_slice_params(ns, self.geometry_points)
            geo_pool = np.asarray(hb["coords"][offset_g::stride_g], dtype=np.float32)

            stride_s, offset_s = self._strided_slice_params(ns, self.surface_points)
            surf_pool = np.asarray(hb["coords"][offset_s::stride_s], dtype=np.float32)
            p_pool = np.asarray(hb["pMeanTrim"][offset_s::stride_s], dtype=np.float32).reshape(-1, 1)
            bcoords_geo = self._sample_local(geo_pool, self.geometry_points)
            # Keep surface coords and pressure aligned using shared local index.
            surf_n = min(surf_pool.shape[0], p_pool.shape[0])
            surf_pool = surf_pool[:surf_n]
            p_pool = p_pool[:surf_n]
            if self.surface_points > 0 and self.surface_points < surf_n:
                sidx = np.random.choice(surf_n, size=self.surface_points, replace=False)
                bcoords_surf = surf_pool[sidx]
                ps = p_pool[sidx]
            else:
                bcoords_surf = surf_pool
                ps = p_pool

        with h5py.File(self._volume_h5_path(run_id), "r") as hv:
            stride_v, offset_v = self._strided_slice_params(nv, self.volume_points)
            vcoords_pool = np.asarray(hv["coords"][offset_v::stride_v], dtype=np.float32)
            u_pool = np.asarray(hv["UMeanTrim"][offset_v::stride_v], dtype=np.float32)
            vol_n = min(vcoords_pool.shape[0], u_pool.shape[0])
            vcoords_pool = vcoords_pool[:vol_n]
            u_pool = u_pool[:vol_n]
            if self.volume_points > 0 and self.volume_points < vol_n:
                vidx = np.random.choice(vol_n, size=self.volume_points, replace=False)
                vcoords = vcoords_pool[vidx]
                u = u_pool[vidx]
            else:
                vcoords = vcoords_pool
                u = u_pool
            if u.ndim == 1:
                u = u.reshape(-1, 1)
            u = u[:, :3]

        geo_mesh = torch.from_numpy(bcoords_geo)
        surf_mesh = torch.from_numpy(bcoords_surf)
        surf_data = torch.from_numpy(ps)
        vol_mesh = torch.from_numpy(vcoords)
        vol_data = torch.from_numpy(u)

        sample_info = {
            "run_id": torch.tensor(int(run_id), dtype=torch.long),
            "source_ns": torch.tensor(int(ns), dtype=torch.long),
            "source_nv": torch.tensor(int(nv), dtype=torch.long),
        }
        if return_sample_info or self.return_sample_info:
            sample_info.update(
                {
                    "geo_idx": None,
                    "surf_idx": None,
                    "vol_idx": None,
                }
            )
            return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, sample_info
        return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data

    def _load_case_from_preprocessed(self, run_id, return_sample_info=False):
        arrays = self._get_preprocessed_arrays(run_id)
        surf_coords = arrays["surf_coords"]
        surf_p = arrays["surf_p"]
        surf_n = arrays["surf_n"]
        surf_wx = arrays["surf_wx"]
        surf_wy = arrays["surf_wy"]
        surf_wz = arrays["surf_wz"]
        vol_coords = arrays["vol_coords"]
        vol_u = arrays["vol_u"]
        vol_p = arrays["vol_p"]

        ns = int(surf_coords.shape[0])
        nv = int(vol_coords.shape[0])
        geo_rng = None
        if self.geometry_epoch_seeded_sampling and 0 < self.geometry_points < ns:
            geo_rng = self._make_epoch_rng(run_id, stream_id=0)
        use_full_geo = self.geometry_points <= 0 or self.geometry_points >= ns
        use_full_surf = self.surface_points <= 0 or self.surface_points >= ns
        use_full_vol = self.volume_points <= 0 or self.volume_points >= nv

        geo_idx_t = self._sample_idx(ns, self.geometry_points, rng=geo_rng, replace=False if geo_rng is not None else None)
        surf_idx_t = self._sample_idx(ns, self.surface_points)
        vol_idx_t = self._sample_idx(nv, self.volume_points)

        if use_full_geo:
            geo_mesh = torch.from_numpy(np.array(surf_coords, dtype=np.float32, copy=True))
        else:
            geo_idx = geo_idx_t.numpy().astype(np.int64, copy=False)
            geo_mesh = torch.from_numpy(np.asarray(surf_coords[geo_idx], dtype=np.float32))

        if use_full_surf:
            surf_mesh = torch.from_numpy(np.array(surf_coords, dtype=np.float32, copy=True))
            surf_data = torch.from_numpy(
                np.concatenate(
                    [
                        np.asarray(surf_p, dtype=np.float32).reshape(-1, 1),
                        np.asarray(surf_n, dtype=np.float32),
                        np.asarray(surf_wx, dtype=np.float32).reshape(-1, 1),
                        np.asarray(surf_wy, dtype=np.float32).reshape(-1, 1),
                        np.asarray(surf_wz, dtype=np.float32).reshape(-1, 1),
                    ],
                    axis=1,
                )
            )
        else:
            surf_idx = surf_idx_t.numpy().astype(np.int64, copy=False)
            surf_mesh = torch.from_numpy(np.asarray(surf_coords[surf_idx], dtype=np.float32))
            surf_data = torch.from_numpy(
                np.concatenate(
                    [
                        np.asarray(surf_p[surf_idx], dtype=np.float32).reshape(-1, 1),
                        np.asarray(surf_n[surf_idx], dtype=np.float32),
                        np.asarray(surf_wx[surf_idx], dtype=np.float32).reshape(-1, 1),
                        np.asarray(surf_wy[surf_idx], dtype=np.float32).reshape(-1, 1),
                        np.asarray(surf_wz[surf_idx], dtype=np.float32).reshape(-1, 1),
                    ],
                    axis=1,
                )
            )

        if use_full_vol:
            vol_mesh = torch.from_numpy(np.asarray(vol_coords, dtype=np.float32))
            vol_data = torch.from_numpy(
                np.concatenate(
                    [
                        np.asarray(vol_p, dtype=np.float32).reshape(-1, 1),
                        np.asarray(vol_u, dtype=np.float32),
                    ],
                    axis=1,
                )
            )
        else:
            vol_idx = vol_idx_t.numpy().astype(np.int64, copy=False)
            vol_mesh = torch.from_numpy(np.asarray(vol_coords[vol_idx], dtype=np.float32))
            vol_data = torch.from_numpy(
                np.concatenate(
                    [
                        np.asarray(vol_p[vol_idx], dtype=np.float32).reshape(-1, 1),
                        np.asarray(vol_u[vol_idx], dtype=np.float32),
                    ],
                    axis=1,
                )
            )

        sample_info = {
            "run_id": torch.tensor(int(run_id), dtype=torch.long),
            "source_ns": torch.tensor(int(ns), dtype=torch.long),
            "source_nv": torch.tensor(int(nv), dtype=torch.long),
        }
        if return_sample_info:
            sample_info.update(
                {
                    "geo_idx": geo_idx_t.to(dtype=torch.long),
                    "surf_idx": surf_idx_t.to(dtype=torch.long),
                }
            )
        if return_sample_info or self.return_sample_info:
            return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, sample_info
        return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data

    def __getitem__(self, idx):
        run_id = self.data[idx]
        ns, _ = self._get_point_counts(run_id)
        need_sample_info = self.return_sample_info or self.return_surface_density or (self.return_geometry_density and self.geometry_points > 0)
        if self.preprocessed_mode:
            loaded = self._load_case_from_preprocessed(run_id, return_sample_info=need_sample_info)
        else:
            # Fast path: sample directly from H5 so we avoid full-array materialization.
            loaded = self._load_case_sampled_from_h5(run_id, return_sample_info=need_sample_info)

        if need_sample_info:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, sample_info = loaded
        else:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = loaded
            sample_info = None

        geo_mesh = self._normalize_pos(geo_mesh, self.min_pos, self.max_pos)
        surf_mesh = self._normalize_pos(surf_mesh, self.min_pos, self.max_pos)
        vol_mesh = self._normalize_pos(vol_mesh, self.min_pos, self.max_pos)

        geo_log_density = None
        surf_log_density = None
        if self.return_geometry_density or self.return_surface_density:
            full_geo_log_density = None
            geo_idx = None if sample_info is None else sample_info.get("geo_idx")
            surf_idx = None if sample_info is None else sample_info.get("surf_idx")
            if self.return_geometry_density:
                if geo_idx is not None and 0 < self.geometry_points < ns:
                    full_geo_log_density = self._load_or_compute_full_geometry_density(run_id, expected_n=ns)
                    geo_log_density = full_geo_log_density.index_select(0, geo_idx.to(dtype=torch.long))
                else:
                    can_cache = self.geometry_points <= 0 or self.geometry_points >= ns
                    full_geo_log_density = self._load_or_compute_geometry_density(run_id, geo_mesh, can_cache=can_cache)
                    geo_log_density = full_geo_log_density
            if self.return_surface_density:
                if surf_idx is not None:
                    if full_geo_log_density is None or int(full_geo_log_density.shape[0]) != int(ns):
                        full_geo_log_density = self._load_or_compute_full_geometry_density(run_id, expected_n=ns)
                    surf_log_density = full_geo_log_density.index_select(0, surf_idx.to(dtype=torch.long))
                if full_geo_log_density is None and surf_log_density is None:
                    can_cache = self.geometry_points <= 0 or self.geometry_points >= ns
                    full_geo_log_density = self._load_or_compute_geometry_density(run_id, geo_mesh, can_cache=can_cache)
                if surf_log_density is None:
                    surf_log_density = estimate_log_sampling_density(
                        surf_mesh.unsqueeze(0),
                        knn_k=self.geometry_density_knn_k,
                        neighbor_hops=self.geometry_density_neighbor_hops,
                        estimator=self.geometry_density_estimator,
                    ).squeeze(0).cpu()

        surf_data = (surf_data - self.mean_surf_data) / torch.clamp(self.std_surf_data, min=1e-12)
        vol_data = (vol_data - self.mean_vol_data) / torch.clamp(self.std_vol_data, min=1e-12)

        if self.return_half_precision:
            geo_mesh = geo_mesh.to(dtype=torch.float16)
            surf_mesh = surf_mesh.to(dtype=torch.float16)
            surf_data = surf_data.to(dtype=torch.float16)
            vol_mesh = vol_mesh.to(dtype=torch.float16)
            vol_data = vol_data.to(dtype=torch.float16)

        if geo_log_density is not None and surf_log_density is not None:
            return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, geo_log_density, surf_log_density
        if surf_log_density is not None:
            return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, surf_log_density
        if geo_log_density is not None:
            return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, geo_log_density
        return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data


# Convenient alias.
DrivAerMLDataset = AhmedMLDatasetV2

~~~

### Density estimation utility


~~~python
from __future__ import annotations

import numpy as np
import torch

try:
    from torch_cluster import knn_graph as torch_cluster_knn_graph
except ImportError:  # pragma: no cover - optional acceleration backend
    torch_cluster_knn_graph = None

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:  # pragma: no cover - optional CPU fallback backend
    NearestNeighbors = None


def knn_edges_as_neighbor_center(points_b, k_cur):
    """Return canonicalized kNN edges as (neighbor, center)."""
    if torch_cluster_knn_graph is None:
        raise RuntimeError("torch_cluster knn_graph backend is not available.")

    try:
        edge_index = torch_cluster_knn_graph(
            points_b.float().contiguous(),
            k=k_cur,
            loop=False,
            flow="source_to_target",
        )
    except TypeError:
        edge_index = torch_cluster_knn_graph(
            points_b.float().contiguous(),
            k=k_cur,
            loop=False,
        )

    e0, e1 = edge_index[0], edge_index[1]
    n = int(points_b.shape[0])

    counts_e1 = torch.bincount(e1, minlength=n)
    counts_e0 = torch.bincount(e0, minlength=n)

    if torch.all(counts_e1 == k_cur):
        nbr, center = e0, e1
    elif torch.all(counts_e0 == k_cur):
        nbr, center = e1, e0
    else:
        raise RuntimeError(
            "Could not canonicalize kNN graph orientation: "
            f"counts_e0 range=({counts_e0.min().item()}, {counts_e0.max().item()}), "
            f"counts_e1 range=({counts_e1.min().item()}, {counts_e1.max().item()}), "
            f"expected exactly {k_cur} per center."
        )

    return nbr, center


def _tiny_f64():
    return torch.finfo(torch.float64).tiny


@torch.no_grad()
def estimate_log_sampling_density_hash(points, knn_k=8, neighbor_hops=1, eps=1e-6):
    """Spatial-hash fallback for local kNN density estimation."""
    pts = points.clamp(0.0, 1.0 - 1e-6)
    bsz, _, _ = pts.shape
    k_eff = max(1, int(knn_k))
    hop_eff = max(0, int(neighbor_hops))

    outputs = []
    for b in range(bsz):
        pts_b = pts[b]
        n = int(pts_b.shape[0])
        if n <= 1:
            outputs.append(torch.zeros((n,), device=pts.device, dtype=pts.dtype))
            continue

        k_cur = min(k_eff, n - 1)
        res = max(4, int(round((float(n) / float(max(k_cur, 1))) ** 0.5)))
        cells = torch.clamp((pts_b * res).floor().to(torch.int64), min=0, max=res - 1)
        hashes = cells[:, 0] + res * (cells[:, 1] + res * cells[:, 2])

        order = torch.argsort(hashes)
        sorted_hashes = hashes[order]
        unique_hashes, counts = torch.unique_consecutive(sorted_hashes, return_counts=True)
        starts = torch.cumsum(counts, dim=0) - counts

        cell_to_indices = {}
        for h, s, c in zip(unique_hashes.tolist(), starts.tolist(), counts.tolist()):
            cell_to_indices[h] = order[s : s + c]

        log_density_b = torch.empty((n,), device=pts.device, dtype=torch.float32)
        identity = torch.arange(n, device=pts.device, dtype=torch.int64)

        for h, idx_cell in cell_to_indices.items():
            cx = h % res
            cy = (h // res) % res
            cz = h // (res * res)

            candidate_hashes = set()
            cand_count = 0
            max_hops = max(hop_eff, 0)
            while cand_count <= k_cur and max_hops < res:
                candidate_hashes.clear()
                for dx in range(-max_hops, max_hops + 1):
                    nx = cx + dx
                    if nx < 0 or nx >= res:
                        continue
                    for dy in range(-max_hops, max_hops + 1):
                        ny = cy + dy
                        if ny < 0 or ny >= res:
                            continue
                        for dz in range(-max_hops, max_hops + 1):
                            nz = cz + dz
                            if nz < 0 or nz >= res:
                                continue
                            candidate_hashes.add(int(nx + res * (ny + res * nz)))

                cand_count = 0
                for nh in candidate_hashes:
                    cand = cell_to_indices.get(nh)
                    if cand is not None:
                        cand_count += int(cand.numel())

                if cand_count <= k_cur:
                    max_hops += 1

            candidate_chunks = [cell_to_indices[nh] for nh in candidate_hashes if nh in cell_to_indices]
            cand_idx = torch.unique(torch.cat(candidate_chunks, dim=0)) if candidate_chunks else idx_cell
            if cand_idx.numel() <= k_cur:
                cand_idx = identity

            d2 = torch.cdist(pts_b[idx_cell].float(), pts_b[cand_idx].float(), p=2.0).pow_(2)
            self_mask = idx_cell[:, None] == cand_idx[None, :]
            d2.masked_fill_(self_mask, float("inf"))
            kth = torch.topk(d2, k=k_cur, dim=-1, largest=False).values[:, -1]
            log_density_b[idx_cell] = -torch.log(torch.clamp(kth, min=eps))

        outputs.append(log_density_b.to(dtype=pts.dtype))

    return torch.stack(outputs, dim=0)


@torch.no_grad()
def estimate_log_sampling_density_tangent_cov(points, knn_k=8, eps=1e-6):
    """Estimate local surface sampling density from tangent covariance area.

    For each point x_j with k nearest neighbors N_k(j), compute the local
    covariance matrix

        C_j = (1 / k) * sum_{l in N_k(j)} (x_l - x_j)(x_l - x_j)^T

    and define the local tangent-plane area scale from the two largest
    eigenvalues:

        A_j propto sqrt(lambda_{j,1} * lambda_{j,2})
        rho_j propto 1 / A_j

    so

        log rho_j = -0.5 * (log(lambda_{j,1} + eps) + log(lambda_{j,2} + eps)).
    """
    pts = points.clamp(0.0, 1.0 - 1e-6)
    bsz, _, _ = pts.shape
    k_eff = max(1, int(knn_k))

    if torch_cluster_knn_graph is not None:
        outputs = []
        for b in range(bsz):
            pts_b = pts[b]
            n = int(pts_b.shape[0])
            if n <= 1:
                outputs.append(torch.zeros((n,), device=pts.device, dtype=pts.dtype))
                continue

            k_cur = min(k_eff, n - 1)
            nbr, center = knn_edges_as_neighbor_center(pts_b, k_cur)
            diffs = pts_b[nbr].float() - pts_b[center].float()
            outer = diffs.unsqueeze(-1) * diffs.unsqueeze(-2)  # [E, 3, 3]

            cov = torch.zeros((n, 3, 3), device=pts.device, dtype=torch.float32)
            center_expanded = center[:, None, None].expand(-1, 3, 3)
            cov.scatter_add_(0, center_expanded, outer)
            cov = cov / float(k_cur)

            eigvals = torch.linalg.eigvalsh(cov)
            lambda_2 = torch.clamp(eigvals[:, 1], min=eps)
            lambda_1 = torch.clamp(eigvals[:, 2], min=eps)
            log_density = -0.5 * (torch.log(lambda_1) + torch.log(lambda_2))
            outputs.append(log_density.to(dtype=pts.dtype))
        return torch.stack(outputs, dim=0)

    if NearestNeighbors is not None:
        outputs = []
        for b in range(bsz):
            pts_b = pts[b]
            n = int(pts_b.shape[0])
            if n <= 1:
                outputs.append(torch.zeros((n,), device=pts.device, dtype=pts.dtype))
                continue

            k_cur = min(k_eff, n - 1)
            pts_np = pts_b.detach().cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=k_cur + 1, algorithm="auto")
            _, indices = nbrs.fit(pts_np).kneighbors(return_distance=True)
            neigh_idx = indices[:, 1:]
            diffs = pts_np[neigh_idx] - pts_np[:, None, :]
            cov = np.einsum("nki,nkj->nij", diffs, diffs) / float(k_cur)
            eigvals = np.linalg.eigvalsh(cov)
            lambda_2 = np.clip(eigvals[:, 1], eps, None)
            lambda_1 = np.clip(eigvals[:, 2], eps, None)
            log_density = -0.5 * (np.log(lambda_1) + np.log(lambda_2))
            outputs.append(torch.from_numpy(log_density).to(device=pts.device, dtype=pts.dtype))
        return torch.stack(outputs, dim=0)

    # Last-resort fallback: keep the old isotropic estimator rather than fail.
    return estimate_log_sampling_density_hash(pts, knn_k=knn_k, neighbor_hops=1, eps=eps)


@torch.no_grad()
def estimate_log_sampling_density_kde(points, knn_k=8):
    """Estimate local sampling density with a Gaussian KDE on the kNN graph.

    For each point x_j and its k nearest neighbors N_k(j), define

        rho_j = (1 / k) * sum_{l in N_k(j)} exp(-||x_j - x_l||^2 / h^2)

    where h^2 is the mean squared edge length over the full local kNN graph
    of the current point cloud. Distances and exponentials are evaluated in
    float64 for numerical stability on dense normalized meshes.

    The 1 / k normalization removes the trivial dependence of the raw kernel
    sum on how many neighbors were requested. The returned quantity is still
    monotone with local sampling density, but its absolute value does not shift
    by +log(k) when knn_k changes.
    """
    pts = points.clamp(0.0, 1.0 - 1e-6)
    bsz, _, _ = pts.shape
    k_eff = max(1, int(knn_k))
    tiny = _tiny_f64()

    if torch_cluster_knn_graph is not None:
        outputs = []
        for b in range(bsz):
            pts_b = pts[b]
            n = int(pts_b.shape[0])
            if n <= 1:
                outputs.append(torch.zeros((n,), device=pts.device, dtype=pts.dtype))
                continue

            k_cur = min(k_eff, n - 1)
            nbr, center = knn_edges_as_neighbor_center(pts_b, k_cur)
            diffs64 = pts_b[nbr].to(dtype=torch.float64) - pts_b[center].to(dtype=torch.float64)
            d2 = diffs64.square().sum(dim=-1)
            h2 = torch.clamp(d2.mean(), min=tiny)
            kernels = torch.exp(-d2 / h2)

            density = torch.zeros((n,), device=pts.device, dtype=torch.float64)
            density.scatter_add_(0, center, kernels)
            density = density / float(k_cur)
            log_density = torch.log(torch.clamp(density, min=tiny))
            outputs.append(log_density.to(dtype=pts.dtype))
        return torch.stack(outputs, dim=0)

    if NearestNeighbors is not None:
        outputs = []
        for b in range(bsz):
            pts_b = pts[b]
            n = int(pts_b.shape[0])
            if n <= 1:
                outputs.append(torch.zeros((n,), device=pts.device, dtype=pts.dtype))
                continue

            k_cur = min(k_eff, n - 1)
            pts_np = pts_b.detach().cpu().numpy().astype(np.float64, copy=False)
            nbrs = NearestNeighbors(n_neighbors=k_cur + 1, algorithm="kd_tree", n_jobs=-1)
            distances, _ = nbrs.fit(pts_np).kneighbors(return_distance=True)
            d2 = np.square(distances[:, 1:], dtype=np.float64)
            h2 = max(float(d2.mean()), np.finfo(np.float64).tiny)
            density = np.exp(-d2 / h2, dtype=np.float64).mean(axis=1)
            log_density = np.log(np.clip(density, np.finfo(np.float64).tiny, None))
            outputs.append(torch.from_numpy(log_density).to(device=pts.device, dtype=pts.dtype))
        return torch.stack(outputs, dim=0)

    raise RuntimeError(
        "KDE density estimation requires either torch_cluster.knn_graph "
        "or sklearn.neighbors.NearestNeighbors to be available."
    )


@torch.no_grad()
def estimate_log_sampling_density(points, knn_k=8, neighbor_hops=1, eps=1e-6, range_tol=1e-4, estimator="rk2"):
    """Estimate local surface sampling density from full-cloud kNN radii."""
    if points.numel() == 0:
        return torch.zeros(points.shape[:2], device=points.device, dtype=points.dtype)

    pts_min = float(points.min().item())
    pts_max = float(points.max().item())
    if pts_min < -range_tol or pts_max > 1.0 + range_tol:
        raise ValueError(
            f"estimate_log_sampling_density expects coordinates normalized to [0, 1]. "
            f"Observed range [{pts_min:.6f}, {pts_max:.6f}]."
        )

    pts = points.clamp(0.0, 1.0 - 1e-6)
    estimator = str(estimator)

    if estimator == "tangent_cov":
        return estimate_log_sampling_density_tangent_cov(pts, knn_k=knn_k, eps=eps)
    if estimator == "kde":
        return estimate_log_sampling_density_kde(pts, knn_k=knn_k)
    if estimator != "rk2":
        raise ValueError(f"Unknown density estimator '{estimator}'. Expected 'rk2', 'tangent_cov', or 'kde'.")

    bsz, _, _ = pts.shape
    k_eff = max(1, int(knn_k))

    if torch_cluster_knn_graph is not None:
        outputs = []
        for b in range(bsz):
            pts_b = pts[b]
            n = int(pts_b.shape[0])
            if n <= 1:
                outputs.append(torch.zeros((n,), device=pts.device, dtype=pts.dtype))
                continue

            k_cur = min(k_eff, n - 1)
            nbr, center = knn_edges_as_neighbor_center(pts_b, k_cur)
            d2 = (pts_b[nbr].float() - pts_b[center].float()).pow(2).sum(dim=-1)

            kth_d2 = torch.zeros(n, device=pts.device, dtype=torch.float32)
            kth_d2.scatter_reduce_(0, center, d2, reduce="amax", include_self=False)
            kth_d2 = torch.clamp(kth_d2, min=eps)
            outputs.append((-torch.log(kth_d2)).to(dtype=pts.dtype))
        return torch.stack(outputs, dim=0)

    if NearestNeighbors is not None:
        outputs = []
        for b in range(bsz):
            pts_b = pts[b]
            n = int(pts_b.shape[0])
            if n <= 1:
                outputs.append(torch.zeros((n,), device=pts.device, dtype=pts.dtype))
                continue

            k_cur = min(k_eff, n - 1)
            nbrs = NearestNeighbors(n_neighbors=k_cur + 1, algorithm="auto")
            distances, _ = nbrs.fit(pts_b.detach().cpu().numpy()).kneighbors(return_distance=True)
            kth_d2 = torch.from_numpy(np.square(distances[:, -1])).to(device=pts.device, dtype=pts.dtype)
            outputs.append(-torch.log(torch.clamp(kth_d2, min=eps)))
        return torch.stack(outputs, dim=0)

    return estimate_log_sampling_density_hash(pts, knn_k=knn_k, neighbor_hops=neighbor_hops, eps=eps)

~~~

### Losses


~~~python
import torch


class RelL2Loss():
    """Relative L2 loss for PDEs adopted from https://github.com/BaratiLab/FactFormer/blob/main/loss_fn.py"""

    def __init__(self, dim=-2, eps=1e-5, reduction='sum', reduce_all=True):
        self.dim = dim
        self.eps = eps
        self.reduction = reduction
        self.reduce_all = reduce_all
    
    def __call__(self, y_hat, y):
        assert y_hat.shape == y.shape

        reduce_fn = torch.mean if self.reduction == 'mean' else torch.sum

        y_norm = reduce_fn((y ** 2), dim=self.dim)
        mask = y_norm < self.eps
        y_norm[mask] = self.eps
        diff = reduce_fn((y_hat - y) ** 2, dim=self.dim)
        diff = diff / y_norm  # [b, c]
        
        if self.reduce_all:
            diff = diff.sqrt().mean() # mean across channels and batch and any other dimensions
        else:
            diff = diff.sqrt() # do nothing
        return diff


class CombinedLoss():
    """Computes a combined loss by summing the surface and volume losses.

    The loss function is applied to the full surface tensor and the full volume
    tensor. This keeps the training objective consistent even when the dataset
    has multiple channels per field group (for example NACA4 now has surface
    pressure + normals and volume pressure + sdf + velocity).
    """

    def __init__(self, loss_fn, fields):
        self.loss_fn = loss_fn
        self.fields = fields

    def __call__(self, y_hat_surf, y_hat_vol, y_surf, y_vol):
        """Compute the combined surface and volume loss."""
        loss_surf = self.loss_fn(y_hat_surf, y_surf)
        loss_vol = self.loss_fn(y_hat_vol, y_vol)
        return loss_surf + loss_vol


class WeightedRelL2Loss():
    """Relative L2 loss with pointwise weights along the spatial/sample dimension."""

    def __init__(self, dim=-2, eps=1e-5, reduction='sum', reduce_all=True):
        self.dim = dim
        self.eps = eps
        self.reduction = reduction
        self.reduce_all = reduce_all

    def _reduce(self, x, weights):
        if self.reduction == "mean":
            denom = torch.clamp(weights.sum(dim=self.dim), min=self.eps)
            return (x * weights).sum(dim=self.dim) / denom
        return (x * weights).sum(dim=self.dim)

    def __call__(self, y_hat, y, point_weights):
        assert y_hat.shape == y.shape

        weights = point_weights
        if weights.ndim == y.ndim - 1:
            weights = weights.unsqueeze(-1)
        if weights.ndim != y.ndim:
            raise ValueError(
                f"point_weights must have shape broadcastable to {tuple(y.shape)}; "
                f"got {tuple(point_weights.shape)}"
            )

        weights = weights.to(device=y.device, dtype=y.dtype)
        y_norm = self._reduce(y ** 2, weights)
        y_norm = torch.clamp(y_norm, min=self.eps)
        diff = self._reduce((y_hat - y) ** 2, weights)
        diff = diff / y_norm

        if self.reduce_all:
            diff = diff.sqrt().mean()
        else:
            diff = diff.sqrt()
        return diff


class DensityWeightedSurfaceCombinedLoss():
    """Surface loss with density weights, standard loss on the volume branch."""

    def __init__(self, surface_loss_fn, volume_loss_fn):
        self.surface_loss_fn = surface_loss_fn
        self.volume_loss_fn = volume_loss_fn

    def __call__(self, y_hat_surf, y_hat_vol, y_surf, y_vol, surface_point_weights):
        loss_surf = self.surface_loss_fn(y_hat_surf, y_surf, surface_point_weights)
        loss_vol = self.volume_loss_fn(y_hat_vol, y_vol)
        return loss_surf + loss_vol

~~~

### Training utilities


~~~python
import wandb
import torch
import torch.nn as nn
import numpy as np
import random
import inspect
from loss.losses import RelL2Loss
from lion_pytorch import Lion
from omegaconf import OmegaConf, open_dict
import os
import json
import re
import hashlib


def make_grad_scaler(config):
    """Build an AMP scaler with optional model-specific stability settings."""
    return torch.amp.GradScaler(
        "cuda",
        init_scale=float(getattr(config, "amp_scaler_init_scale", 65536.0)),
        growth_factor=float(getattr(config, "amp_scaler_growth_factor", 2.0)),
        backoff_factor=float(getattr(config, "amp_scaler_backoff_factor", 0.5)),
        growth_interval=int(getattr(config, "amp_scaler_growth_interval", 2000)),
    )


def reset_scheduler_for_extension(scheduler, optimizer, total_steps):
    """Start a fresh cosine schedule for a post-checkpoint training extension."""
    if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        raise ValueError("scheduler_reset_on_resume currently supports only the cosine scheduler.")
    total_steps = max(1, int(total_steps))
    base_lrs = list(scheduler.base_lrs)
    if len(base_lrs) != len(optimizer.param_groups):
        raise ValueError("Scheduler and optimizer parameter-group counts do not match.")
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr
    scheduler.T_max = total_steps
    scheduler.last_epoch = -1
    scheduler._step_count = 0
    scheduler._last_lr = list(base_lrs)


def _slugify(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "run"


def _join_nonempty(parts):
    return "-".join(_slugify(part) for part in parts if part not in (None, "", []))


def infer_fields_from_config(config):
    dataset = getattr(config, "dataset", None)
    if dataset == "NACA4":
        return {"surface": ["pressure", "normal_x", "normal_y"], "volume": ["pressure", "sdf", "velocity_x", "velocity_y"]}
    if dataset in {"AhmedMLV2", "DrivAerML"}:
        return {"surface": ["pressure", "normal_x", "normal_y", "normal_z", "wall_shear_x", "wall_shear_y", "wall_shear_z"], "volume": ["pressure", "velocity_x", "velocity_y", "velocity_z"]}
    if dataset in {"ShapeNetCar", "AhmedML", "ShiftSUV"}:
        return {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}
    if dataset == "ShiftWing":
        return {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}
    if dataset == "ShiftSubmarine":
        return {
            "surface": ["pressure", "wall_shear_x", "wall_shear_y", "wall_shear_z"],
            "volume": ["pressure", "velocity_x", "velocity_y", "velocity_z"],
        }
    return None


def get_field_tag(fields):
    if not fields:
        return None
    surface = fields.get("surface", [])
    volume = fields.get("volume", [])
    surface_tag = "+".join([
        field.replace("velocity_", "v").replace("normal_", "n").replace("pressure", "p").replace("sdf", "sdf")
        for field in surface
    ])
    volume_tag = "+".join([
        field.replace("velocity_", "v").replace("normal_", "n").replace("pressure", "p").replace("sdf", "sdf")
        for field in volume
    ])
    return f"s-{surface_tag}-v-{volume_tag}"


def _safe_wandb_tag(tag, max_len=64):
    """W&B tags must be <=64 chars; compact long tags deterministically."""
    tag = str(tag)
    if len(tag) <= max_len:
        return tag
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8]
    # Keep a readable prefix and append stable hash.
    keep = max_len - len("-") - len(digest)
    return f"{tag[:keep]}-{digest}"


def get_run_name(config, fields=None):
    variant = getattr(config, "manifest_variant", None)
    parts = [
        config.model_name,
        getattr(config, "model_tag", None) if getattr(config, "model_tag", "") else None,
        getattr(config, "dataset", None),
        variant if variant and variant != "full" else None,
        f"s{getattr(config, 'random_seed', 'na')}",
    ]
    return _join_nonempty(parts)


def get_output_run_name(config, fields=None):
    return get_run_name(config, fields)



def apply_naca4_auto_point_budget(config, dataset_obj, for_cat=False):
    """Auto-resolve point budgets from the minimum non-zero surface count across the dataset."""
    if getattr(config, "dataset", None) != "NACA4":
        return None

    if not hasattr(dataset_obj, "get_min_surface_points_nonzero"):
        return None

    min_surface = int(dataset_obj.get_min_surface_points_nonzero())
    if min_surface <= 0:
        raise ValueError("Could not infer a positive minimum surface-point count from NACA4 dataset.")

    num_blocks = int(getattr(getattr(config, "architecture", {}), "num_encoder_decoder_blocks", 1))
    num_blocks = max(num_blocks, 1)

    effective_surface_points = min_surface
    anchor_points = max(1, effective_surface_points // num_blocks)

    config.num_body_points = effective_surface_points
    config.num_surface_points = effective_surface_points

    if hasattr(config, "architecture"):
        with open_dict(config.architecture):
            config.architecture.subsampled_geometry_points = effective_surface_points
            config.architecture.latent_geometry_points = anchor_points

    info = {
        "min_surface_points_nonzero": min_surface,
        "effective_surface_points": effective_surface_points,
        "num_blocks": num_blocks,
        "anchor_points": anchor_points,
    }

    if for_cat:
        with open_dict(config):
            config.stage1_surface_input_points = effective_surface_points
            config.stage1_surface_query_points = effective_surface_points
            config.stage2_surface_input_points = effective_surface_points
            config.stage2_surface_query_points = effective_surface_points
            config.stage3_surface_input_points = effective_surface_points

        stage3_vq = int(getattr(config, "stage3_volume_query_points", 0))
        if stage3_vq <= 0:
            stage3_vq = int(getattr(config, "num_volume_points", 0))
        if stage3_vq <= 0 and hasattr(dataset_obj, "get_min_volume_points_nonzero"):
            stage3_vq = int(dataset_obj.get_min_volume_points_nonzero())
        if stage3_vq <= 0:
            raise ValueError("Could not infer a positive stage3 volume query count.")

        with open_dict(config):
            config.stage3_volume_query_points = stage3_vq
            # Request: stage1 volume query should be 4x old value and equal to stage3 volume query.
            config.stage1_volume_query_points = stage3_vq

        info.update({
            "stage1_surface_input_points": int(config.stage1_surface_input_points),
            "stage1_surface_query_points": int(config.stage1_surface_query_points),
            "stage1_volume_query_points": int(config.stage1_volume_query_points),
            "stage2_surface_input_points": int(config.stage2_surface_input_points),
            "stage2_surface_query_points": int(config.stage2_surface_query_points),
            "stage3_surface_input_points": int(config.stage3_surface_input_points),
            "stage3_volume_query_points": int(config.stage3_volume_query_points),
        })

    return info


def print_point_budget(prefix, info):
    if not info:
        return
    print(f"[{prefix}] min surface points (non-zero across dataset): {info['min_surface_points_nonzero']}")
    print(f"[{prefix}] effective surface points: {info['effective_surface_points']}")
    print(f"[{prefix}] encoder/decoder blocks (M): {info['num_blocks']}")
    print(f"[{prefix}] anchor points (effective_surface/M): {info['anchor_points']}")
    for key in [
        "stage1_surface_input_points",
        "stage1_surface_query_points",
        "stage1_volume_query_points",
        "stage2_surface_input_points",
        "stage2_surface_query_points",
        "stage3_surface_input_points",
        "stage3_volume_query_points",
    ]:
        if key in info:
            print(f"[{prefix}] {key}: {info[key]}")

def initialize_gpu(random_seed, high_precision=True):
    """Initializes the GPU settings and sets the random seed."""
    
    # Device settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    if high_precision:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    return device


def initialize_wandb(config, wandb_config, model_files=[]):
    """Initializes wandb with the given config."""

    fields = getattr(config, "fields", None) or infer_fields_from_config(config)
    run_name = get_run_name(config, fields)
    tags = [
        f"seed_{config.random_seed}",
        getattr(config, "dataset", "unknown"),
        getattr(config, "model_name", "unknown"),
    ]
    if getattr(config, "manifest_variant", None) and getattr(config, "manifest_variant", "full") != "full":
        tags.append(f"variant_{config.manifest_variant}")
    if fields:
        tags.append(_safe_wandb_tag(f"fields_{get_field_tag(fields)}"))
    tags = [_safe_wandb_tag(t) for t in tags]

    run = wandb.init(
        name=run_name,
        project=wandb_config.project,
        entity=wandb_config.entity,
        mode=getattr(wandb_config, "mode", "online"),
        tags=tags,
    )
    
    # Add config to wandb
    wandb.config.update(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    
    # Add model files to wandb
    if model_files:
        artifact = wandb.Artifact("model-code", type="code")
        for file in model_files:
            artifact.add_file(file)
        wandb.log_artifact(artifact)
    
    print(f"Model {config.model_name}, "
          f"random seed: {config.random_seed}, "
          f"epochs: {config.epochs}, "
          f"learning rate: {config.learning_rate}")
    
    return run


def get_model_checkpoint_name(config):
    """Returns the model checkpoint name based on the config."""

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    variant = getattr(config, "manifest_variant", None)
    parts = [
        config.model_name,
        getattr(config, "model_tag", None) if getattr(config, "model_tag", "") else None,
        getattr(config, "dataset", None),
        variant if variant and variant != "full" else None,
        f"s{getattr(config, 'random_seed', 'na')}",
    ]
    return _join_nonempty(parts)


def count_model_params(model):
    """Calculates number of parameters of the given model. Complex-valued weights count as two weights (for imaginary
    and real part)."""
    
    params = []
    for p in model.parameters():
        if p.requires_grad:
            if torch.is_complex(p):
                params.append(2 * p.numel())
            else:
                params.append(p.numel())
    return sum(params)


def exclude_params_from_weight_decay(model,
                                     exclude=["bias", "filter_bias", "norm", "query_pos", "modulation_weight", "B", "hash", "table"],
                                     verbose=False):
    """Excludes the given parameters from the weight decay."""
    
    named_parameters = model.named_parameters()
    decay_parameters = []
    decay_parameters_names = []
    no_decay_parameters = []
    no_decay_parameters_names = []

    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if not any(ex in name for ex in exclude):
            decay_parameters_names.append(name)
            decay_parameters.append(param)
        else:
            no_decay_parameters_names.append(name)
            no_decay_parameters.append(param)

    if verbose:
        print("Exclude from weight decay:", no_decay_parameters_names)
        print("Weight decay for:", decay_parameters_names)

    grouped_parameters = [
        {"params": decay_parameters},
        {"params": no_decay_parameters, "weight_decay": 0.0}
    ]
    return grouped_parameters


def get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=-2, extra_param_groups=None):
    """Returns the optimizer, scheduler, and loss for the given model and config.
    
    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        config (object): Configuration object containing optimizer, scheduler, and loss function settings.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data, used to determine the number of steps per epoch.
        loss_dim (int, optional): Dimension over which to compute the relative L2 loss. Defaults to -2.
    
    Returns:
        tuple: A tuple containing:
            - optimizer (torch.optim.Optimizer): Configured optimizer.
            - scheduler (torch.optim.lr_scheduler._LRScheduler): Configured learning rate scheduler.
            - loss_fn (torch.nn.Module): Primary loss function.
            - rel_l2_loss_fn (torch.nn.Module): Relative L2 loss function (always returned for evaluation).
    
    Raises:
        ValueError: If an unsupported optimizer, scheduler, or loss function is specified in the config.
    """
    
    def _cuda_optimizer_impl_kwargs(optimizer_cls):
        if not torch.cuda.is_available():
            return {}
        try:
            parameters = inspect.signature(optimizer_cls).parameters
        except (TypeError, ValueError):
            return {}
        if "fused" in parameters:
            return {"fused": True}
        if "foreach" in parameters:
            return {"foreach": True}
        return {}

    extra_param_groups = list(extra_param_groups or [])

    # Get optimizer
    if config.optimizer == "adam":
        # we have to exclude the bias and weights from norms
        grouped_parameters = exclude_params_from_weight_decay(model)
        grouped_parameters = grouped_parameters + extra_param_groups
        optimizer = torch.optim.Adam(
            grouped_parameters,
            lr=config.learning_rate,
            weight_decay=1e-5,
            **_cuda_optimizer_impl_kwargs(torch.optim.Adam),
        )
    elif config.optimizer == "adamw":
        # we have to exclude the bias and weights from norms
        grouped_parameters = exclude_params_from_weight_decay(model, exclude=["bias", "norm", "query_pos", "B"])
        grouped_parameters = grouped_parameters + extra_param_groups
        optimizer = torch.optim.AdamW(
            grouped_parameters,
            lr=config.learning_rate,
            weight_decay=1e-4,
            **_cuda_optimizer_impl_kwargs(torch.optim.AdamW),
        )
    elif config.optimizer == "lion":
        grouped_parameters = exclude_params_from_weight_decay(model, exclude=["bias", "norm", "query_pos", "B"])
        grouped_parameters = grouped_parameters + extra_param_groups
        optimizer = Lion(grouped_parameters, lr=config.learning_rate, weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not supported!")
    
    # Get scheduler
    scheduler_warmup_fraction = float(
        getattr(config, "scheduler_warmup_fraction", getattr(config, "scheduler_warumup_fraction", 0.2))
    )

    if config.scheduler == "one-cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=config.learning_rate,
                                                        pct_start=scheduler_warmup_fraction,
                                                        div_factor=1e2, final_div_factor=1e3,
                                                        total_steps=config.epochs * len(train_loader))
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs * len(train_loader))
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
    elif config.scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1)
    else:
        raise ValueError("Scheduler not supported!")
    
    # Get loss functions
    if config.loss_fn == "mse":
        loss_fn = nn.MSELoss(reduction="mean")
    elif config.loss_fn == "l1":
        loss_fn = nn.L1Loss(reduction="mean")
    elif config.loss_fn == "rel_l2":
        loss_fn = RelL2Loss(dim=loss_dim, reduction="sum")
    else:
        raise ValueError("Loss function not supported!")
    
    # Get rel. L2 loss functions
    rel_l2_loss_fn = RelL2Loss(dim=loss_dim, reduction="sum")

    return optimizer, scheduler, loss_fn, rel_l2_loss_fn


def store_inference_results(dir, model_checkpoint_name, test_losses):
    """Stores inference results in a JSON file."""
    
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    with open(dir + "/" + model_checkpoint_name + "_full_inference.json", 'w') as f: 
        json.dump(test_losses, f)

~~~

### Vanilla surface-volume trainer


~~~python
from __future__ import annotations

import os
from timeit import default_timer

import hydra
import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from data.datasets import get_dataset
from loss.losses import CombinedLoss
from utils.utils import (
    apply_naca4_auto_point_budget,
    count_model_params,
    get_model_checkpoint_name,
    get_optimizer_scheduler_loss,
    initialize_gpu,
    initialize_wandb,
    make_grad_scaler,
    print_point_budget,
    reset_scheduler_for_extension,
)

CANON_SURF_FIELDS = ["pressure", "normal_x", "normal_y"]
CANON_VOL_FIELDS = ["pressure", "sdf", "velocity_x", "velocity_y"]


def unwrap_model(model):
    return model.module if isinstance(model, (DDP, DataParallel)) else model


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return {"enabled": False, "rank": 0, "local_rank": 0, "world_size": 1}
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training for the point-cloud models requires CUDA.")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return {"enabled": True, "rank": rank, "local_rank": local_rank, "world_size": world_size}


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def average_distributed_metrics(metrics, sample_count, device):
    keys = list(metrics)
    values = torch.tensor(
        [float(metrics[key]) for key in keys] + [float(sample_count)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    denominator = max(float(values[-1].item()), 1.0)
    return {key: float(values[index].item()) / denominator for index, key in enumerate(keys)}


def synchronized_step_valid(local_valid, device, distributed):
    if not distributed:
        return bool(local_valid)
    flag = torch.tensor(1 if local_valid else 0, device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def init_metric_dict(surface_fields, volume_fields):
    metrics = {"loss": 0.0, "rel_l2": 0.0, "rel_l2_surf": 0.0, "rel_l2_vol": 0.0}
    for field_name in surface_fields:
        metrics[f"rel_l2_surf_{field_name}"] = 0.0
    for field_name in volume_fields:
        metrics[f"rel_l2_vol_{field_name}"] = 0.0
    return metrics


def accumulate_channel_metrics(metrics, prefix, pred, gt, field_names, rel_l2_loss_fn, batch_size):
    for channel_idx, field_name in enumerate(field_names):
        channel_loss = rel_l2_loss_fn(pred[..., channel_idx:channel_idx + 1], gt[..., channel_idx:channel_idx + 1])
        metrics[f"{prefix}_{field_name}"] += channel_loss.item() * batch_size


def gradient_diagnostics(model):
    grad_sq = None
    param_sq = None
    max_grad = None
    finite = True
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        parameter_sq = parameter.detach().float().pow(2).sum()
        param_sq = parameter_sq if param_sq is None else param_sq + parameter_sq
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not torch.isfinite(gradient).all():
            finite = False
        gradient_sq = gradient.pow(2).sum()
        grad_sq = gradient_sq if grad_sq is None else grad_sq + gradient_sq
        gradient_max = gradient.abs().max()
        max_grad = gradient_max if max_grad is None else torch.maximum(max_grad, gradient_max)
    reference = next(model.parameters()).detach()
    zero = reference.new_zeros((), dtype=torch.float32)
    return {
        "grad_norm": torch.sqrt(grad_sq.clamp_min(0.0)) if grad_sq is not None else zero,
        "parameter_norm": torch.sqrt(param_sq.clamp_min(0.0)) if param_sq is not None else zero,
        "max_grad": max_grad if max_grad is not None else zero,
        "finite": finite,
    }


def snapshot_model_buffers(model):
    return {name: buffer.detach().clone() for name, buffer in unwrap_model(model).named_buffers()}


def restore_model_buffers(model, snapshot):
    if snapshot is None:
        return
    for name, value in unwrap_model(model).named_buffers():
        if name in snapshot:
            value.copy_(snapshot[name])


def model_buffers_are_finite(model):
    return all(bool(torch.isfinite(buffer).all().item()) for buffer in unwrap_model(model).buffers())


def add_canonical_field_metrics(wandb_dict, split, surface_fields, volume_fields, metric_values=None):
    metric_values = metric_values or {}
    for f in CANON_SURF_FIELDS:
        if f not in surface_fields:
            continue
        src_key = f"rel_l2_surf_{f}"
        wandb_dict[f"{split}/rel_l2_surf_{f}"] = metric_values.get(src_key, np.nan)
    for f in CANON_VOL_FIELDS:
        if f not in volume_fields:
            continue
        src_key = f"rel_l2_vol_{f}"
        wandb_dict[f"{split}/rel_l2_vol_{f}"] = metric_values.get(src_key, np.nan)


def add_all_field_metrics(wandb_dict, split, surface_fields, volume_fields, metric_values=None):
    metric_values = metric_values or {}
    for f in surface_fields:
        wandb_dict[f"{split}/rel_l2_surf_{f}"] = metric_values.get(f"rel_l2_surf_{f}", np.nan)
    for f in volume_fields:
        wandb_dict[f"{split}/rel_l2_vol_{f}"] = metric_values.get(f"rel_l2_vol_{f}", np.nan)


def load_partial_state_dict(model, checkpoint_path, device):
    if not checkpoint_path:
        return 0, 0
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = checkpoint.get("model_state_dict", checkpoint)
    target_model = model.module if hasattr(model, "module") else model
    target = target_model.state_dict()
    if any(key.startswith("module.") for key in source.keys()):
        source = {key.removeprefix("module."): value for key, value in source.items()}
    filtered = {}
    matched = 0
    skipped = 0
    for key, value in source.items():
        if key in target and target[key].shape == value.shape:
            filtered[key] = value
            matched += 1
        else:
            skipped += 1
    target.update(filtered)
    target_model.load_state_dict(target, strict=False)
    print(f"[resume] Loaded {matched} tensors from {checkpoint_path}; skipped {skipped} incompatible tensors.")
    return matched, skipped


def load_full_training_state(
    model,
    optimizer,
    scheduler,
    scaler,
    checkpoint_path,
    device,
    steps_per_epoch=None,
    load_scaler=True,
    reset_scheduler=False,
    target_epochs=None,
):
    """Restore a vanilla trainer checkpoint without replaying completed epochs."""
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be provided for full-state resume.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = checkpoint.get("model_state_dict")
    if source is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain model_state_dict.")

    target_model = model.module if hasattr(model, "module") else model
    if any(key.startswith("module.") for key in source):
        source = {key.removeprefix("module."): value for key, value in source.items()}
    target_model.load_state_dict(source, strict=True)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if optimizer_state is None or scheduler_state is None:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing optimizer_state_dict or scheduler_state_dict."
        )
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if load_scaler and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    resumed_epoch = int(checkpoint.get("epoch", -1))
    start_epoch = resumed_epoch + 1
    if reset_scheduler:
        if steps_per_epoch is None or target_epochs is None:
            raise ValueError("steps_per_epoch and target_epochs are required to reset a resumed scheduler.")
        extension_epochs = max(int(target_epochs) - start_epoch, 1)
        reset_scheduler_for_extension(scheduler, optimizer, extension_epochs * int(steps_per_epoch))
        print(f"[resume] Reset cosine scheduler for {extension_epochs} extension epochs.")
    global_step = checkpoint.get("global_step")
    if global_step is None:
        global_step = (resumed_epoch + 1) * int(steps_per_epoch or 0)
    global_step = int(global_step)
    best_rel_l2 = float(checkpoint.get("best_rel_l2", checkpoint.get("rel_l2_loss", np.inf)))
    print(
        f"[resume] Restored full training state from {checkpoint_path}: "
        f"epoch={resumed_epoch}, next_epoch={start_epoch}, global_step={global_step}, "
        f"best_rel_l2={best_rel_l2:.6g}"
    )
    return start_epoch, global_step, best_rel_l2


def _parse_batch(batch, params_dim):
    geo_log_density = None
    if params_dim > 0:
        if len(batch) == 7:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = batch
        elif len(batch) == 6:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params = batch
        else:
            raise ValueError(f"Unexpected batch size {len(batch)}")
    else:
        if len(batch) == 6:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, geo_log_density = batch
            params = None
        elif len(batch) == 5:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
            params = None
        else:
            raise ValueError(f"Unexpected batch size {len(batch)}")
    return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _record_stream(value, stream):
    if torch.is_tensor(value):
        if value.is_cuda:
            value.record_stream(stream)
        return
    if isinstance(value, dict):
        for item in value.values():
            _record_stream(item, stream)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _record_stream(item, stream)


class CudaPrefetchLoader:
    """Overlap host-to-device copies with the current training step."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.enabled = device.type == "cuda" and torch.cuda.is_available()
        self.stream = torch.cuda.Stream(device=device) if self.enabled else None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if not self.enabled:
            yield from self.loader
            return

        loader_iter = iter(self.loader)
        next_batch = None

        def preload():
            nonlocal next_batch
            try:
                batch = next(loader_iter)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(self.stream):
                next_batch = _move_to_device(batch, self.device)

        preload()
        while next_batch is not None:
            torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
            batch = next_batch
            _record_stream(batch, torch.cuda.current_stream(device=self.device))
            preload()
            yield batch


def run_surface_volume_training(cfg: DictConfig, model_cls, accepts_geo_log_density=False):
    config = cfg.experiment
    wandb_config = cfg.wandb
    multi_gpu_strategy = str(getattr(config, "multi_gpu_strategy", "single")).lower()
    if multi_gpu_strategy not in {"single", "data_parallel", "ddp"}:
        raise ValueError(
            f"Unsupported vanilla multi_gpu_strategy={multi_gpu_strategy!r}; "
            "use single, data_parallel, or ddp."
        )
    if multi_gpu_strategy == "data_parallel" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "multi_gpu_strategy=data_parallel must be launched with plain python, not torchrun."
        )
    if multi_gpu_strategy == "ddp" and int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        raise RuntimeError("multi_gpu_strategy=ddp requires torchrun with at least two processes.")
    dist_info = setup_distributed() if multi_gpu_strategy == "ddp" else {
        "enabled": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
    }
    is_main = not dist_info["enabled"] or dist_info["rank"] == 0
    run = initialize_wandb(config, wandb_config) if is_main else None
    device = initialize_gpu(config.random_seed, high_precision=False)

    gradient_norm = config.gradient_norm
    track_gradient_diagnostics = bool(getattr(config, "track_gradient_diagnostics", False))
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(config.precision, torch.float16)
    amp = config.amp
    if is_main:
        print(gradient_norm, amp, dtype)

    train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)

    def apply_vanilla_field_subset():
        nonlocal fields, surf_channels, vol_channels
        if config.dataset == "NACA4":
            fields = {"surface": ["pressure"], "volume": ["pressure", "velocity_x", "velocity_y"]}
            surf_channels = 1
            vol_channels = 3

    apply_vanilla_field_subset()
    if is_main:
        print(f"[{config.model_name}] training signals -> surface: {fields['surface']} | volume: {fields['volume']}")

    point_info = apply_naca4_auto_point_budget(config, train_data, for_cat=False)
    if point_info is not None:
        if is_main:
            print_point_budget(config.model_name, point_info)
        train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
        apply_vanilla_field_subset()
        if is_main:
            print(f"[{config.model_name}] training signals -> surface: {fields['surface']} | volume: {fields['volume']}")

    use_surface_supervision = len(fields["surface"]) > 0

    prefetch_factor = int(getattr(config, "prefetch_factor", 2))
    pin_memory = bool(getattr(config, "pin_memory", True))
    dl_common = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=pin_memory)
    if config.num_workers > 0:
        dl_common["prefetch_factor"] = prefetch_factor
        dl_common["persistent_workers"] = True

    train_sampler = DistributedSampler(train_data, shuffle=True) if dist_info["enabled"] else None
    test_sampler = DistributedSampler(test_data, shuffle=False) if dist_info["enabled"] else None
    train_loader = torch.utils.data.DataLoader(
        train_data,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **dl_common,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        sampler=test_sampler,
        **dl_common,
    )
    cuda_batch_prefetch = bool(getattr(config, "cuda_batch_prefetch", False)) and device.type == "cuda"
    train_batch_source = CudaPrefetchLoader(train_loader, device) if cuda_batch_prefetch else train_loader
    test_batch_source = CudaPrefetchLoader(test_loader, device) if cuda_batch_prefetch else test_loader
    if is_main:
        print(
            f"[dataloader] world_size={dist_info['world_size']}, "
            f"num_workers_per_rank={config.num_workers}, "
            f"prefetch_factor={prefetch_factor}, "
            f"cuda_batch_prefetch={cuda_batch_prefetch}"
        )

    mean_surf = stats[0][:surf_channels].to(device)
    std_surf = stats[1][:surf_channels].to(device)
    if config.dataset == "NACA4" and vol_channels == 2:
        mean_vol = stats[2][2:4].to(device)
        std_vol = stats[3][2:4].to(device)
    elif config.dataset == "NACA4" and vol_channels == 3:
        mean_vol = torch.stack([stats[2][0], stats[2][2], stats[2][3]]).to(device)
        std_vol = torch.stack([stats[3][0], stats[3][2], stats[3][3]]).to(device)
    else:
        mean_vol = stats[2][:vol_channels].to(device)
        std_vol = stats[3][:vol_channels].to(device)

    model_kwargs = {
        "spatial_dim": spatial_dim,
        "surface_channels": surf_channels,
        "volume_channels": vol_channels,
        "parameter_channels": params_dim,
    }
    merged_kwargs = {**model_kwargs, **config.architecture} if "architecture" in config else model_kwargs
    print(f"Model kwargs: {merged_kwargs}")
    model = model_cls(**merged_kwargs).to(device)
    rollback_buffers = bool(getattr(model, "rollback_buffers_on_nonfinite", False))
    if multi_gpu_strategy == "data_parallel" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        visible_gpus = torch.cuda.device_count()
        if int(config.batch_size) < visible_gpus:
            print(
                f"[multi-gpu warning] batch_size={int(config.batch_size)} is smaller than the number of visible GPUs "
                f"({visible_gpus}); DataParallel will underutilize devices."
            )
        model = DataParallel(
            model,
            device_ids=list(range(visible_gpus)),
            output_device=0,
            dim=0,
        )
        if is_main:
            print(f"[multi-gpu] Using DataParallel on {visible_gpus} GPUs.")
    elif dist_info["enabled"]:
        model = DDP(
            model,
            device_ids=[dist_info["local_rank"]],
            output_device=dist_info["local_rank"],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        if is_main:
            print(f"[multi-gpu] Using DDP on {dist_info['world_size']} GPUs.")

    resume_ckpt = str(getattr(config, "resume_ckpt", "")).strip()
    init_ckpt = str(getattr(config, "init_ckpt", "")).strip()
    resume_full_state = bool(getattr(config, "resume_full_state", False))
    if resume_full_state and not resume_ckpt:
        raise ValueError("resume_full_state=True requires experiment.resume_ckpt to be set.")
    if not resume_full_state and resume_ckpt:
        if is_main:
            print(f"[init] Loading model weights from experiment.resume_ckpt={resume_ckpt}")
        load_partial_state_dict(model, resume_ckpt, device)
    elif not resume_full_state and init_ckpt:
        if is_main:
            print(f"[init] Loading model weights from experiment.init_ckpt={init_ckpt}")
        load_partial_state_dict(model, init_ckpt, device)

    if is_main:
        print(f"Total parameters: {count_model_params(model)}")
    model_checkpoint_name = get_model_checkpoint_name(config)
    if is_main:
        print(f"Checkpoint name: {model_checkpoint_name}")
    if run is not None and bool(getattr(config, "wandb_watch_model", False)):
        run.watch(model, log="all")

    scaler = make_grad_scaler(config)
    optimizer, scheduler, loss_fn, rel_l2_loss_fn = get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=1)
    combined_loss_fn = CombinedLoss(loss_fn, fields) if use_surface_supervision else None

    loss_test_min = np.inf
    global_step = 0
    start_epoch = 0
    if resume_full_state:
        start_epoch, global_step, loss_test_min = load_full_training_state(
            model,
            optimizer,
            scheduler,
            scaler,
            resume_ckpt,
            device,
            steps_per_epoch=len(train_loader),
            load_scaler=not bool(getattr(config, "amp_scaler_reset_on_resume", False)),
            reset_scheduler=bool(getattr(config, "scheduler_reset_on_resume", False)),
            target_epochs=int(config.epochs),
        )
    log_every_n_steps = getattr(config, "log_every_n_steps", 10)

    try:
        for ep in tqdm(range(start_epoch, config.epochs), desc="Epochs", dynamic_ncols=True, disable=not is_main):
            t1 = default_timer()
            # Propagate the epoch to datasets that use epoch-seeded point sampling.
            if hasattr(train_data, "set_epoch"):
                train_data.set_epoch(ep)
            if hasattr(test_data, "set_epoch"):
                test_data.set_epoch(0)
            if train_sampler is not None:
                train_sampler.set_epoch(ep)
            train_losses = init_metric_dict(fields["surface"], fields["volume"])
            if track_gradient_diagnostics:
                for key in ("gradient_norm_raw", "parameter_norm", "gradient_max_abs", "gradient_nonfinite"):
                    train_losses[key] = 0.0
            test_losses = init_metric_dict(fields["surface"], fields["volume"])
            train_sample_count = 0

            model.train()
            train_pbar = tqdm(
                train_batch_source,
                desc=f"Train {ep + 1}/{config.epochs}",
                leave=False,
                dynamic_ncols=True,
                disable=not is_main,
            )
            for batch_idx, batch in enumerate(train_pbar):
                geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = _parse_batch(batch, params_dim)
                geo_mesh = geo_mesh.to(device)
                surf_mesh = surf_mesh.to(device)
                surf_data = surf_data.to(device)
                vol_mesh = vol_mesh.to(device)
                vol_data = vol_data.to(device)
                if params is not None:
                    params = params.to(device)
                if geo_log_density is not None:
                    geo_log_density = geo_log_density.to(device)

                if config.dataset == "NACA4":
                    surf_data = surf_data[..., :1]
                    vol_data = torch.cat([vol_data[..., :1], vol_data[..., 2:4]], dim=-1)

                optimizer.zero_grad(set_to_none=True)
                buffer_snapshot = snapshot_model_buffers(model) if rollback_buffers else None
                if amp:
                    with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                        if accepts_geo_log_density:
                            y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params, geo_log_density=geo_log_density)
                        else:
                            y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)
                        # Keep the large pointwise reductions in float32. A float16
                        # sum over 65k query points can overflow and produce NaN
                        # gradients even when the forward loss is finite.
                        loss = (
                            combined_loss_fn(y_hat_surf.float(), y_hat_vol.float(), surf_data.float(), vol_data.float())
                            if use_surface_supervision
                            else loss_fn(y_hat_vol.float(), vol_data.float())
                        )
                    loss_is_finite = bool(torch.isfinite(loss).item())
                    if not synchronized_step_valid(loss_is_finite, device, dist_info["enabled"]):
                        if is_main:
                            print(f"[warn] Non-finite training loss at epoch {ep} batch {batch_idx}; skipping optimizer step.")
                        restore_model_buffers(model, buffer_snapshot)
                        optimizer.zero_grad(set_to_none=True)
                        if rollback_buffers:
                            scaler.update(new_scale=max(1.0, 0.5 * scaler.get_scale()))
                        continue
                    prev_scale = scaler.get_scale()
                    scaler.scale(loss).backward()
                    if gradient_norm is not None or track_gradient_diagnostics:
                        scaler.unscale_(optimizer)
                    gradient_stats = gradient_diagnostics(model) if track_gradient_diagnostics else None
                    gradients_are_finite = gradient_stats is None or gradient_stats["finite"]
                    if not synchronized_step_valid(gradients_are_finite, device, dist_info["enabled"]):
                        if is_main:
                            print(
                                f"[warn] Non-finite gradients at epoch {ep} batch {batch_idx}; "
                                "skipping optimizer step and reducing AMP scale."
                            )
                        restore_model_buffers(model, buffer_snapshot)
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update(new_scale=max(1.0, 0.5 * scaler.get_scale()))
                        continue
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if rollback_buffers and not model_buffers_are_finite(model):
                        restore_model_buffers(model, buffer_snapshot)
                        raise FloatingPointError(
                            f"Non-finite PTv3 state after optimizer step at epoch={ep} batch={batch_idx}."
                        )
                    if scaler.get_scale() >= prev_scale:
                        scheduler.step()
                else:
                    if accepts_geo_log_density:
                        y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params, geo_log_density=geo_log_density)
                    else:
                        y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)
                    loss = (
                        combined_loss_fn(y_hat_surf.float(), y_hat_vol.float(), surf_data.float(), vol_data.float())
                        if use_surface_supervision
                        else loss_fn(y_hat_vol.float(), vol_data.float())
                    )
                    loss_is_finite = bool(torch.isfinite(loss).item())
                    if not synchronized_step_valid(loss_is_finite, device, dist_info["enabled"]):
                        if is_main:
                            print(f"[warn] Non-finite training loss at epoch {ep} batch {batch_idx}; skipping optimizer step.")
                        restore_model_buffers(model, buffer_snapshot)
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    loss.backward()
                    gradient_stats = gradient_diagnostics(model) if track_gradient_diagnostics else None
                    gradients_are_finite = gradient_stats is None or gradient_stats["finite"]
                    if not synchronized_step_valid(gradients_are_finite, device, dist_info["enabled"]):
                        if is_main:
                            print(f"[warn] Non-finite gradients at epoch {ep} batch {batch_idx}; skipping optimizer step.")
                        restore_model_buffers(model, buffer_snapshot)
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                    optimizer.step()
                    if rollback_buffers and not model_buffers_are_finite(model):
                        restore_model_buffers(model, buffer_snapshot)
                        raise FloatingPointError(
                            f"Non-finite PTv3 state after optimizer step at epoch={ep} batch={batch_idx}."
                        )
                    scheduler.step()

                if not track_gradient_diagnostics:
                    gradient_stats = None

                batch_size = surf_data.size(0)
                train_sample_count += batch_size
                train_losses["loss"] += loss.item() * batch_size
                if gradient_stats is not None:
                    train_losses["gradient_norm_raw"] += gradient_stats["grad_norm"] * batch_size
                    train_losses["parameter_norm"] += gradient_stats["parameter_norm"] * batch_size
                    train_losses["gradient_max_abs"] += gradient_stats["max_grad"] * batch_size
                    train_losses["gradient_nonfinite"] += (0.0 if gradient_stats["finite"] else 1.0) * batch_size
                with torch.no_grad():
                    surface_loss = rel_l2_loss_fn(y_hat_surf.float(), surf_data.float()) if use_surface_supervision else torch.tensor(0.0, device=device)
                    volume_loss = rel_l2_loss_fn(y_hat_vol.float(), vol_data.float())
                    train_losses["rel_l2_surf"] += surface_loss.item() * batch_size
                    train_losses["rel_l2_vol"] += volume_loss.item() * batch_size
                    train_losses["rel_l2"] += (surface_loss + volume_loss).item() * batch_size
                    if use_surface_supervision:
                        pred_surf_train = y_hat_surf[..., :] * std_surf + mean_surf
                        gt_surf_train = surf_data * std_surf + mean_surf
                        accumulate_channel_metrics(train_losses, "rel_l2_surf", pred_surf_train, gt_surf_train, fields["surface"], rel_l2_loss_fn, batch_size)
                    pred_vol_train = y_hat_vol[..., :] * std_vol + mean_vol
                    gt_vol_train = vol_data * std_vol + mean_vol
                    accumulate_channel_metrics(train_losses, "rel_l2_vol", pred_vol_train, gt_vol_train, fields["volume"], rel_l2_loss_fn, batch_size)

                global_step += 1
                if run is not None and (batch_idx % log_every_n_steps == 0 or batch_idx == len(train_loader) - 1):
                    batch_log = {
                        "train/batch_loss": loss.item(),
                        "train/batch_rel_l2": (surface_loss + volume_loss).item(),
                        "train/batch_rel_l2_surf": surface_loss.item(),
                        "train/batch_rel_l2_vol": volume_loss.item(),
                        "train/batch_amp_scale": float(scaler.get_scale()) if amp else 1.0,
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": ep,
                    }
                    if gradient_stats is not None:
                        batch_log.update(
                            {
                                "train/batch_gradient_norm_raw": gradient_stats["grad_norm"],
                                "train/batch_parameter_norm": gradient_stats["parameter_norm"],
                                "train/batch_gradient_max_abs": gradient_stats["max_grad"],
                                "train/batch_gradient_nonfinite": 0.0 if gradient_stats["finite"] else 1.0,
                            }
                        )
                    wandb.log(batch_log, step=global_step)
                    train_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            model.eval()
            test_sample_count = 0
            test_pbar = tqdm(
                test_batch_source,
                desc=f"Eval  {ep + 1}/{config.epochs}",
                leave=False,
                dynamic_ncols=True,
                disable=not is_main,
            )
            with torch.no_grad():
                for batch in test_pbar:
                    geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = _parse_batch(batch, params_dim)
                    geo_mesh = geo_mesh.to(device)
                    surf_mesh = surf_mesh.to(device)
                    surf_data = surf_data.to(device)
                    vol_mesh = vol_mesh.to(device)
                    vol_data = vol_data.to(device)
                    if params is not None:
                        params = params.to(device)
                    if geo_log_density is not None:
                        geo_log_density = geo_log_density.to(device)

                    if config.dataset == "NACA4":
                        surf_data = surf_data[..., :1]
                        vol_data = torch.cat([vol_data[..., :1], vol_data[..., 2:4]], dim=-1)

                    if amp:
                        with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                            if accepts_geo_log_density:
                                y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params, geo_log_density=geo_log_density)
                            else:
                                y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)
                    else:
                        if accepts_geo_log_density:
                            y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params, geo_log_density=geo_log_density)
                        else:
                            y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)

                    if use_surface_supervision:
                        pred_surf = y_hat_surf[..., :] * std_surf + mean_surf
                        gt_surf = surf_data * std_surf + mean_surf
                    pred_vol = y_hat_vol[..., :] * std_vol + mean_vol
                    gt_vol = vol_data * std_vol + mean_vol

                    batch_size = surf_data.size(0)
                    test_sample_count += batch_size
                    if use_surface_supervision:
                        batch_loss = combined_loss_fn(y_hat_surf.float(), y_hat_vol.float(), surf_data.float(), vol_data.float())
                        surface_rel_l2 = rel_l2_loss_fn(y_hat_surf.float(), surf_data.float())
                    else:
                        batch_loss = loss_fn(y_hat_vol.float(), vol_data.float())
                        surface_rel_l2 = torch.tensor(0.0, device=device)
                    test_losses["loss"] += batch_loss.item() * batch_size

                    volume_rel_l2 = rel_l2_loss_fn(y_hat_vol.float(), vol_data.float())
                    test_losses["rel_l2_surf"] += surface_rel_l2.item() * batch_size
                    test_losses["rel_l2_vol"] += volume_rel_l2.item() * batch_size
                    test_losses["rel_l2"] += (surface_rel_l2 + volume_rel_l2).item() * batch_size
                    if use_surface_supervision:
                        accumulate_channel_metrics(test_losses, "rel_l2_surf", pred_surf, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size)
                    accumulate_channel_metrics(test_losses, "rel_l2_vol", pred_vol, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size)
                    test_pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

            if dist_info["enabled"]:
                train_losses = average_distributed_metrics(train_losses, train_sample_count, device)
                test_losses = average_distributed_metrics(test_losses, test_sample_count, device)
            else:
                for loss_name in train_losses.keys():
                    train_losses[loss_name] /= len(train_loader.dataset)
                for loss_name in test_losses.keys():
                    test_losses[loss_name] /= len(test_loader.dataset)

            if is_main and test_losses["rel_l2"] < loss_test_min:
                loss_test_min = test_losses["rel_l2"]
                torch.save({
                    "epoch": ep,
                    "global_step": global_step,
                    "best_rel_l2": loss_test_min,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": test_losses["loss"],
                    "rel_l2_loss": test_losses["rel_l2"],
                    "surface_fields": fields["surface"],
                    "volume_fields": fields["volume"],
                    "metric_values": {k: v for k, v in test_losses.items() if k.startswith("rel_l2")},
                }, "checkpoints/" + model_checkpoint_name + "_best.pt")

            if is_main:
                torch.save({
                    "epoch": ep,
                    "global_step": global_step,
                    "best_rel_l2": loss_test_min,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": test_losses["loss"],
                    "rel_l2_loss": test_losses["rel_l2"],
                    "surface_fields": fields["surface"],
                    "volume_fields": fields["volume"],
                    "metric_values": {k: v for k, v in test_losses.items() if k.startswith("rel_l2")},
                }, "checkpoints/" + model_checkpoint_name + "_last.pt")

            t2 = default_timer()
            if is_main:
                print(
                    f"epoch: {ep}, t2-t1 (epoch time): {t2-t1:.5f}, "
                    f"train loss: {train_losses['loss']:.5f}, test loss: {test_losses['loss']:.5f}"
                )
                wandb_dict = {"lr": scheduler.get_last_lr()[0]}
                wandb_dict.update({f"train/{key}": value for key, value in train_losses.items()})
                wandb_dict.update({f"test/{key}": value for key, value in test_losses.items()})
                add_all_field_metrics(wandb_dict, "train", fields["surface"], fields["volume"], metric_values=train_losses)
                add_all_field_metrics(wandb_dict, "test", fields["surface"], fields["volume"], metric_values=test_losses)
                add_canonical_field_metrics(wandb_dict, "train", fields["surface"], fields["volume"], metric_values=train_losses)
                add_canonical_field_metrics(wandb_dict, "test", fields["surface"], fields["volume"], metric_values=test_losses)
                wandb_dict["meta/training_surface_signals"] = ",".join(fields["surface"])
                wandb_dict["meta/training_volume_signals"] = ",".join(fields["volume"])
                wandb.log(wandb_dict, step=global_step)
    finally:
        if run is not None:
            best_ckpt = os.path.join("checkpoints", model_checkpoint_name + "_best.pt")
            last_ckpt = os.path.join("checkpoints", model_checkpoint_name + "_last.pt")
            if os.path.isfile(best_ckpt) or os.path.isfile(last_ckpt):
                artifact = wandb.Artifact("model", type="model")
                if os.path.isfile(best_ckpt):
                    artifact.add_file(best_ckpt)
                if os.path.isfile(last_ckpt):
                    artifact.add_file(last_ckpt)
                run.log_artifact(artifact)
            run.finish()
        cleanup_distributed()

~~~

### SATLOSS6 two-view trainer


~~~python
from __future__ import annotations

import math
import os
import gc
from contextlib import contextmanager
from timeit import default_timer

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from data.datasets import get_dataset
from loss.losses import CombinedLoss
from utils.utils import (
    apply_naca4_auto_point_budget,
    count_model_params,
    get_model_checkpoint_name,
    get_optimizer_scheduler_loss,
    initialize_gpu,
    initialize_wandb,
    make_grad_scaler,
    print_point_budget,
    reset_scheduler_for_extension,
)

CANON_SURF_FIELDS = ["pressure", "normal_x", "normal_y"]
CANON_VOL_FIELDS = ["pressure", "sdf", "velocity_x", "velocity_y"]


def is_dist_enabled():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_enabled() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_enabled() else 1


def is_main_process():
    return get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, (DDP, DataParallel)) else model


def gradient_diagnostics(model):
    """Return finite gradient and parameter norms for training observability."""
    grad_sq = None
    param_sq = None
    max_grad = None
    finite = True
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        parameter_sq = parameter.detach().float().pow(2).sum()
        param_sq = parameter_sq if param_sq is None else param_sq + parameter_sq
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not torch.isfinite(gradient).all():
            finite = False
        gradient_sq = gradient.pow(2).sum()
        grad_sq = gradient_sq if grad_sq is None else grad_sq + gradient_sq
        gradient_max = gradient.abs().max()
        max_grad = gradient_max if max_grad is None else torch.maximum(max_grad, gradient_max)
    reference = next(model.parameters()).detach()
    zero = reference.new_zeros((), dtype=torch.float32)
    return {
        "grad_norm": torch.sqrt(grad_sq.clamp_min(0.0)) if grad_sq is not None else zero,
        "parameter_norm": torch.sqrt(param_sq.clamp_min(0.0)) if param_sq is not None else zero,
        "max_grad": max_grad if max_grad is not None else zero,
        "finite": finite,
    }


def snapshot_model_buffers(model):
    return {name: buffer.detach().clone() for name, buffer in unwrap_model(model).named_buffers()}


def restore_model_buffers(model, snapshot):
    if snapshot is None:
        return
    for name, value in unwrap_model(model).named_buffers():
        if name in snapshot:
            value.copy_(snapshot[name])


def model_buffers_are_finite(model):
    return all(bool(torch.isfinite(buffer).all().item()) for buffer in unwrap_model(model).buffers())


def _last_linear_params(module):
    last_linear = None
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            last_linear = submodule
    if last_linear is None:
        return []
    params = [last_linear.weight]
    if last_linear.bias is not None:
        params.append(last_linear.bias)
    return params


def resolve_gradnorm_reference_params(model):
    base_model = unwrap_model(model)
    for attr_name in ("mlp", "output_head", "head", "surface_decoder", "volume_decoder"):
        if hasattr(base_model, attr_name):
            params = _last_linear_params(getattr(base_model, attr_name))
            if params:
                # Use the last linear weight as the GradNorm reference layer.
                return (params[0],)

    trainable_params = [param for param in base_model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("Could not resolve GradNorm reference parameters for the model.")
    return (trainable_params[-1],)


def resolve_balance_reference_parameter(model):
    """Return a shared backbone parameter used by layer-local balancers.

    The prediction head is intentionally avoided here. It is not shared by
    the task branches in many operator models and its gradients can make a
    task balancer overreact directly at the output.
    """
    base_model = unwrap_model(model)
    for attr_name in ("encoder_blocks", "geometry_encoder_blocks", "geometry_encoder"):
        blocks = getattr(base_model, attr_name, None)
        if blocks is None:
            continue
        for block in reversed(list(blocks)):
            params = _last_linear_params(block)
            if params:
                return params[0]

    reference_params = resolve_gradnorm_reference_params(model)
    if len(reference_params) != 1:
        raise ValueError("Layer-local gradient balancing requires exactly one reference parameter.")
    return reference_params[0]


def config_layer_gradient(task_losses, reference_parameter, return_diagnostics=False):
    """Compute the official ConFIG direction for one reference parameter.

    The surrounding model gradient is still produced by the ordinary weighted
    objective. Only this selected layer is replaced by the conflict-free
    direction, which keeps the experiment layer-local rather than full-network.
    """
    from conflictfree.grad_operator import ConFIG_update

    gradients = []
    for loss in task_losses:
        gradient = torch.autograd.grad(
            loss,
            reference_parameter,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if gradient is None:
            gradient = torch.zeros_like(reference_parameter)
        gradients.append(gradient.float().reshape(-1))

    gradient_matrix = torch.stack(gradients, dim=0)
    finite = bool(torch.isfinite(gradient_matrix).all().item())
    nonzero = bool((gradient_matrix.norm(dim=1) > 1.0e-12).all().item())
    ordinary_direction = gradient_matrix.sum(dim=0)
    ordinary_norm = ordinary_direction.norm()
    used_fallback = not (finite and nonzero)
    if finite and nonzero:
        direction = ConFIG_update(gradient_matrix, use_least_square=True)
        if not bool(torch.isfinite(direction).all().item()):
            direction = gradient_matrix.sum(dim=0)
            used_fallback = True
    else:
        direction = ordinary_direction

    # ConFIG's projection-length rule can produce a very different magnitude
    # from the configured weighted sum. Keep its conflict-free direction, but
    # preserve the optimizer step scale of the fixed-sum baseline for this
    # layer.
    direction_norm = direction.norm()
    if bool(torch.isfinite(direction_norm).item()) and float(direction_norm.item()) > 1.0e-12:
        if bool(torch.isfinite(ordinary_norm).item()) and float(ordinary_norm.item()) > 1.0e-12:
            direction = direction * (ordinary_norm / direction_norm)
    else:
        direction = ordinary_direction
        used_fallback = True

    direction = direction.to(dtype=reference_parameter.dtype).view_as(reference_parameter)
    if not return_diagnostics:
        return direction
    final_norm = direction.float().norm()
    cosine = torch.zeros((), device=direction.device, dtype=torch.float32)
    if float(final_norm.item()) > 1.0e-12 and float(ordinary_norm.item()) > 1.0e-12:
        cosine = F.cosine_similarity(direction.float().reshape(1, -1), ordinary_direction.reshape(1, -1), dim=1)[0]
    scale = torch.ones((), device=direction.device, dtype=torch.float32)
    if float(direction_norm.item()) > 1.0e-12:
        scale = (ordinary_norm / direction_norm).float()
    diagnostics = {
        "reference_grad_norm": ordinary_norm.detach().float(),
        "direction_norm": final_norm.detach().float(),
        "direction_scale": scale.detach().float(),
        "direction_cosine": cosine.detach().float(),
        "used_fallback": torch.tensor(float(used_fallback), device=direction.device),
    }
    return direction, diagnostics


def _config_full_update(gradient_matrix):
    """Apply the standard non-momentum ConFIG operator to full gradients.

    For m loss gradients and P parameters, the paper's least-squares solve can
    be reduced from an m-by-P solve to the m-by-m Gram system
    (U U^T) a = 1, followed by U^T a. This is the same minimum-norm
    least-squares solution while keeping the expensive linear algebra in loss
    space rather than parameter space.
    """
    with torch.no_grad():
        had_nonfinite = not bool(torch.isfinite(gradient_matrix).all().item())
        if had_nonfinite:
            # Keep AMP enabled and skip only invalid gradient coordinates.
            # This is intentionally a loose fallback: finite task-gradient
            # directions still contribute, while one overflowed coordinate
            # cannot terminate a long run.
            torch.nan_to_num(gradient_matrix, nan=0.0, posinf=0.0, neginf=0.0, out=gradient_matrix)

        gradient_norms = gradient_matrix.float().norm(dim=1)
        unit_gradients = gradient_matrix.float()
        unit_gradients.div_(gradient_norms.unsqueeze(1).clamp_min(1.0e-12))
        zero_rows = gradient_norms <= 1.0e-12
        unit_gradients.masked_fill_(zero_rows.unsqueeze(1), 0.0)

        gram = unit_gradients @ unit_gradients.transpose(0, 1)
        equal_weights = torch.ones(gradient_matrix.shape[0], device=gradient_matrix.device, dtype=gradient_matrix.dtype)
        coefficients = torch.linalg.lstsq(gram, equal_weights).solution
        best_direction = unit_gradients.transpose(0, 1) @ coefficients
        best_norm = best_direction.norm()
        unit_direction = best_direction / best_norm.clamp_min(1.0e-12)
        unit_direction = torch.nan_to_num(unit_direction, nan=0.0, posinf=0.0, neginf=0.0)

        # ProjectionLength from the reference implementation:
        # |g_c| = sum_i <g_i, g_c / |g_c|>.
        projection_lengths = (unit_gradients @ unit_direction) * gradient_norms
        config_gradient = unit_direction * projection_lengths.sum()
        min_cosine = (unit_gradients @ unit_direction).min()

        diagnostics = {
            "mean_grad_norm": gradient_norms.mean().detach().float(),
            "direction_norm": config_gradient.norm().detach().float(),
            "min_cosine": min_cosine.detach().float(),
            "used_fallback": torch.tensor(float(had_nonfinite), device=gradient_matrix.device),
            "nonfinite_gradients": torch.tensor(float(had_nonfinite), device=gradient_matrix.device),
        }
        return config_gradient, diagnostics


def config_full_backward(
    task_losses,
    parameters,
    scaler=None,
    amp_enabled=False,
    vectorized=True,
    allow_sequential_fallback=True,
):
    """Compute full-network ConFIG gradients and assign them to parameters.

    Each task is differentiated separately through one retained forward graph,
    matching full ConFIG rather than the paper's alternating M-ConFIG variant.
    Only flattened detached gradient rows are retained; parameter gradients are
    assigned from the final ConFIG vector in place.
    """
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not parameters:
        raise ValueError("ConFIG_full requires at least one trainable parameter.")

    task_count = len(task_losses)
    stacked_losses = torch.stack(list(task_losses))
    amp_scale = 1.0
    if amp_enabled and scaler is not None and scaler.is_enabled():
        stacked_losses = scaler.scale(stacked_losses)
        amp_scale = float(scaler.get_scale())

    gradient_matrix = None
    if vectorized:
        try:
            gradients = torch.autograd.grad(
                stacked_losses,
                parameters,
                grad_outputs=torch.eye(
                    task_count,
                    device=stacked_losses.device,
                    dtype=stacked_losses.dtype,
                ),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
                is_grads_batched=True,
            )
            gradient_parts = []
            for parameter, gradient in zip(parameters, gradients):
                if gradient is None:
                    gradient_parts.append(
                        torch.zeros((task_count, parameter.numel()), device=parameter.device, dtype=torch.float32)
                    )
                else:
                    gradient_parts.append(gradient.detach().float().reshape(task_count, -1))
            gradient_matrix = torch.cat(gradient_parts, dim=1)
            del gradients, gradient_parts
        except RuntimeError as exc:
            if not allow_sequential_fallback:
                raise RuntimeError(
                    "Vectorized ConFIG_full gradients failed. Set "
                    "experiment.config_full_vectorized_gradients=False to use the sequential path."
                ) from exc
            print(f"[ConFIG_full] Vectorized gradients unavailable; using sequential fallback: {exc}")

    if gradient_matrix is None:
        gradient_rows = []
        for task_index, task_loss in enumerate(task_losses):
            scaled_loss = task_loss
            if amp_enabled and scaler is not None and scaler.is_enabled():
                scaled_loss = scaler.scale(task_loss)
            gradients = torch.autograd.grad(
                scaled_loss,
                parameters,
                retain_graph=task_index < len(task_losses) - 1,
                create_graph=False,
                allow_unused=True,
            )
            row_parts = []
            for parameter, gradient in zip(parameters, gradients):
                if gradient is None:
                    row_parts.append(torch.zeros(parameter.numel(), device=parameter.device, dtype=torch.float32))
                else:
                    row_parts.append(gradient.detach().float().reshape(-1))
            gradient_rows.append(torch.cat(row_parts, dim=0))
            del gradients, row_parts, scaled_loss
        gradient_matrix = torch.stack(gradient_rows, dim=0)
        del gradient_rows

    del stacked_losses
    config_gradient, diagnostics = _config_full_update(gradient_matrix)
    if amp_scale != 1.0:
        # The assigned parameter gradients stay AMP-scaled for the normal
        # scaler.unscale_/step path; expose raw norms in W&B diagnostics.
        diagnostics["mean_grad_norm"] = diagnostics["mean_grad_norm"] / amp_scale
        diagnostics["direction_norm"] = diagnostics["direction_norm"] / amp_scale
    diagnostics["amp_scale"] = torch.tensor(amp_scale, device=parameters[0].device, dtype=torch.float32)

    offset = 0
    for parameter in parameters:
        parameter_size = parameter.numel()
        parameter.grad = config_gradient[offset:offset + parameter_size].view_as(parameter).to(dtype=parameter.dtype)
        offset += parameter_size

    return diagnostics


def external_gradnorm_backward(
    external_gradnorm,
    losses,
    reference_parameter,
    weighted_total_loss,
    min_loss_weights=None,
    scaler=None,
    amp_enabled=False,
):
    """Apply GradNorm weight updates and the weighted model backward separately.

    The upstream helper backpropagates its auxiliary GradNorm loss through the
    model as well as through its temporary loss-weight parameters. That adds a
    second-order model update to the task objective. We retain its state and
    update rule, but restrict the GradNorm derivative to the temporary weights;
    the model receives only the configured weighted task gradient.
    """
    losses = torch.stack(list(losses))
    if external_gradnorm.initted.device != losses.device:
        external_gradnorm.to(losses.device)

    step = external_gradnorm.step.item()
    external_gradnorm.step.add_(int(external_gradnorm.training))
    weighted_total_loss = weighted_total_loss.float()

    should_update = (
        external_gradnorm.training
        and not external_gradnorm.frozen
        and step >= external_gradnorm.update_after_step
        and (step % external_gradnorm.update_every) == 0
        and bool(external_gradnorm.loss_mask.any().item())
    )
    if not should_update:
        model_loss = weighted_total_loss
        if amp_enabled and scaler is not None and scaler.is_enabled():
            model_loss = scaler.scale(model_loss)
        model_loss.backward()
        return torch.zeros((), device=losses.device, dtype=torch.float32)

    loss_mask = external_gradnorm.loss_mask.to(device=losses.device, dtype=torch.bool)
    if external_gradnorm.has_restoring_force:
        if not bool(external_gradnorm.initted.item()):
            initial_losses = losses.detach().clone()
            if is_dist_enabled():
                dist.all_reduce(initial_losses)
                initial_losses.div_(get_world_size())
            external_gradnorm.initial_losses.copy_(initial_losses)
            external_gradnorm.initted.fill_(True)
        elif external_gradnorm.initial_losses_decay < 1.0:
            meaned_losses = losses.detach().clone()
            if is_dist_enabled():
                dist.all_reduce(meaned_losses)
                meaned_losses.div_(get_world_size())
            external_gradnorm.initial_losses.lerp_(meaned_losses, 1.0 - external_gradnorm.initial_losses_decay)

    selected_losses = losses[loss_mask]
    selected_weights = nn.Parameter(external_gradnorm.loss_weights[loss_mask].detach().clone())
    grad_norms = []
    for weight, loss in zip(selected_weights, selected_losses):
        gradient = torch.autograd.grad(
            weight * loss,
            reference_parameter,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is None:
            gradient = torch.zeros_like(reference_parameter)
        grad_norms.append(gradient.float().norm(p=2))
    grad_norms = torch.stack(grad_norms)
    grad_norm_average = grad_norms.mean()
    if is_dist_enabled():
        dist.all_reduce(grad_norm_average)
        grad_norm_average.div_(get_world_size())

    if external_gradnorm.has_restoring_force:
        loss_ratio = selected_losses.detach() / external_gradnorm.initial_losses[loss_mask].clamp_min(1.0e-8)
        relative_training_rate = F.normalize(loss_ratio, p=1, dim=0) * selected_losses.numel()
        gradient_target = (grad_norm_average * relative_training_rate.pow(-external_gradnorm.alpha)).detach()
    else:
        gradient_target = grad_norm_average.expand_as(grad_norms).detach()

    gradnorm_loss = F.l1_loss(grad_norms, gradient_target)
    weight_gradient = torch.autograd.grad(
        gradnorm_loss,
        selected_weights,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if weight_gradient is None:
        weight_gradient = torch.zeros_like(selected_weights)

    # The weighted loss keeps a detached view of the current weight buffer.
    # Backpropagate it before updating that buffer in place.
    model_loss = weighted_total_loss
    if amp_enabled and scaler is not None and scaler.is_enabled():
        model_loss = scaler.scale(model_loss)
    model_loss.backward()

    with torch.no_grad():
        updated_weights = selected_weights.detach() - weight_gradient.detach() * float(external_gradnorm.learning_rate)
        total_weight = external_gradnorm.init_loss_weights_for_sum[loss_mask].sum()
        if min_loss_weights is None:
            renormalized_weights = F.normalize(updated_weights, p=1, dim=0) * total_weight
        else:
            floors = torch.as_tensor(
                min_loss_weights,
                device=updated_weights.device,
                dtype=updated_weights.dtype,
            )[loss_mask]
            floors = floors.clamp_min(0.0)
            if bool((floors.sum() >= total_weight).item()):
                raise ValueError("external_gradnorm_min_weights must sum to less than the GradNorm weight total.")
            remaining_weight = total_weight - floors.sum()
            free_weights = (updated_weights.clamp_min(0.0) - floors).clamp_min(0.0)
            free_sum = free_weights.sum()
            if bool((free_sum > 1.0e-12).item()):
                renormalized_weights = floors + remaining_weight * free_weights / free_sum
            else:
                renormalized_weights = floors + remaining_weight / float(floors.numel())
        external_gradnorm.loss_weights[loss_mask] = renormalized_weights
        external_gradnorm.loss_weights_grad[loss_mask] = 0.0

    return gradnorm_loss.detach().float()


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
        }

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return {
        "enabled": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }


def cleanup_distributed():
    """Release CUDA-side training resources before shutting down NCCL.

    DDP can finish the final epoch and checkpoint successfully while a
    prefetched batch, CUDA stream, or persistent worker still owns cached
    allocations.  Letting NCCL tear down against that pressure can turn a
    successful run into an OOM during process exit.  Cleanup failures from
    this finalization path are non-training errors and are only ignored when
    CUDA/NCCL reports the known shutdown-time failure.
    """
    if not is_dist_enabled():
        return

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except RuntimeError as exc:
            if is_main_process():
                print(f"[cleanup] CUDA synchronize warning before NCCL shutdown: {exc}")
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if is_main_process():
                print(f"[cleanup] CUDA cache release warning before NCCL shutdown: {exc}")

    try:
        dist.destroy_process_group()
    except RuntimeError as exc:
        message = str(exc).lower()
        if not any(token in message for token in ("nccl", "cuda", "out of memory")):
            raise
        if is_main_process():
            print(f"[cleanup] NCCL shutdown warning after training completed: {exc}")


def distributed_average_scalars(values):
    if not is_dist_enabled():
        return values
    tensor = torch.tensor(values, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(get_world_size())
    return tensor.cpu().tolist()


def init_metric_dict(surface_fields, volume_fields, extra_keys=None):
    metrics = {
        "loss": 0.0,
        "rel_l2": 0.0,
        "rel_l2_surf": 0.0,
        "rel_l2_vol": 0.0,
    }
    for field_name in surface_fields:
        metrics[f"rel_l2_surf_{field_name}"] = 0.0
    for field_name in volume_fields:
        metrics[f"rel_l2_vol_{field_name}"] = 0.0
    for key in extra_keys or []:
        metrics[key] = 0.0
    return metrics


def init_metric_tensor_dict(surface_fields, volume_fields, device, extra_keys=None):
    metrics = {
        "loss": torch.zeros((), device=device, dtype=torch.float32),
        "rel_l2": torch.zeros((), device=device, dtype=torch.float32),
        "rel_l2_surf": torch.zeros((), device=device, dtype=torch.float32),
        "rel_l2_vol": torch.zeros((), device=device, dtype=torch.float32),
    }
    for field_name in surface_fields:
        metrics[f"rel_l2_surf_{field_name}"] = torch.zeros((), device=device, dtype=torch.float32)
    for field_name in volume_fields:
        metrics[f"rel_l2_vol_{field_name}"] = torch.zeros((), device=device, dtype=torch.float32)
    for key in extra_keys or []:
        metrics[key] = torch.zeros((), device=device, dtype=torch.float32)
    return metrics


def accumulate_channel_metrics(metrics, prefix, pred, gt, field_names, rel_l2_loss_fn, batch_size, metric_weight=1.0):
    for channel_idx, field_name in enumerate(field_names):
        channel_loss = rel_l2_loss_fn(pred[..., channel_idx:channel_idx + 1], gt[..., channel_idx:channel_idx + 1])
        metrics[f"{prefix}_{field_name}"] += channel_loss.item() * batch_size * metric_weight


def accumulate_channel_metrics_tensor(metrics, prefix, pred, gt, field_names, rel_l2_loss_fn, batch_size, metric_weight=1.0):
    for channel_idx, field_name in enumerate(field_names):
        channel_loss = rel_l2_loss_fn(pred[..., channel_idx:channel_idx + 1], gt[..., channel_idx:channel_idx + 1]).detach()
        metrics[f"{prefix}_{field_name}"] += channel_loss * float(batch_size) * float(metric_weight)


def add_canonical_field_metrics(wandb_dict, split, surface_fields, volume_fields, metric_values=None):
    metric_values = metric_values or {}
    for f in CANON_SURF_FIELDS:
        key = f"{split}/rel_l2_surf_{f}"
        src_key = f"rel_l2_surf_{f}"
        wandb_dict[key] = metric_values.get(src_key, np.nan) if f in surface_fields else np.nan
    for f in CANON_VOL_FIELDS:
        key = f"{split}/rel_l2_vol_{f}"
        src_key = f"rel_l2_vol_{f}"
        wandb_dict[key] = metric_values.get(src_key, np.nan) if f in volume_fields else np.nan


def add_all_field_metrics(wandb_dict, split, surface_fields, volume_fields, metric_values=None):
    metric_values = metric_values or {}
    for f in surface_fields:
        src_key = f"rel_l2_surf_{f}"
        wandb_dict[f"{split}/rel_l2_surf_{f}"] = metric_values.get(src_key, np.nan)
    for f in volume_fields:
        src_key = f"rel_l2_vol_{f}"
        wandb_dict[f"{split}/rel_l2_vol_{f}"] = metric_values.get(src_key, np.nan)


def set_dataset_epoch(dataset, epoch):
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def load_partial_state_dict(model, checkpoint_path, device):
    if not checkpoint_path:
        return 0, 0
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = checkpoint.get("model_state_dict", checkpoint)
    target = model.state_dict()
    filtered = {}
    matched = 0
    skipped = 0
    for key, value in source.items():
        if key in target and target[key].shape == value.shape:
            filtered[key] = value
            matched += 1
        else:
            skipped += 1
    target.update(filtered)
    model.load_state_dict(target, strict=False)
    print(f"[resume] Loaded {matched} tensors from {checkpoint_path}; skipped {skipped} incompatible tensors.")
    return matched, skipped


def load_full_training_state(
    model,
    optimizer,
    scheduler,
    scaler,
    checkpoint_path,
    device,
    load_scaler=True,
    uncertainty_balancer=None,
    external_gradnorm=None,
    reset_scheduler=False,
    steps_per_epoch=None,
    target_epochs=None,
):
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be provided for full-state resume.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = checkpoint.get("model_state_dict")
    if source is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain model_state_dict for full-state resume.")

    model.load_state_dict(source, strict=True)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if optimizer_state is None or scheduler_state is None:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing optimizer/scheduler state required for full-state resume."
        )
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if load_scaler and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    if uncertainty_balancer is not None:
        uncertainty_state = checkpoint.get("task_weighting_state_dict")
        if uncertainty_state is None:
            uncertainty_state = checkpoint.get("uncertainty_balancer_state_dict")
        if uncertainty_state is None:
            raise KeyError(
                f"Checkpoint {checkpoint_path} is missing task_weighting_state_dict required for full-state resume."
            )
        uncertainty_balancer.load_state_dict(uncertainty_state, strict=True)
    if external_gradnorm is not None:
        external_gradnorm_state = checkpoint.get("external_gradnorm_state_dict")
        if external_gradnorm_state is None:
            raise KeyError(
                f"Checkpoint {checkpoint_path} is missing external_gradnorm_state_dict required for full-state resume."
            )
        external_gradnorm.load_state_dict(external_gradnorm_state, strict=True)

    resumed_epoch = int(checkpoint.get("epoch", -1))
    start_epoch = resumed_epoch + 1
    if reset_scheduler:
        if steps_per_epoch is None or target_epochs is None:
            raise ValueError("steps_per_epoch and target_epochs are required to reset a resumed scheduler.")
        extension_epochs = max(int(target_epochs) - start_epoch, 1)
        reset_scheduler_for_extension(scheduler, optimizer, extension_epochs * int(steps_per_epoch))
        if is_main_process():
            print(f"[resume] Reset cosine scheduler for {extension_epochs} extension epochs.")
    global_step = int(checkpoint.get("global_step", 0))
    best_robust_rel_l2 = float(checkpoint.get("best_robust_rel_l2", np.inf))
    print(
        f"[resume] Restored full training state from {checkpoint_path}: "
        f"epoch={resumed_epoch}, next_epoch={start_epoch}, global_step={global_step}, "
        f"best_robust_rel_l2={best_robust_rel_l2:.6g}"
    )
    return start_epoch, global_step, best_robust_rel_l2


def build_training_checkpoint(
    *,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    loss,
    rel_l2_loss,
    surface_fields,
    volume_fields,
    global_step,
    best_robust_rel_l2,
    extra_metrics=None,
):
    checkpoint = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_robust_rel_l2": float(best_robust_rel_l2),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": float(loss),
        "rel_l2_loss": float(rel_l2_loss),
        "surface_fields": surface_fields,
        "volume_fields": volume_fields,
    }
    if extra_metrics:
        checkpoint.update(extra_metrics)
    return checkpoint


def unpack_batch(batch, params_dim):
    geo_log_density = None
    if params_dim > 0:
        if len(batch) == 7:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = batch
        elif len(batch) == 6:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params = batch
        else:
            raise ValueError(f"Unexpected parameterized batch size: {len(batch)}")
    else:
        params = None
        if len(batch) == 6:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, geo_log_density = batch
        elif len(batch) == 5:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
    return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density


def gather_points(points, idx):
    return torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, points.shape[-1]))


def gather_scalar(points, idx):
    return torch.gather(points, 1, idx)


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def move_batch_to_device(batch, device, keep_cpu_indices=()):
    """Move a batch while retaining selected top-level values on the host."""
    if not isinstance(batch, (list, tuple)) or not keep_cpu_indices:
        return move_to_device(batch, device)
    keep_cpu_indices = set(int(index) for index in keep_cpu_indices)
    moved = [
        value if index in keep_cpu_indices else move_to_device(value, device)
        for index, value in enumerate(batch)
    ]
    return type(batch)(moved)


def record_batch_stream(value, stream):
    if torch.is_tensor(value):
        if value.is_cuda:
            value.record_stream(stream)
        return
    if isinstance(value, dict):
        for item in value.values():
            record_batch_stream(item, stream)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            record_batch_stream(item, stream)


class CudaPrefetchLoader:
    def __init__(self, loader, device, keep_cpu_indices=()):
        self.loader = loader
        self.device = device
        self.keep_cpu_indices = tuple(keep_cpu_indices)
        self.enabled = device.type == "cuda" and torch.cuda.is_available()
        self.stream = torch.cuda.Stream(device=device) if self.enabled else None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if not self.enabled:
            yield from self.loader
            return

        loader_iter = iter(self.loader)
        next_batch = None

        def preload():
            nonlocal next_batch
            try:
                batch = next(loader_iter)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(self.stream):
                next_batch = move_batch_to_device(batch, self.device, self.keep_cpu_indices)

        preload()
        while next_batch is not None:
            torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
            batch = next_batch
            record_batch_stream(batch, torch.cuda.current_stream(device=self.device))
            preload()
            yield batch


def _cpu_generator(seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def _resolve_sampling_mode(mode, mixed_inverse_density_prob, generator):
    mode = str(mode)
    if mode != "mixed":
        return mode
    draw = torch.rand((), generator=generator).item()
    return "inverse_density_wor" if draw < float(mixed_inverse_density_prob) else "uniform_wor"


def sample_shared_shift_family(family_probabilities, generator):
    """Choose one sampling family for both views in a training batch."""
    probabilities = torch.as_tensor(list(family_probabilities), dtype=torch.float32)
    if probabilities.numel() != 3 or bool((probabilities < 0.0).any().item()):
        raise ValueError("shared shift family probabilities must contain three non-negative values.")
    total = float(probabilities.sum().item())
    if total <= 0.0:
        raise ValueError("shared shift family probabilities must have a positive sum.")
    draw = float(torch.rand((), generator=generator).item()) * total
    cumulative = 0.0
    for family_id, probability in enumerate(probabilities.tolist()):
        cumulative += float(probability)
        if draw < cumulative or family_id == 2:
            return family_id
    return 2


def sample_uniform_beta(beta_min, beta_max, generator):
    beta_min = float(beta_min)
    beta_max = float(beta_max)
    if beta_max < beta_min:
        beta_min, beta_max = beta_max, beta_min
    if abs(beta_max - beta_min) < 1e-12:
        return beta_min
    return float(torch.empty((), dtype=torch.float32).uniform_(beta_min, beta_max, generator=generator).item())


def _sample_gaussian_ball_mask_indices(
    geo_row,
    n_points,
    num_points,
    generator,
    std_fraction,
    prob_at_1sigma,
    min_survivors,
):
    if geo_row is None:
        raise RuntimeError("gaussian_ball_mask sampling requires geometry coordinates.")

    if num_points <= 0 or num_points >= n_points:
        base_idx = torch.arange(n_points, dtype=torch.long)
    else:
        base_idx = torch.randperm(n_points, generator=generator)[:num_points].to(dtype=torch.long)

    base_idx_for_geo = base_idx.to(device=geo_row.device, dtype=torch.long)
    geo_subset = geo_row.index_select(0, base_idx_for_geo).detach().float().cpu()
    if geo_subset.shape[0] == 0:
        raise RuntimeError("gaussian_ball_mask sampling produced an empty base subset.")

    center_rel = int(torch.randint(0, int(geo_subset.shape[0]), (1,), generator=generator, dtype=torch.long).item())
    center = geo_subset[center_rel]
    extent = (geo_subset.max(dim=0).values - geo_subset.min(dim=0).values).amax().item()
    sigma = max(float(std_fraction) * float(extent), 1.0e-8)
    prob_at_1sigma = min(max(float(prob_at_1sigma), 1.0e-8), 0.999999)
    gaussian_coeff = -math.log(prob_at_1sigma)
    dist = torch.linalg.vector_norm(geo_subset - center.unsqueeze(0), dim=1)
    remove_prob = torch.exp(-gaussian_coeff * (dist / sigma).pow(2)).clamp_(0.0, 1.0)
    keep_mask = torch.rand((geo_subset.shape[0],), generator=generator, dtype=torch.float32) >= remove_prob

    min_survivors = max(1, min(int(min_survivors), int(base_idx.shape[0])))
    if int(keep_mask.sum().item()) < min_survivors:
        keep_scores = 1.0 - remove_prob
        keep_rel = torch.topk(keep_scores, k=min_survivors, largest=True).indices
        keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
        keep_mask[keep_rel] = True

    return base_idx[keep_mask].to(dtype=torch.long), "gaussian_ball_mask"


def _sample_box_mask_indices(
    geo_row,
    n_points,
    num_points,
    generator,
    std_fraction_of_largest_extent,
):
    """Sample a base view and remove a 2-sigma box around a random center."""
    if geo_row is None:
        raise RuntimeError("box_mask sampling requires geometry coordinates.")

    if num_points <= 0 or num_points >= n_points:
        base_idx = torch.arange(n_points, dtype=torch.long)
    else:
        base_idx = torch.randperm(n_points, generator=generator)[:num_points].to(dtype=torch.long)

    base_idx_for_geo = base_idx.to(device=geo_row.device, dtype=torch.long)
    geo_subset = geo_row.index_select(0, base_idx_for_geo).detach().float().cpu()
    if geo_subset.shape[0] == 0:
        raise RuntimeError("box_mask sampling produced an empty base subset.")

    center_rel = int(torch.randint(0, int(geo_subset.shape[0]), (1,), generator=generator, dtype=torch.long).item())
    center = geo_subset[center_rel]
    extent = (geo_subset.max(dim=0).values - geo_subset.min(dim=0).values).amax().item()
    sigma = max(float(std_fraction_of_largest_extent) * float(extent), 1.0e-8)
    side_length = 2.0 * sigma
    half_side = 0.5 * side_length
    remove_mask = (torch.abs(geo_subset - center.unsqueeze(0)) <= half_side).all(dim=1)
    keep_mask = ~remove_mask

    if not bool(keep_mask.any()):
        raise RuntimeError("box_mask sampling removed every point from the secondary view.")

    return base_idx[keep_mask].to(dtype=torch.long), "box_mask"


def _sample_single_view_indices(
    geo_row,
    log_density_row,
    n_points,
    num_points,
    mode,
    inverse_density_beta,
    mixed_inverse_density_prob,
    generator,
    gaussian_mask_std_fraction=0.05,
    gaussian_mask_prob_at_1sigma=0.33,
    gaussian_mask_min_survivors=16384,
    sinusoidal_axis=None,
    sinusoidal_mix_fraction=0.0,
):
    resolved_mode = _resolve_sampling_mode(mode, mixed_inverse_density_prob, generator)

    if num_points <= 0:
        return torch.arange(n_points, dtype=torch.long), resolved_mode

    if resolved_mode == "uniform_wor":
        if num_points >= n_points:
            return torch.arange(n_points, dtype=torch.long), resolved_mode
        return torch.randperm(n_points, generator=generator)[:num_points].to(dtype=torch.long), resolved_mode

    if resolved_mode == "uniform_wr":
        return torch.randint(0, n_points, (num_points,), generator=generator, dtype=torch.long), resolved_mode

    if resolved_mode in {"inverse_density_wor", "inverse_density_wr"}:
        if log_density_row is None:
            raise RuntimeError(
                f"Sampling mode {resolved_mode!r} requires geometry log density, but none was provided."
            )
        replacement = resolved_mode.endswith("_wr")
        if not replacement and num_points >= n_points:
            return torch.arange(n_points, dtype=torch.long), resolved_mode

        log_weights = (-float(inverse_density_beta) * log_density_row.float()).cpu()
        log_weights = log_weights - torch.max(log_weights)
        weights = torch.exp(log_weights).clamp_min(1e-12)
        if not torch.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            weights = torch.ones_like(weights)
        idx = torch.multinomial(weights, num_samples=num_points, replacement=replacement, generator=generator)
        return idx.to(dtype=torch.long), resolved_mode

    if resolved_mode == "sinusoidal_axis_mixture_wor":
        axis = int(sinusoidal_axis)
        if axis not in (0, 1):
            raise ValueError(f"sinusoidal_axis must be 0 or 1, got {sinusoidal_axis!r}")
        if num_points >= n_points:
            return torch.arange(n_points, dtype=torch.long), resolved_mode

        coordinates = geo_row[:, axis].detach().float()
        coordinate_min = coordinates.amin()
        coordinate_max = coordinates.amax()
        span = (coordinate_max - coordinate_min).clamp_min(1.0e-8)
        normalized = ((coordinates - coordinate_min) / span).clamp(0.0, 1.0)
        weights = (torch.sin(math.pi * normalized).pow(2) + 1.0e-6).cpu()
        mix_fraction = min(max(float(sinusoidal_mix_fraction), 0.0), 1.0)
        weighted_count = min(max(int(round(mix_fraction * num_points)), 0), num_points)
        uniform_count = num_points - weighted_count

        if weighted_count > 0:
            weighted_idx = torch.multinomial(
                weights,
                num_samples=weighted_count,
                replacement=False,
                generator=generator,
            )
            selected = torch.zeros(n_points, dtype=torch.bool)
            selected[weighted_idx] = True
        else:
            weighted_idx = torch.empty(0, dtype=torch.long)
            selected = torch.zeros(n_points, dtype=torch.bool)

        if uniform_count > 0:
            remaining = torch.nonzero(~selected, as_tuple=False).squeeze(-1)
            uniform_rel = torch.randperm(int(remaining.numel()), generator=generator)[:uniform_count]
            uniform_idx = remaining[uniform_rel]
        else:
            uniform_idx = torch.empty(0, dtype=torch.long)
        return torch.cat([weighted_idx, uniform_idx], dim=0).to(dtype=torch.long), resolved_mode

    if resolved_mode == "gaussian_ball_mask":
        return _sample_gaussian_ball_mask_indices(
            geo_row,
            n_points=n_points,
            num_points=num_points,
            generator=generator,
            std_fraction=gaussian_mask_std_fraction,
            prob_at_1sigma=gaussian_mask_prob_at_1sigma,
            min_survivors=gaussian_mask_min_survivors,
        )

    if resolved_mode == "box_mask":
        return _sample_box_mask_indices(
            geo_row,
            n_points=n_points,
            num_points=num_points,
            generator=generator,
            std_fraction_of_largest_extent=gaussian_mask_std_fraction,
        )

    raise ValueError(f"Unsupported sampling mode: {resolved_mode}")


def sample_geometry_view(
    geo_mesh,
    geo_log_density,
    num_points,
    mode,
    inverse_density_beta,
    mixed_inverse_density_prob,
    seed,
    gaussian_mask_std_fraction=0.05,
    gaussian_mask_prob_at_1sigma=0.33,
    gaussian_mask_min_survivors=16384,
    sinusoidal_axis=None,
    sinusoidal_mix_fraction=0.0,
    return_indices=False,
):
    if geo_log_density is None and _sampling_mode_requires_density(mode):
        raise RuntimeError(f"Sampling mode {mode!r} requires geometry log density for view sampling.")

    idx_rows = []
    resolved_modes = []
    batch_size = int(geo_mesh.shape[0])
    n_points = int(geo_mesh.shape[1])
    for batch_idx in range(batch_size):
        generator = _cpu_generator(seed + 1009 * batch_idx)
        geo_row = geo_mesh[batch_idx]
        log_density_row = None if geo_log_density is None else geo_log_density[batch_idx]
        idx_row, resolved_mode = _sample_single_view_indices(
            geo_row,
            log_density_row,
            n_points=n_points,
            num_points=num_points,
            mode=mode,
            inverse_density_beta=inverse_density_beta,
            mixed_inverse_density_prob=mixed_inverse_density_prob,
            generator=generator,
            gaussian_mask_std_fraction=gaussian_mask_std_fraction,
            gaussian_mask_prob_at_1sigma=gaussian_mask_prob_at_1sigma,
            gaussian_mask_min_survivors=gaussian_mask_min_survivors,
            sinusoidal_axis=sinusoidal_axis,
            sinusoidal_mix_fraction=sinusoidal_mix_fraction,
        )
        idx_rows.append(idx_row)
        resolved_modes.append(resolved_mode)

    row_lengths = [int(idx_row.numel()) for idx_row in idx_rows]
    if len(set(row_lengths)) > 1:
        common_num_points = max(1, min(row_lengths))
        trimmed_rows = []
        for batch_idx, idx_row in enumerate(idx_rows):
            if int(idx_row.numel()) == common_num_points:
                trimmed_rows.append(idx_row)
                continue
            trim_generator = _cpu_generator(seed + 200003 + 1009 * batch_idx)
            keep_rel = torch.randperm(int(idx_row.numel()), generator=trim_generator)[:common_num_points]
            trimmed_rows.append(idx_row[keep_rel].to(dtype=torch.long))
        idx_rows = trimmed_rows

    idx = torch.stack(idx_rows, dim=0)
    if geo_mesh.device.type != idx.device.type or geo_mesh.device != idx.device:
        idx = idx.to(device=geo_mesh.device, non_blocking=(geo_mesh.device.type == "cuda"))
    sampled_density = None if geo_log_density is None else gather_scalar(geo_log_density, idx)
    sampled_geometry = gather_points(geo_mesh, idx)
    if return_indices:
        return sampled_geometry, sampled_density, resolved_modes, idx
    return sampled_geometry, sampled_density, resolved_modes


def consistency_warmup_factor(epoch, warmup_epochs):
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch + 1) / float(warmup_epochs))


def prediction_consistency_smooth_l1_loss(
    y1_surf,
    y1_vol,
    y2_surf,
    y2_vol,
    beta=0.05,
    symmetric_detached=False,
    average_groups=False,
):
    if symmetric_detached:
        surf_loss = 0.5 * (
            F.smooth_l1_loss(y1_surf, y2_surf.detach(), beta=beta)
            + F.smooth_l1_loss(y2_surf, y1_surf.detach(), beta=beta)
        )
        vol_loss = 0.5 * (
            F.smooth_l1_loss(y1_vol, y2_vol.detach(), beta=beta)
            + F.smooth_l1_loss(y2_vol, y1_vol.detach(), beta=beta)
        )
    else:
        surf_target = (0.5 * (y1_surf.detach() + y2_surf.detach())).to(dtype=y1_surf.dtype)
        vol_target = (0.5 * (y1_vol.detach() + y2_vol.detach())).to(dtype=y1_vol.dtype)
        surf_loss = 0.5 * (
            F.smooth_l1_loss(y1_surf, surf_target, beta=beta)
            + F.smooth_l1_loss(y2_surf, surf_target, beta=beta)
        )
        vol_loss = 0.5 * (
            F.smooth_l1_loss(y1_vol, vol_target, beta=beta)
            + F.smooth_l1_loss(y2_vol, vol_target, beta=beta)
        )
    return 0.5 * (surf_loss + vol_loss) if average_groups else surf_loss + vol_loss


def soft_worst_case_loss(loss_a, loss_b, tau=0.1):
    tau = max(float(tau), 1e-6)
    pair = torch.stack([loss_a.float(), loss_b.float()], dim=0)
    return torch.logsumexp(pair / tau, dim=0) * tau


def latent_consistency_loss(latent_teacher, latent_student):
    latent_teacher = F.layer_norm(latent_teacher.detach().float(), (latent_teacher.shape[-1],))
    latent_student = F.layer_norm(latent_student.float(), (latent_student.shape[-1],))
    return F.mse_loss(latent_student, latent_teacher)


class LearnedTaskWeighting(nn.Module):
    def __init__(
        self,
        task_names,
        init_logits=None,
        min_logit=-4.0,
        max_logit=4.0,
        min_weights=None,
        base_weights=None,
        warmup_epochs=0,
    ):
        super().__init__()
        self.task_names = tuple(task_names)
        if init_logits is None:
            init_logits = [0.0] * len(self.task_names)
        if len(init_logits) != len(self.task_names):
            raise ValueError(f"Expected {len(self.task_names)} init logits, got {len(init_logits)}.")
        self.logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))
        self.min_logit = float(min_logit)
        self.max_logit = float(max_logit)
        if min_weights is None:
            min_weights = [0.0] * len(self.task_names)
        if base_weights is None:
            base_weights = [1.0 / len(self.task_names)] * len(self.task_names)
        if len(min_weights) != len(self.task_names):
            raise ValueError(f"Expected {len(self.task_names)} min weights, got {len(min_weights)}.")
        if len(base_weights) != len(self.task_names):
            raise ValueError(f"Expected {len(self.task_names)} base weights, got {len(base_weights)}.")
        min_weights_t = torch.tensor(min_weights, dtype=torch.float32)
        base_weights_t = torch.tensor(base_weights, dtype=torch.float32)
        if torch.any(min_weights_t < 0):
            raise ValueError("min_weights must be non-negative.")
        if float(min_weights_t.sum().item()) >= 1.0:
            raise ValueError("Sum of min_weights must be < 1.")
        if torch.any(base_weights_t < 0):
            raise ValueError("base_weights must be non-negative.")
        if not torch.isclose(base_weights_t.sum(), torch.tensor(1.0, dtype=torch.float32), atol=1e-5):
            raise ValueError("base_weights must sum to 1.")
        self.register_buffer("min_weights", min_weights_t)
        self.register_buffer("base_weights", base_weights_t)
        self.warmup_epochs = int(warmup_epochs)

    def combine(self, losses, epoch_idx=None):
        if len(losses) != len(self.task_names):
            raise ValueError(f"Expected {len(self.task_names)} task losses, got {len(losses)}.")
        stacked_losses = torch.stack([loss.float() for loss in losses])
        clamped_logits = torch.clamp(self.logits, min=self.min_logit, max=self.max_logit)
        raw_weights = torch.softmax(clamped_logits, dim=0)
        free_mass = 1.0 - self.min_weights.sum()
        learned_weights = self.min_weights + free_mass * raw_weights
        if self.warmup_epochs > 0 and epoch_idx is not None:
            mix = min(1.0, float(epoch_idx + 1) / float(self.warmup_epochs))
            weights = (1.0 - mix) * self.base_weights + mix * learned_weights
        else:
            weights = learned_weights
        weighted_terms = weights * stacked_losses
        total_loss = weighted_terms.sum()
        return {
            "total_loss": total_loss,
            "weights": weights.detach(),
            "raw_weights": raw_weights.detach(),
            "logits": clamped_logits.detach(),
            "per_task_terms": weighted_terms.detach(),
        }


def move_optional_tensor(x, device):
    if x is None:
        return None
    return x.to(device, non_blocking=True)


def duplicate_batch_tensor(x):
    if x is None:
        return None
    return torch.cat([x, x], dim=0)


def _optional_positive_int(value):
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _sampling_mode_requires_density(mode):
    mode = str(mode)
    return mode == "mixed" or mode.startswith("inverse_density")


@contextmanager
def temporary_model_attr(model, attr_name, value):
    value = _optional_positive_int(value)
    if value is None:
        yield
        return

    base_model = unwrap_model(model)
    if not hasattr(base_model, attr_name):
        yield
        return

    old_value = getattr(base_model, attr_name)
    setattr(base_model, attr_name, value)
    try:
        yield
    finally:
        setattr(base_model, attr_name, old_value)


def forward_model_view(
    model,
    geo,
    surf_mesh,
    vol_mesh,
    params,
    *,
    model_requires_density,
    geo_log_density=None,
    return_latent=False,
    subsampled_geometry_points=None,
):
    with temporary_model_attr(model, "subsampled_geometry_points", subsampled_geometry_points):
        if model_requires_density:
            if return_latent:
                return model(
                    geo,
                    surf_mesh,
                    vol_mesh,
                    params,
                    geo_log_density=geo_log_density,
                    return_latent=True,
                )
            return model(
                geo,
                surf_mesh,
                vol_mesh,
                params,
                geo_log_density=geo_log_density,
            )
        if return_latent:
            return model(
                geo,
                surf_mesh,
                vol_mesh,
                params,
                return_latent=True,
            )
        return model(
            geo,
            surf_mesh,
            vol_mesh,
            params,
        )


def inference_model_view(
    model,
    geo,
    surf_mesh,
    vol_mesh,
    params,
    *,
    model_requires_density,
    geo_log_density=None,
    subsampled_geometry_points=None,
):
    with temporary_model_attr(model, "subsampled_geometry_points", subsampled_geometry_points):
        if model_requires_density:
            return model.inference(geo, surf_mesh, vol_mesh, params, geo_log_density=geo_log_density)
        return model.inference(geo, surf_mesh, vol_mesh, params)


def evaluate_loader(
    model,
    loader,
    config,
    device,
    dtype,
    amp,
    params_dim,
    mean_surf,
    std_surf,
    mean_vol,
    std_vol,
    fields,
    rel_l2_loss_fn,
    combined_loss_fn,
    loss_fn,
    use_surface_supervision,
    mode_name,
    num_view_points,
    eval_subsampled_geometry_points,
    fixed_seed_offset,
    model_requires_density,
    cuda_batch_prefetch,
    keep_cpu_indices=(),
):
    metrics = init_metric_dict(fields["surface"], fields["volume"])
    model.eval()
    sample_count = 0

    eval_loader = (
        CudaPrefetchLoader(loader, device, keep_cpu_indices=keep_cpu_indices)
        if cuda_batch_prefetch
        else loader
    )
    pbar = tqdm(eval_loader, desc=f"Eval {mode_name}", leave=False, dynamic_ncols=True, disable=not is_main_process())
    with torch.inference_mode():
        for batch_idx, batch in enumerate(pbar):
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = unpack_batch(batch, params_dim)

            view_geo, view_log_density, _ = sample_geometry_view(
                geo_mesh,
                geo_log_density,
                num_points=num_view_points,
                mode=mode_name,
                inverse_density_beta=float(getattr(config, "inverse_density_beta", 1.0)),
                mixed_inverse_density_prob=float(getattr(config, "mixed_inverse_density_prob", 0.5)),
                seed=int(config.random_seed + fixed_seed_offset + batch_idx * 10007),
            )

            params = move_optional_tensor(params, device)
            surf_mesh = surf_mesh.to(device, non_blocking=True)
            surf_data = surf_data.to(device, non_blocking=True)
            vol_mesh = vol_mesh.to(device, non_blocking=True)
            vol_data = vol_data.to(device, non_blocking=True)
            view_geo = view_geo.to(device, non_blocking=True)
            if model_requires_density:
                view_log_density = view_log_density.to(device, non_blocking=True)

            with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=amp):
                y_hat_surf, y_hat_vol = inference_model_view(
                    model,
                    view_geo,
                    surf_mesh,
                    vol_mesh,
                    params,
                    model_requires_density=model_requires_density,
                    geo_log_density=view_log_density if model_requires_density else None,
                    subsampled_geometry_points=eval_subsampled_geometry_points,
                )

            if use_surface_supervision:
                pred_surf = y_hat_surf * std_surf + mean_surf
                gt_surf = surf_data * std_surf + mean_surf
                batch_loss = combined_loss_fn(y_hat_surf, y_hat_vol, surf_data, vol_data)
                surface_rel_l2 = rel_l2_loss_fn(y_hat_surf, surf_data)
            else:
                pred_surf = None
                gt_surf = None
                batch_loss = loss_fn(y_hat_vol, vol_data)
                surface_rel_l2 = torch.tensor(0.0, device=device)

            pred_vol = y_hat_vol * std_vol + mean_vol
            gt_vol = vol_data * std_vol + mean_vol
            volume_rel_l2 = rel_l2_loss_fn(y_hat_vol, vol_data)

            batch_size = surf_data.size(0)
            sample_count += batch_size
            metrics["loss"] += batch_loss.item() * batch_size
            metrics["rel_l2_surf"] += surface_rel_l2.item() * batch_size
            metrics["rel_l2_vol"] += volume_rel_l2.item() * batch_size
            metrics["rel_l2"] += (surface_rel_l2 + volume_rel_l2).item() * batch_size

            if use_surface_supervision:
                accumulate_channel_metrics(metrics, "rel_l2_surf", pred_surf, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size)
            accumulate_channel_metrics(metrics, "rel_l2_vol", pred_vol, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size)

            pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

    if is_dist_enabled():
        key_list = list(metrics.keys())
        metric_tensor = torch.tensor(
            [metrics[key] for key in key_list] + [float(sample_count)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        sample_count = max(int(metric_tensor[-1].item()), 1)
        for idx, key in enumerate(key_list):
            metrics[key] = float(metric_tensor[idx].item())

    denom = max(int(sample_count), 1)
    for key in metrics.keys():
        metrics[key] /= denom
    return metrics


def run_consistency_training(cfg, model_ctor, model_requires_density):
    config = cfg.experiment
    wandb_config = cfg.wandb
    multi_gpu_strategy = str(getattr(config, "multi_gpu_strategy", "auto")).lower()
    if multi_gpu_strategy == "data_parallel" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "multi_gpu_strategy=data_parallel must be launched with plain python, not torchrun/DDP. "
            "Use CUDA_VISIBLE_DEVICES=1,2 python smart/train_satloss4.py ..."
        )

    dist_info = setup_distributed() if multi_gpu_strategy != "data_parallel" else {
        "enabled": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
    }
    run = initialize_wandb(config, wandb_config) if is_main_process() else None

    device = initialize_gpu(config.random_seed, high_precision=False)

    gradient_norm = config.gradient_norm
    track_gradient_diagnostics = bool(getattr(config, "track_gradient_diagnostics", False))
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(config.precision, torch.float16)
    amp = config.amp
    if is_main_process():
        print(gradient_norm, amp, dtype)

    train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)

    def apply_vanilla_smart_field_subset():
        nonlocal fields, surf_channels, vol_channels
        if config.dataset == "NACA4":
            fields = {"surface": ["pressure"], "volume": ["pressure", "velocity_x", "velocity_y"]}
            surf_channels = 1
            vol_channels = 3

    apply_vanilla_smart_field_subset()
    if is_main_process():
        print(f"[{config.model_name}] training signals -> surface: {fields['surface']} | volume: {fields['volume']}")

    point_info = apply_naca4_auto_point_budget(config, train_data, for_cat=False)
    if point_info is not None:
        if is_main_process():
            print_point_budget(config.model_name, point_info)
        train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
        apply_vanilla_smart_field_subset()
        if is_main_process():
            print(f"[{config.model_name}] training signals -> surface: {fields['surface']} | volume: {fields['volume']}")

    use_surface_supervision = len(fields["surface"]) > 0
    set_dataset_epoch(train_data, 0)
    set_dataset_epoch(test_data, 0)

    prefetch_factor = int(getattr(config, "prefetch_factor", 2))
    pin_memory = bool(getattr(config, "pin_memory", True))
    cuda_batch_prefetch = bool(getattr(config, "cuda_batch_prefetch", device.type == "cuda")) and device.type == "cuda"
    effective_num_workers = int(getattr(config, "num_workers", 0))
    dl_common = dict(batch_size=config.batch_size, num_workers=effective_num_workers, pin_memory=pin_memory)
    if effective_num_workers > 0:
        dl_common["prefetch_factor"] = prefetch_factor
        # AhmedMLDatasetV2 stores the current epoch in a shared multiprocessing
        # value, so persistent workers remain synchronized under DDP.
        dl_common["persistent_workers"] = True
    if is_main_process():
        print(
            f"[dataloader] world_size={dist_info['world_size']}, "
            f"num_workers_per_rank={effective_num_workers}, "
            f"prefetch_factor={dl_common.get('prefetch_factor', 'n/a')}, "
            f"persistent_workers={dl_common.get('persistent_workers', False)}, "
            f"cuda_batch_prefetch={cuda_batch_prefetch}"
        )

    keep_geometry_cpu_for_view_sampling = bool(
        getattr(config, "keep_geometry_cpu_for_view_sampling", False)
    )
    geometry_cpu_indices = (0, 6) if params_dim > 0 else (0, 5)

    train_sampler = DistributedSampler(train_data, shuffle=True) if dist_info["enabled"] else None
    train_loader = torch.utils.data.DataLoader(
        train_data,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **dl_common,
    )
    # Validation is also distributed.  Keeping the complete test set on rank
    # 0 makes SATLOSS8 appear to hang after an epoch because PTv3 evaluates
    # every held-out geometry, including the expensive local-query decoder,
    # while all other ranks wait at the barrier.  Each rank evaluates a
    # disjoint deterministic shard; evaluate_loader all-reduces the metrics.
    test_sampler = DistributedSampler(test_data, shuffle=False) if dist_info["enabled"] else None
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        sampler=test_sampler,
        **dl_common,
    )

    mean_surf = stats[0][:surf_channels].to(device)
    std_surf = stats[1][:surf_channels].to(device)
    if config.dataset == "NACA4" and vol_channels == 2:
        mean_vol = stats[2][2:4].to(device)
        std_vol = stats[3][2:4].to(device)
    elif config.dataset == "NACA4" and vol_channels == 3:
        mean_vol = torch.stack([stats[2][0], stats[2][2], stats[2][3]]).to(device)
        std_vol = torch.stack([stats[3][0], stats[3][2], stats[3][3]]).to(device)
    else:
        mean_vol = stats[2][:vol_channels].to(device)
        std_vol = stats[3][:vol_channels].to(device)

    model_kwargs = {
        "spatial_dim": spatial_dim,
        "surface_channels": surf_channels,
        "volume_channels": vol_channels,
        "parameter_channels": params_dim,
    }
    merged_kwargs = {**model_kwargs, **config.architecture} if "architecture" in config else model_kwargs
    if is_main_process():
        print(f"Model kwargs: {merged_kwargs}")
    model = model_ctor(**merged_kwargs).to(device)
    rollback_buffers = bool(getattr(model, "rollback_buffers_on_nonfinite", False))

    if is_main_process():
        print(f"Total parameters: {count_model_params(model)}")
    model_checkpoint_name = get_model_checkpoint_name(config)
    if is_main_process():
        print(f"Checkpoint name: {model_checkpoint_name}")
    if run is not None and bool(getattr(config, "wandb_watch_model", False)):
        run.watch(model, log="all")

    use_prediction_consistency = bool(getattr(config, "use_prediction_consistency", True))
    task_weighting_method = str(getattr(config, "task_weighting_method", "legacy")).strip().lower()
    use_config_layer = task_weighting_method in {"config", "config_layer", "config-layer"}
    use_config_full = task_weighting_method in {"config_full", "config-full", "configfull"}
    use_external_gradnorm = task_weighting_method in {"gradnorm_external", "external_gradnorm", "gradnorm"}
    use_fixed_sum = task_weighting_method in {"fixed_sum", "fixed-sum", "fixedsum"}
    use_learned_task_weighting = bool(
        getattr(config, "use_learned_task_weighting", getattr(config, "use_uncertainty_weighting", False))
    )
    if sum((use_config_layer, use_config_full, use_external_gradnorm, use_fixed_sum, use_learned_task_weighting)) > 1:
        raise ValueError(
            "Only one task weighting backend can be enabled: task_weighting_method=CONFIG, "
            "task_weighting_method=config_full, "
            "task_weighting_method=gradnorm_external, task_weighting_method=fixed_sum, "
            "or use_learned_task_weighting."
        )
    scaler = make_grad_scaler(config)
    uncertainty_balancer = None
    config_reference_parameter = None
    config_full_parameters = None
    config_task_weights = None
    external_gradnorm = None
    external_gradnorm_min_weights = None
    extra_optimizer_param_groups = []
    if use_learned_task_weighting:
        learned_task_names = ["supervised_mean", "supervised_worst"]
        if use_prediction_consistency:
            learned_task_names.append("prediction_consistency")
        uncertainty_balancer = LearnedTaskWeighting(
            task_names=tuple(learned_task_names),
            init_logits=list(getattr(config, "task_weight_init_logits", [0.0] * len(learned_task_names))),
            min_logit=float(getattr(config, "task_weight_logit_min", -4.0)),
            max_logit=float(getattr(config, "task_weight_logit_max", 4.0)),
            min_weights=list(getattr(config, "task_weight_min_weights", [0.4] * len(learned_task_names))),
            base_weights=list(getattr(config, "task_weight_base_weights", [1.0 / len(learned_task_names)] * len(learned_task_names))),
            warmup_epochs=int(getattr(config, "task_weight_warmup_epochs", 25)),
        ).to(device)
        extra_optimizer_param_groups.append(
            {
                "params": list(uncertainty_balancer.parameters()),
                "lr": float(getattr(config, "task_weight_lr", 1.0e-3)),
                "weight_decay": 0.0,
            }
        )
    if use_config_layer or use_config_full or use_external_gradnorm or use_fixed_sum:
        task_count = 3 if use_prediction_consistency else 2
        configured_task_weights = list(
            getattr(config, "config_task_base_weights", getattr(config, "task_weight_base_weights", []))
        )
        if not configured_task_weights:
            configured_task_weights = [1.0 / task_count] * task_count
        if not use_prediction_consistency and len(configured_task_weights) == 3:
            configured_task_weights = configured_task_weights[:2]
        if len(configured_task_weights) != task_count:
            raise ValueError(
                f"Expected {task_count} task weights for {task_weighting_method}, "
                f"got {len(configured_task_weights)}."
            )
        weight_sum = float(sum(float(weight) for weight in configured_task_weights))
        if weight_sum <= 0.0:
            raise ValueError(f"Task weights for {task_weighting_method} must have a positive sum.")
        # The no-consistency ablation may retain the first two SATLOSS view
        # weights (0.2, 0.2) while disabling the third task.  Do not force
        # those weights to sum to one when explicitly requested; all existing
        # configurations retain the historical normalization by default.
        normalize_fixed_sum_weights = bool(getattr(config, "fixed_sum_normalize_weights", True))
        if normalize_fixed_sum_weights or not use_fixed_sum:
            config_task_weights = [float(weight) / weight_sum for weight in configured_task_weights]
        else:
            config_task_weights = [float(weight) for weight in configured_task_weights]
        if use_config_layer or use_external_gradnorm:
            config_reference_parameter = resolve_balance_reference_parameter(model)
        if use_config_full:
            config_full_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if use_external_gradnorm:
            configured_min_weights = getattr(config, "external_gradnorm_min_weights", None)
            if configured_min_weights is not None:
                external_gradnorm_min_weights = [float(weight) for weight in configured_min_weights]
                if len(external_gradnorm_min_weights) != task_count:
                    raise ValueError(
                        f"Expected {task_count} external_gradnorm_min_weights, "
                        f"got {len(external_gradnorm_min_weights)}."
                    )
                if any(weight < 0.0 for weight in external_gradnorm_min_weights):
                    raise ValueError("external_gradnorm_min_weights must be non-negative.")
                if sum(external_gradnorm_min_weights) >= 1.0:
                    raise ValueError("external_gradnorm_min_weights must sum to less than 1.0.")
            try:
                from gradnorm_pytorch import GradNormLossWeighter
            except ImportError as exc:
                raise RuntimeError(
                    "task_weighting_method=gradnorm_external requires gradnorm-pytorch. "
                    "Install it with: python -m pip install gradnorm-pytorch"
                ) from exc
            external_gradnorm = GradNormLossWeighter(
                num_losses=task_count,
                loss_weights=config_task_weights,
                learning_rate=float(getattr(config, "external_gradnorm_lr", 1.0e-4)),
                restoring_force_alpha=float(getattr(config, "external_gradnorm_alpha", 0.0)),
                grad_norm_parameters=config_reference_parameter,
                initial_losses_decay=float(getattr(config, "external_gradnorm_initial_losses_decay", 1.0)),
                update_after_step=int(getattr(config, "external_gradnorm_update_after_step", 0)),
                update_every=int(getattr(config, "external_gradnorm_update_every", 1)),
            ).to(device)
    if is_main_process():
        active_backend = (
            task_weighting_method
            if use_config_layer or use_config_full or use_external_gradnorm
            else "uncertainty"
            if use_learned_task_weighting
            else "fixed_sum"
            if use_fixed_sum
            else "fixed"
        )
        print(f"[loss balancing] backend={active_backend}")
    optimizer, scheduler, loss_fn, rel_l2_loss_fn = get_optimizer_scheduler_loss(
        model,
        config,
        train_loader,
        loss_dim=1,
        extra_param_groups=extra_optimizer_param_groups,
    )
    combined_loss_fn = CombinedLoss(loss_fn, fields) if use_surface_supervision else None

    best_robust_rel_l2 = np.inf
    global_step = 0
    start_epoch = 0
    log_every_n_steps = getattr(config, "log_every_n_steps", 10)

    primary_view_points = int(getattr(config, "primary_view_geometry_points", getattr(config, "view_geometry_points", 0)))
    secondary_view_points = int(getattr(config, "secondary_view_geometry_points", getattr(config, "view_geometry_points", 0)))
    shared_shift_sampling_mode = str(getattr(config, "train_shared_shift_sampling_mode", "")).strip().lower()
    if shared_shift_sampling_mode in {"random_beta_sine", "random_beta_sine_axis", "shared_random"}:
        shared_shift_view_points = int(getattr(config, "shared_shift_view_geometry_points", 0))
        if shared_shift_view_points > 0:
            primary_view_points = shared_shift_view_points
            secondary_view_points = shared_shift_view_points
    eval_aligned_view_points = int(
        getattr(config, "eval_aligned_view_geometry_points", getattr(config, "eval_view_geometry_points", primary_view_points))
    )
    eval_shifted_view_points = int(
        getattr(config, "eval_shifted_view_geometry_points", getattr(config, "eval_view_geometry_points", secondary_view_points))
    )
    primary_view_subsampled_geometry_points = _optional_positive_int(
        getattr(config, "primary_view_subsampled_geometry_points", 0)
    )
    secondary_view_subsampled_geometry_points = _optional_positive_int(
        getattr(config, "secondary_view_subsampled_geometry_points", 0)
    )
    eval_aligned_subsampled_geometry_points = _optional_positive_int(
        getattr(config, "eval_aligned_subsampled_geometry_points", primary_view_subsampled_geometry_points or 0)
    )
    eval_shifted_subsampled_geometry_points = _optional_positive_int(
        getattr(config, "eval_shifted_subsampled_geometry_points", secondary_view_subsampled_geometry_points or 0)
    )

    init_ckpt = str(getattr(config, "init_ckpt", "")).strip()
    resume_ckpt = str(getattr(config, "resume_ckpt", "")).strip()
    resume_full_state = bool(getattr(config, "resume_full_state", False))
    if resume_full_state:
        if not resume_ckpt:
            raise ValueError("resume_full_state=True requires experiment.resume_ckpt to be set.")
        start_epoch, global_step, best_robust_rel_l2 = load_full_training_state(
            model,
            optimizer,
            scheduler,
            scaler,
            resume_ckpt,
            device,
            load_scaler=bool(amp) and not bool(getattr(config, "amp_scaler_reset_on_resume", False)),
            uncertainty_balancer=uncertainty_balancer,
            external_gradnorm=external_gradnorm,
            reset_scheduler=bool(getattr(config, "scheduler_reset_on_resume", False)),
            steps_per_epoch=len(train_loader),
            target_epochs=int(config.epochs),
        )
    elif resume_ckpt:
        if is_main_process():
            print(f"[init] experiment.resume_ckpt is being used for partial model initialization from {resume_ckpt}")
        load_partial_state_dict(model, resume_ckpt, device)
    elif init_ckpt:
        if is_main_process():
            print(f"[init] Loading model weights from {init_ckpt}")
        load_partial_state_dict(model, init_ckpt, device)

    train_model = model
    if multi_gpu_strategy == "data_parallel" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        visible_gpus = torch.cuda.device_count()
        if int(config.batch_size) < visible_gpus and is_main_process():
            print(
                f"[multi-gpu warning] batch_size={int(config.batch_size)} is smaller than the number of visible GPUs "
                f"({visible_gpus}); DataParallel will underutilize devices."
            )
        # Keep DataParallel as lightweight as possible:
        # - make device placement explicit
        # - avoid buffer broadcasts because SMART-family models do not use BN/SyncBN
        device_ids = list(range(visible_gpus))
        train_model = DataParallel(
            model,
            device_ids=device_ids,
            output_device=device_ids[0],
            dim=0,
        )
        if is_main_process():
            print(f"[multi-gpu] Using DataParallel on {visible_gpus} GPUs.")
    elif dist_info["enabled"]:
        ddp_kwargs = {
            "device_ids": [dist_info["local_rank"]] if device.type == "cuda" else None,
            "output_device": dist_info["local_rank"] if device.type == "cuda" else None,
            "broadcast_buffers": False,
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
            "static_graph": True,
        }
        train_model = DDP(model, **ddp_kwargs)

    train_extra_keys = [
        "loss_supervised_primary",
        "loss_supervised_secondary",
        "loss_supervised_mean",
        "loss_supervised_worst",
        "loss_supervised_worst_soft",
        "loss_prediction_consistency",
        "primary_inverse_density_fraction",
        "primary_inverse_density_beta",
        "secondary_inverse_density_fraction",
        "secondary_inverse_density_beta",
        "shared_shift_family_id",
        "primary_shift_intensity",
        "secondary_shift_intensity",
        "learned_weight_supervised_mean",
        "learned_weight_supervised_worst",
        "learned_weight_prediction_consistency",
        "learned_logit_supervised_mean",
        "learned_logit_supervised_worst",
        "learned_logit_prediction_consistency",
        "external_gradnorm_weight_primary",
        "external_gradnorm_weight_secondary",
        "external_gradnorm_weight_prediction_consistency",
        "external_gradnorm_loss",
        "config_reference_grad_norm",
        "config_direction_norm",
        "config_direction_scale",
        "config_direction_cosine",
        "config_direction_fallback",
        "config_full_mean_grad_norm",
        "config_full_direction_norm",
        "config_full_min_cosine",
        "config_full_fallback",
        "gradient_norm_raw",
        "parameter_norm",
        "gradient_max_abs",
        "gradient_nonfinite",
    ]

    fuse_consistency_views = bool(getattr(config, "fuse_consistency_views", False))
    if (
        primary_view_points != secondary_view_points
        or primary_view_subsampled_geometry_points != secondary_view_subsampled_geometry_points
    ):
        if fuse_consistency_views and is_main_process():
            print(
                "[consistency] Disabling fused consistency views because primary and secondary "
                "geometry budgets differ."
            )
        fuse_consistency_views = False

    configured_source_points = int(getattr(config, "num_body_points", 0))
    if (
        use_prediction_consistency
        and configured_source_points > 0
        and primary_view_points >= configured_source_points
        and secondary_view_points >= configured_source_points
    ):
        raise ValueError(
            f"{config.model_name} has no effective two-view geometry augmentation: "
            f"num_body_points={configured_source_points}, "
            f"primary_view_geometry_points={primary_view_points}, "
            f"secondary_view_geometry_points={secondary_view_points}. "
            "Set experiment.num_body_points=0 (full cloud) or make both view budgets smaller "
            "than the dataset geometry budget."
        )
    if is_main_process() and use_prediction_consistency:
        source_text = "full geometry cloud" if configured_source_points <= 0 else f"{configured_source_points} dataset geometry points"
        print(
            f"[consistency] source={source_text}, "
            f"views=({primary_view_points}, {secondary_view_points}), "
            f"modes=({getattr(config, 'train_primary_sampling_mode', 'uniform_wor')}, "
            f"{getattr(config, 'train_secondary_sampling_mode', 'uniform_wor')})"
        )

    try:
        for ep in tqdm(range(start_epoch, config.epochs), desc="Epochs", dynamic_ncols=True):
            t1 = default_timer()
            set_dataset_epoch(train_data, ep)
            set_dataset_epoch(test_data, 0)
            if train_sampler is not None:
                train_sampler.set_epoch(ep)

            train_losses = init_metric_tensor_dict(fields["surface"], fields["volume"], device, extra_keys=train_extra_keys)
            train_model.train()
            train_sample_count = 0
            train_batch_source = (
                CudaPrefetchLoader(
                    train_loader,
                    device,
                    keep_cpu_indices=geometry_cpu_indices if keep_geometry_cpu_for_view_sampling else (),
                )
                if cuda_batch_prefetch
                else train_loader
            )
            train_pbar = tqdm(
                train_batch_source,
                desc=f"Train {ep + 1}/{config.epochs}",
                leave=False,
                dynamic_ncols=True,
                miniters=max(1, int(log_every_n_steps)),
                mininterval=1.0,
                smoothing=0.0,
                disable=not is_main_process(),
            )

            warmup = consistency_warmup_factor(ep, getattr(config, "consistency_warmup_epochs", 0))
            pred_consistency_weight = warmup * float(getattr(config, "prediction_consistency_weight", 1.0))
            if not use_prediction_consistency:
                pred_consistency_weight = 0.0
            use_latent_consistency = bool(getattr(config, "use_latent_consistency", False))

            for batch_idx, batch in enumerate(train_pbar):
                geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = unpack_batch(batch, params_dim)
                primary_sampling_mode = str(getattr(config, "train_primary_sampling_mode", "uniform_wor"))
                secondary_sampling_mode = str(getattr(config, "train_secondary_sampling_mode", "mixed"))
                shared_shift_family_id = -1
                primary_sine_axis = None
                secondary_sine_axis = None
                primary_sine_mix_fraction = 0.0
                secondary_sine_mix_fraction = 0.0
                if shared_shift_sampling_mode in {"random_beta_sine", "random_beta_sine_axis", "shared_random"}:
                    family_generator = _cpu_generator(
                        int(config.random_seed + ep * 1000003 + batch_idx * 10007 + 5)
                    )
                    shared_shift_family_id = sample_shared_shift_family(
                        getattr(config, "shared_shift_family_probabilities", [1.0 / 3.0] * 3),
                        family_generator,
                    )
                    if shared_shift_family_id == 0:
                        primary_sampling_mode = "inverse_density_wor"
                        secondary_sampling_mode = "inverse_density_wor"
                    else:
                        primary_sampling_mode = "sinusoidal_axis_mixture_wor"
                        secondary_sampling_mode = "sinusoidal_axis_mixture_wor"
                        sine_axis = 1 if shared_shift_family_id == 1 else 0
                        primary_sine_axis = sine_axis
                        secondary_sine_axis = sine_axis
                if geo_log_density is None and (
                    _sampling_mode_requires_density(primary_sampling_mode)
                    or _sampling_mode_requires_density(secondary_sampling_mode)
                ):
                    raise RuntimeError(
                        f"{config.model_name} requested density-based geometry-view resampling "
                        f"(primary={primary_sampling_mode!r}, secondary={secondary_sampling_mode!r}) "
                        "but the dataset did not return geometry log density."
                    )

                primary_beta_generator = _cpu_generator(int(config.random_seed + ep * 1000003 + batch_idx * 10007 + 17))
                if shared_shift_family_id == 0:
                    primary_inverse_density_beta = sample_uniform_beta(
                        getattr(config, "shared_shift_beta_min", 0.0),
                        getattr(config, "shared_shift_beta_max", 1.0),
                        primary_beta_generator,
                    )
                elif bool(getattr(config, "randomize_primary_inverse_density_beta", False)):
                    primary_inverse_density_beta = sample_uniform_beta(
                        getattr(config, "primary_inverse_density_beta_min", 0.0),
                        getattr(config, "primary_inverse_density_beta_max", 0.5),
                        primary_beta_generator,
                    )
                else:
                    primary_inverse_density_beta = float(getattr(config, "inverse_density_beta", 1.0))
                if shared_shift_family_id in {1, 2}:
                    primary_sine_mix_fraction = sample_uniform_beta(
                        getattr(config, "shared_shift_sine_min", 0.0),
                        getattr(config, "shared_shift_sine_max", 0.5),
                        primary_beta_generator,
                    )
                primary_view_geo, primary_view_density, primary_modes = sample_geometry_view(
                    geo_mesh,
                    geo_log_density,
                    num_points=primary_view_points,
                    mode=primary_sampling_mode,
                    inverse_density_beta=primary_inverse_density_beta,
                    mixed_inverse_density_prob=float(getattr(config, "mixed_inverse_density_prob", 0.5)),
                    seed=int(config.random_seed + ep * 1000003 + batch_idx * 10007 + 11),
                    gaussian_mask_std_fraction=float(getattr(config, "gaussian_mask_std_fraction_of_largest_extent", 0.05)),
                    gaussian_mask_prob_at_1sigma=float(getattr(config, "gaussian_mask_prob_at_1sigma", 0.33)),
                    gaussian_mask_min_survivors=int(getattr(config, "gaussian_mask_min_survivors", 16384)),
                    sinusoidal_axis=primary_sine_axis,
                    sinusoidal_mix_fraction=primary_sine_mix_fraction,
                )
                secondary_beta_generator = _cpu_generator(int(config.random_seed + ep * 1000003 + batch_idx * 10007 + 23))
                if shared_shift_family_id == 0:
                    secondary_inverse_density_beta = sample_uniform_beta(
                        getattr(config, "shared_shift_beta_min", 0.0),
                        getattr(config, "shared_shift_beta_max", 1.0),
                        secondary_beta_generator,
                    )
                elif bool(getattr(config, "randomize_secondary_inverse_density_beta", False)):
                    secondary_inverse_density_beta = sample_uniform_beta(
                        getattr(config, "secondary_inverse_density_beta_min", 0.1),
                        getattr(config, "secondary_inverse_density_beta_max", 0.5),
                        secondary_beta_generator,
                    )
                else:
                    secondary_inverse_density_beta = float(getattr(config, "inverse_density_beta", 1.0))
                if shared_shift_family_id in {1, 2}:
                    secondary_sine_mix_fraction = sample_uniform_beta(
                        getattr(config, "shared_shift_sine_min", 0.0),
                        getattr(config, "shared_shift_sine_max", 0.5),
                        secondary_beta_generator,
                    )
                secondary_view_geo, secondary_view_density, secondary_modes = sample_geometry_view(
                    geo_mesh,
                    geo_log_density,
                    num_points=secondary_view_points,
                    mode=secondary_sampling_mode,
                    inverse_density_beta=secondary_inverse_density_beta,
                    mixed_inverse_density_prob=float(getattr(config, "mixed_inverse_density_prob", 0.5)),
                    seed=int(config.random_seed + ep * 1000003 + batch_idx * 10007 + 29),
                    gaussian_mask_std_fraction=float(getattr(config, "gaussian_mask_std_fraction_of_largest_extent", 0.05)),
                    gaussian_mask_prob_at_1sigma=float(getattr(config, "gaussian_mask_prob_at_1sigma", 0.33)),
                    gaussian_mask_min_survivors=int(getattr(config, "gaussian_mask_min_survivors", 16384)),
                    sinusoidal_axis=secondary_sine_axis,
                    sinusoidal_mix_fraction=secondary_sine_mix_fraction,
                )

                primary_shift_intensity = (
                    float(primary_inverse_density_beta)
                    if shared_shift_family_id == 0
                    else float(primary_sine_mix_fraction)
                )
                secondary_shift_intensity = (
                    float(secondary_inverse_density_beta)
                    if shared_shift_family_id == 0
                    else float(secondary_sine_mix_fraction)
                )

                params = move_optional_tensor(params, device)
                surf_mesh = surf_mesh.to(device, non_blocking=True)
                surf_data = surf_data.to(device, non_blocking=True)
                vol_mesh = vol_mesh.to(device, non_blocking=True)
                vol_data = vol_data.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                buffer_snapshot = snapshot_model_buffers(model) if rollback_buffers else None

                primary_view_geo = primary_view_geo.to(device, non_blocking=True)
                if model_requires_density:
                    primary_view_density = primary_view_density.to(device, non_blocking=True)
                secondary_view_geo = secondary_view_geo.to(device, non_blocking=True)
                if model_requires_density:
                    secondary_view_density = secondary_view_density.to(device, non_blocking=True)

                with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=amp):
                    if fuse_consistency_views:
                        fused_geo = torch.cat([primary_view_geo, secondary_view_geo], dim=0)
                        fused_surf_mesh = duplicate_batch_tensor(surf_mesh)
                        fused_vol_mesh = duplicate_batch_tensor(vol_mesh)
                        fused_params = duplicate_batch_tensor(params)
                        fused_density = (
                            torch.cat([primary_view_density, secondary_view_density], dim=0)
                            if model_requires_density else None
                        )
                        fused_out = forward_model_view(
                            train_model,
                            fused_geo,
                            fused_surf_mesh,
                            fused_vol_mesh,
                            fused_params,
                            model_requires_density=model_requires_density,
                            geo_log_density=fused_density,
                            return_latent=use_latent_consistency,
                            subsampled_geometry_points=primary_view_subsampled_geometry_points,
                        )
                        if use_latent_consistency:
                            fused_surf, fused_vol, fused_latent = fused_out
                            y1_surf, y2_surf = fused_surf.chunk(2, dim=0)
                            y1_vol, y2_vol = fused_vol.chunk(2, dim=0)
                            latent1, latent2 = fused_latent.chunk(2, dim=0)
                        else:
                            fused_surf, fused_vol = fused_out
                            y1_surf, y2_surf = fused_surf.chunk(2, dim=0)
                            y1_vol, y2_vol = fused_vol.chunk(2, dim=0)
                            latent1 = None
                            latent2 = None
                    else:
                        primary_out = forward_model_view(
                            train_model,
                            primary_view_geo,
                            surf_mesh,
                            vol_mesh,
                            params,
                            model_requires_density=model_requires_density,
                            geo_log_density=primary_view_density if model_requires_density else None,
                            return_latent=use_latent_consistency,
                            subsampled_geometry_points=primary_view_subsampled_geometry_points,
                        )
                        if use_latent_consistency:
                            y1_surf, y1_vol, latent1 = primary_out
                        else:
                            y1_surf, y1_vol = primary_out
                            latent1 = None
                        secondary_out = forward_model_view(
                            train_model,
                            secondary_view_geo,
                            surf_mesh,
                            vol_mesh,
                            params,
                            model_requires_density=model_requires_density,
                            geo_log_density=secondary_view_density if model_requires_density else None,
                            return_latent=use_latent_consistency,
                            subsampled_geometry_points=secondary_view_subsampled_geometry_points,
                        )
                        if use_latent_consistency:
                            y2_surf, y2_vol, latent2 = secondary_out
                        else:
                            y2_surf, y2_vol = secondary_out
                            latent2 = None

                    y1_surf_f = y1_surf.float()
                    y1_vol_f = y1_vol.float()
                    y2_surf_f = y2_surf.float()
                    y2_vol_f = y2_vol.float()

                supervised_primary = combined_loss_fn(y1_surf_f, y1_vol_f, surf_data, vol_data) if use_surface_supervision else loss_fn(y1_vol_f, vol_data)
                supervised_secondary = combined_loss_fn(y2_surf_f, y2_vol_f, surf_data, vol_data) if use_surface_supervision else loss_fn(y2_vol_f, vol_data)
                y1_surf_teacher = y1_surf_f.detach()
                y1_vol_teacher = y1_vol_f.detach()
                if use_prediction_consistency:
                    symmetric_detached_consistency = bool(
                        getattr(config, "prediction_consistency_symmetric_detached", False)
                    )
                    pred_consistency = prediction_consistency_smooth_l1_loss(
                        y1_surf_f if symmetric_detached_consistency else y1_surf_teacher,
                        y1_vol_f if symmetric_detached_consistency else y1_vol_teacher,
                        y2_surf_f,
                        y2_vol_f,
                        beta=float(getattr(config, "prediction_consistency_smooth_l1_beta", 0.05)),
                        symmetric_detached=symmetric_detached_consistency,
                        average_groups=bool(getattr(config, "prediction_consistency_average_groups", False)),
                    )
                else:
                    pred_consistency = y1_vol_f.new_zeros(())
                supervised_mean = 0.5 * (supervised_primary + supervised_secondary)
                supervised_worst = torch.maximum(supervised_primary, supervised_secondary)
                supervised_worst_soft = soft_worst_case_loss(
                    supervised_primary,
                    supervised_secondary,
                    tau=float(getattr(config, "soft_worst_case_tau", 0.1)),
                )
                task_losses = [
                    supervised_primary.float(),
                    supervised_secondary.float(),
                    supervised_mean.float(),
                    supervised_worst_soft.float(),
                    pred_consistency.float(),
                ]
                external_gradnorm_tasks = None
                config_full_info = None
                if use_learned_task_weighting:
                    learned_tasks = [
                        task_losses[2],
                        task_losses[3],
                    ]
                    if use_prediction_consistency:
                        learned_tasks.append((pred_consistency_weight * task_losses[4]).float())
                    uncertainty_info = uncertainty_balancer.combine(
                        learned_tasks,
                        epoch_idx=ep,
                    )
                    current_task_weights = uncertainty_info["weights"].float()
                    weighted_total_loss = uncertainty_info["total_loss"].float()
                    config_info = None
                    external_gradnorm_info = None
                elif use_config_layer:
                    config_tasks = [
                        task_losses[2],
                        task_losses[3],
                    ]
                    if use_prediction_consistency:
                        config_tasks.append((pred_consistency_weight * task_losses[4]).float())
                    external_gradnorm_tasks = config_tasks
                    current_task_weights = torch.tensor(config_task_weights, device=device, dtype=torch.float32)
                    weighted_total_loss = sum(
                        weight * task_loss for weight, task_loss in zip(config_task_weights, config_tasks)
                    ).float()
                    uncertainty_info = None
                    config_info = {"task_losses": config_tasks}
                    external_gradnorm_info = None
                elif use_config_full:
                    config_full_tasks = [
                        config_task_weights[0] * task_losses[2],
                        config_task_weights[1] * task_losses[3],
                    ]
                    if use_prediction_consistency:
                        config_full_tasks.append(
                            (config_task_weights[2] * pred_consistency_weight * task_losses[4]).float()
                        )
                    current_task_weights = torch.tensor(config_task_weights, device=device, dtype=torch.float32)
                    weighted_total_loss = sum(config_full_tasks).float()
                    uncertainty_info = None
                    config_info = None
                    config_full_info = {"task_losses": config_full_tasks}
                    external_gradnorm_info = None
                elif use_external_gradnorm:
                    external_gradnorm_tasks = [
                        task_losses[2],
                        task_losses[3],
                    ]
                    if use_prediction_consistency:
                        external_gradnorm_tasks.append((pred_consistency_weight * task_losses[4]).float())
                    current_task_weights = external_gradnorm.loss_weights.detach().float()
                    weighted_total_loss = sum(
                        weight * task_loss
                        for weight, task_loss in zip(current_task_weights, external_gradnorm_tasks)
                    ).float()
                    uncertainty_info = None
                    config_info = None
                    external_gradnorm_info = {"task_losses": external_gradnorm_tasks}
                elif use_fixed_sum:
                    fixed_sum_tasks = [
                        task_losses[0]
                        if bool(getattr(config, "fixed_sum_use_view_losses", False))
                        else task_losses[2],
                        task_losses[1]
                        if bool(getattr(config, "fixed_sum_use_view_losses", False))
                        else task_losses[3],
                    ]
                    if use_prediction_consistency:
                        fixed_sum_tasks.append((pred_consistency_weight * task_losses[4]).float())
                    current_task_weights = torch.tensor(config_task_weights, device=device, dtype=torch.float32)
                    weighted_total_loss = sum(
                        weight * task_loss for weight, task_loss in zip(config_task_weights, fixed_sum_tasks)
                    ).float()
                    uncertainty_info = None
                    config_info = None
                    external_gradnorm_info = None
                else:
                    uncertainty_info = None
                    config_info = None
                    external_gradnorm_info = None
                    current_task_weights = None
                    weighted_total_loss = (
                        supervised_mean
                        + (pred_consistency_weight * pred_consistency if use_prediction_consistency else 0.0)
                    ).float()

                config_direction = None
                config_diagnostics = None
                config_full_diagnostics = None
                if config_info is not None:
                    config_direction, config_diagnostics = config_layer_gradient(
                        [
                            weight * task_loss
                            for weight, task_loss in zip(config_task_weights, config_info["task_losses"])
                        ],
                        config_reference_parameter,
                        return_diagnostics=True,
                    )

                if not torch.isfinite(weighted_total_loss):
                    restore_model_buffers(model, buffer_snapshot)
                    raise FloatingPointError(
                        f"Non-finite consistency loss detected at epoch={ep} batch={batch_idx}: "
                        f"supervised_primary={float(supervised_primary.detach().item()):.6g}, "
                        f"supervised_secondary={float(supervised_secondary.detach().item()):.6g}, "
                        f"supervised_mean={float(supervised_mean.detach().item()):.6g}, "
                        f"supervised_worst_soft={float(supervised_worst_soft.detach().item()):.6g}, "
                        f"pred_consistency={float(pred_consistency.detach().item()):.6g}"
                    )

                external_gradnorm_loss = weighted_total_loss.new_zeros((), dtype=torch.float32)
                if use_config_full:
                    config_full_diagnostics = config_full_backward(
                        config_full_info["task_losses"],
                        config_full_parameters,
                        scaler=scaler,
                        amp_enabled=amp,
                        vectorized=bool(getattr(config, "config_full_vectorized_gradients", False)),
                        allow_sequential_fallback=bool(
                            getattr(config, "config_full_allow_sequential_fallback", True)
                        ),
                    )
                elif use_external_gradnorm:
                    external_gradnorm_loss = external_gradnorm_backward(
                        external_gradnorm,
                        external_gradnorm_tasks,
                        config_reference_parameter,
                        weighted_total_loss,
                        min_loss_weights=external_gradnorm_min_weights,
                        scaler=scaler,
                        amp_enabled=amp,
                    )
                elif amp:
                    scaler.scale(weighted_total_loss).backward()
                else:
                    weighted_total_loss.backward()

                if amp:
                    scaler.unscale_(optimizer)
                if config_direction is not None:
                    if config_reference_parameter.grad is None:
                        config_reference_parameter.grad = config_direction.clone()
                    else:
                        config_reference_parameter.grad.copy_(config_direction)
                gradient_stats = gradient_diagnostics(train_model) if track_gradient_diagnostics else None
                if gradient_stats is not None and not gradient_stats["finite"]:
                    print(
                        f"[warn] Non-finite gradients at epoch={ep} batch={batch_idx}; "
                        "skipping optimizer step and reducing AMP scale."
                    )
                    restore_model_buffers(model, buffer_snapshot)
                    optimizer.zero_grad(set_to_none=True)
                    if amp:
                        scaler.update(new_scale=max(1.0, 0.5 * scaler.get_scale()))
                    continue
                if gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                if amp:
                    scaler.step(optimizer)
                    if config_full_diagnostics is not None and float(config_full_diagnostics["used_fallback"].item()) > 0.0:
                        scaler.update(new_scale=max(1.0, 0.5 * scaler.get_scale()))
                    else:
                        scaler.update()
                else:
                    optimizer.step()
                if rollback_buffers and not model_buffers_are_finite(model):
                    restore_model_buffers(model, buffer_snapshot)
                    raise FloatingPointError(
                        f"Non-finite PTv3 state after optimizer step at epoch={ep} batch={batch_idx}."
                    )
                scheduler.step()

                total_loss = weighted_total_loss.detach()
                batch_size = surf_data.size(0)
                train_sample_count += batch_size
                secondary_inverse_density_fraction = float(
                    sum(mode.startswith("inverse_density") for mode in secondary_modes) / max(len(secondary_modes), 1)
                )
                primary_inverse_density_fraction = float(
                    sum(mode.startswith("inverse_density") for mode in primary_modes) / max(len(primary_modes), 1)
                )

                with torch.inference_mode():
                    surf_rel_primary = rel_l2_loss_fn(y1_surf_teacher, surf_data) if use_surface_supervision else torch.tensor(0.0, device=device)
                    vol_rel_primary = rel_l2_loss_fn(y1_vol_teacher, vol_data)
                    surf_rel_secondary = rel_l2_loss_fn(y2_surf, surf_data) if use_surface_supervision else torch.tensor(0.0, device=device)
                    vol_rel_secondary = rel_l2_loss_fn(y2_vol, vol_data)

                    surface_loss = 0.5 * (surf_rel_primary + surf_rel_secondary)
                    volume_loss = 0.5 * (vol_rel_primary + vol_rel_secondary)

                    batch_size_float = float(batch_size)
                    train_losses["loss"] += total_loss.detach().float() * batch_size_float
                    train_losses["rel_l2_surf"] += surface_loss.detach().float() * batch_size_float
                    train_losses["rel_l2_vol"] += volume_loss.detach().float() * batch_size_float
                    train_losses["rel_l2"] += (surface_loss + volume_loss).detach().float() * batch_size_float
                    train_losses["loss_supervised_primary"] += supervised_primary.detach().float() * batch_size_float
                    train_losses["loss_supervised_secondary"] += supervised_secondary.detach().float() * batch_size_float
                    train_losses["loss_supervised_mean"] += supervised_mean.detach().float() * batch_size_float
                    train_losses["loss_supervised_worst"] += supervised_worst.detach().float() * batch_size_float
                    train_losses["loss_supervised_worst_soft"] += supervised_worst_soft.detach().float() * batch_size_float
                    train_losses["loss_prediction_consistency"] += pred_consistency.detach().float() * batch_size_float
                    if gradient_stats is not None:
                        train_losses["gradient_norm_raw"] += gradient_stats["grad_norm"] * batch_size_float
                        train_losses["parameter_norm"] += gradient_stats["parameter_norm"] * batch_size_float
                        train_losses["gradient_max_abs"] += gradient_stats["max_grad"] * batch_size_float
                        train_losses["gradient_nonfinite"] += float(not gradient_stats["finite"]) * batch_size_float
                    train_losses["primary_inverse_density_fraction"] += torch.tensor(
                        primary_inverse_density_fraction, device=device, dtype=torch.float32
                    ) * batch_size_float
                    train_losses["primary_inverse_density_beta"] += torch.tensor(
                        float(primary_inverse_density_beta), device=device, dtype=torch.float32
                    ) * batch_size_float
                    train_losses["secondary_inverse_density_fraction"] += torch.tensor(
                        secondary_inverse_density_fraction, device=device, dtype=torch.float32
                    ) * batch_size_float
                    train_losses["secondary_inverse_density_beta"] += torch.tensor(
                        float(secondary_inverse_density_beta), device=device, dtype=torch.float32
                    ) * batch_size_float
                    train_losses["shared_shift_family_id"] += torch.tensor(
                        float(shared_shift_family_id), device=device, dtype=torch.float32
                    ) * batch_size_float
                    train_losses["primary_shift_intensity"] += torch.tensor(
                        primary_shift_intensity, device=device, dtype=torch.float32
                    ) * batch_size_float
                    train_losses["secondary_shift_intensity"] += torch.tensor(
                        secondary_shift_intensity, device=device, dtype=torch.float32
                    ) * batch_size_float
                    if uncertainty_info is not None:
                        train_losses["learned_weight_supervised_mean"] += current_task_weights[0].detach().float() * batch_size_float
                        train_losses["learned_weight_supervised_worst"] += current_task_weights[1].detach().float() * batch_size_float
                        train_losses["learned_logit_supervised_mean"] += uncertainty_info["logits"][0].detach().float() * batch_size_float
                        train_losses["learned_logit_supervised_worst"] += uncertainty_info["logits"][1].detach().float() * batch_size_float
                        if use_prediction_consistency:
                            train_losses["learned_weight_prediction_consistency"] += current_task_weights[2].detach().float() * batch_size_float
                            train_losses["learned_logit_prediction_consistency"] += uncertainty_info["logits"][2].detach().float() * batch_size_float
                    if external_gradnorm_info is not None:
                        train_losses["external_gradnorm_weight_primary"] += current_task_weights[0].detach().float() * batch_size_float
                        train_losses["external_gradnorm_weight_secondary"] += current_task_weights[1].detach().float() * batch_size_float
                        if use_prediction_consistency:
                            train_losses["external_gradnorm_weight_prediction_consistency"] += current_task_weights[2].detach().float() * batch_size_float
                        train_losses["external_gradnorm_loss"] += external_gradnorm_loss.detach().float() * batch_size_float
                    if config_diagnostics is not None:
                        train_losses["config_reference_grad_norm"] += config_diagnostics["reference_grad_norm"] * batch_size_float
                        train_losses["config_direction_norm"] += config_diagnostics["direction_norm"] * batch_size_float
                        train_losses["config_direction_scale"] += config_diagnostics["direction_scale"] * batch_size_float
                        train_losses["config_direction_cosine"] += config_diagnostics["direction_cosine"] * batch_size_float
                        train_losses["config_direction_fallback"] += config_diagnostics["used_fallback"] * batch_size_float
                    if config_full_diagnostics is not None:
                        train_losses["config_full_mean_grad_norm"] += config_full_diagnostics["mean_grad_norm"] * batch_size_float
                        train_losses["config_full_direction_norm"] += config_full_diagnostics["direction_norm"] * batch_size_float
                        train_losses["config_full_min_cosine"] += config_full_diagnostics["min_cosine"] * batch_size_float
                        train_losses["config_full_fallback"] += config_full_diagnostics["used_fallback"] * batch_size_float

                    if use_surface_supervision:
                        pred_surf_primary = y1_surf_teacher * std_surf + mean_surf
                        pred_surf_secondary = y2_surf.detach() * std_surf + mean_surf
                        gt_surf = surf_data * std_surf + mean_surf
                        accumulate_channel_metrics_tensor(train_losses, "rel_l2_surf", pred_surf_primary, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size, metric_weight=0.5)
                        accumulate_channel_metrics_tensor(train_losses, "rel_l2_surf", pred_surf_secondary, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size, metric_weight=0.5)

                    pred_vol_primary = y1_vol_teacher * std_vol + mean_vol
                    pred_vol_secondary = y2_vol.detach() * std_vol + mean_vol
                    gt_vol = vol_data * std_vol + mean_vol
                    accumulate_channel_metrics_tensor(train_losses, "rel_l2_vol", pred_vol_primary, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size, metric_weight=0.5)
                    accumulate_channel_metrics_tensor(train_losses, "rel_l2_vol", pred_vol_secondary, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size, metric_weight=0.5)

                global_step += 1
                should_log = batch_idx % log_every_n_steps == 0 or batch_idx == len(train_loader) - 1
                if should_log:
                    # Every DDP rank must enter the reduction. Only the main
                    # rank owns the W&B run and emits the resulting log.
                    log_scalars = distributed_average_scalars(
                        [
                            total_loss.item(),
                            (surface_loss + volume_loss).item(),
                            surface_loss.item(),
                            volume_loss.item(),
                            supervised_primary.item(),
                            supervised_secondary.item(),
                            supervised_mean.item(),
                            supervised_worst_soft.item(),
                            pred_consistency.item(),
                            primary_inverse_density_fraction,
                            float(primary_inverse_density_beta),
                            secondary_inverse_density_fraction,
                            float(secondary_inverse_density_beta),
                            float(shared_shift_family_id),
                            primary_shift_intensity,
                            secondary_shift_intensity,
                        ]
                    )
                    if run is not None:
                        wandb.log(
                            {
                                "train/batch_loss": log_scalars[0],
                                "train/batch_rel_l2": log_scalars[1],
                                "train/batch_rel_l2_surf": log_scalars[2],
                                "train/batch_rel_l2_vol": log_scalars[3],
                                "train/batch_supervised_primary": log_scalars[4],
                                "train/batch_supervised_secondary": log_scalars[5],
                                "train/batch_supervised_mean": log_scalars[6],
                                "train/batch_supervised_worst_soft": log_scalars[7],
                                "train/batch_prediction_consistency": log_scalars[8],
                                "train/batch_primary_inverse_density_fraction": log_scalars[9],
                                "train/batch_primary_inverse_density_beta": log_scalars[10],
                                "train/batch_secondary_inverse_density_fraction": log_scalars[11],
                                "train/batch_secondary_inverse_density_beta": log_scalars[12],
                                "train/batch_shared_shift_family_id": log_scalars[13],
                                "train/batch_primary_shift_intensity": log_scalars[14],
                                "train/batch_secondary_shift_intensity": log_scalars[15],
                                "train/prediction_consistency_weight": pred_consistency_weight,
                                "train/batch_amp_scale": float(scaler.get_scale()) if amp else 1.0,
                                "lr": scheduler.get_last_lr()[0],
                                "epoch": ep,
                            },
                            step=global_step,
                        )
                    if gradient_stats is not None and run is not None:
                        wandb.log(
                            {
                                "train/batch_gradient_norm_raw": float(gradient_stats["grad_norm"].item()),
                                "train/batch_parameter_norm": float(gradient_stats["parameter_norm"].item()),
                                "train/batch_gradient_max_abs": float(gradient_stats["max_grad"].item()),
                                "train/batch_gradient_nonfinite": float(not gradient_stats["finite"]),
                            },
                            step=global_step,
                        )
                    if uncertainty_info is not None:
                        uncertainty_values = [
                            float(current_task_weights[0].item()),
                            float(current_task_weights[1].item()),
                            float(uncertainty_info["logits"][0].item()),
                            float(uncertainty_info["logits"][1].item()),
                        ]
                        if use_prediction_consistency:
                            uncertainty_values.extend(
                                [
                                    float(current_task_weights[2].item()),
                                    float(uncertainty_info["logits"][2].item()),
                                ]
                            )
                        uncertainty_log_scalars = distributed_average_scalars(uncertainty_values)
                        uncertainty_log_dict = {
                            "train/batch_learned_weight_supervised_mean": uncertainty_log_scalars[0],
                            "train/batch_learned_weight_supervised_worst": uncertainty_log_scalars[1],
                            "train/batch_learned_logit_supervised_mean": uncertainty_log_scalars[2],
                            "train/batch_learned_logit_supervised_worst": uncertainty_log_scalars[3],
                        }
                        if use_prediction_consistency:
                            uncertainty_log_dict["train/batch_learned_weight_prediction_consistency"] = uncertainty_log_scalars[4]
                            uncertainty_log_dict["train/batch_learned_logit_prediction_consistency"] = uncertainty_log_scalars[5]
                        if run is not None:
                            wandb.log(uncertainty_log_dict, step=global_step)
                    if external_gradnorm_info is not None and run is not None:
                        wandb.log(
                            {
                                "train/batch_external_gradnorm_loss": float(external_gradnorm_loss.item()),
                                "train/batch_external_gradnorm_weight_primary": float(current_task_weights[0].item()),
                                "train/batch_external_gradnorm_weight_secondary": float(current_task_weights[1].item()),
                                "train/batch_external_gradnorm_weight_prediction_consistency": float(current_task_weights[2].item()),
                            },
                            step=global_step,
                        )
                    if config_diagnostics is not None and run is not None:
                        wandb.log(
                            {
                                "train/batch_config_reference_grad_norm": float(config_diagnostics["reference_grad_norm"].item()),
                                "train/batch_config_direction_norm": float(config_diagnostics["direction_norm"].item()),
                                "train/batch_config_direction_scale": float(config_diagnostics["direction_scale"].item()),
                                "train/batch_config_direction_cosine": float(config_diagnostics["direction_cosine"].item()),
                                "train/batch_config_direction_fallback": float(config_diagnostics["used_fallback"].item()),
                            },
                            step=global_step,
                        )
                    if config_full_diagnostics is not None and run is not None:
                        wandb.log(
                            {
                                "train/batch_config_full_mean_grad_norm": float(config_full_diagnostics["mean_grad_norm"].item()),
                                "train/batch_config_full_direction_norm": float(config_full_diagnostics["direction_norm"].item()),
                                "train/batch_config_full_min_cosine": float(config_full_diagnostics["min_cosine"].item()),
                                "train/batch_config_full_fallback": float(config_full_diagnostics["used_fallback"].item()),
                                "train/batch_config_full_amp_scale": float(config_full_diagnostics["amp_scale"].item()),
                            },
                            step=global_step,
                        )
                if should_log:
                    train_pbar.set_postfix(loss=f"{total_loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                    train_pbar.refresh()

            if is_dist_enabled():
                dist.barrier()
            aligned_metrics = evaluate_loader(
                model=model,
                loader=test_loader,
                config=config,
                device=device,
                dtype=dtype,
                amp=amp,
                params_dim=params_dim,
                mean_surf=mean_surf,
                std_surf=std_surf,
                mean_vol=mean_vol,
                std_vol=std_vol,
                fields=fields,
                rel_l2_loss_fn=rel_l2_loss_fn,
                combined_loss_fn=combined_loss_fn,
                loss_fn=loss_fn,
                use_surface_supervision=use_surface_supervision,
                mode_name=str(getattr(config, "eval_aligned_sampling_mode", "uniform_wor")),
                num_view_points=eval_aligned_view_points,
                eval_subsampled_geometry_points=eval_aligned_subsampled_geometry_points,
                fixed_seed_offset=50000011,
                model_requires_density=model_requires_density,
                cuda_batch_prefetch=cuda_batch_prefetch,
                keep_cpu_indices=geometry_cpu_indices if keep_geometry_cpu_for_view_sampling else (),
            )
            shifted_metrics = evaluate_loader(
                model=model,
                loader=test_loader,
                config=config,
                device=device,
                dtype=dtype,
                amp=amp,
                params_dim=params_dim,
                mean_surf=mean_surf,
                std_surf=std_surf,
                mean_vol=mean_vol,
                std_vol=std_vol,
                fields=fields,
                rel_l2_loss_fn=rel_l2_loss_fn,
                combined_loss_fn=combined_loss_fn,
                loss_fn=loss_fn,
                use_surface_supervision=use_surface_supervision,
                mode_name=str(getattr(config, "eval_shifted_sampling_mode", "inverse_density_wor")),
                num_view_points=eval_shifted_view_points,
                eval_subsampled_geometry_points=eval_shifted_subsampled_geometry_points,
                fixed_seed_offset=70000029,
                model_requires_density=model_requires_density,
                cuda_batch_prefetch=cuda_batch_prefetch,
                keep_cpu_indices=geometry_cpu_indices if keep_geometry_cpu_for_view_sampling else (),
            )
            if is_dist_enabled():
                dist.barrier()

            if is_dist_enabled():
                key_list = list(train_losses.keys())
                metric_tensor = torch.tensor(
                    [float(train_losses[key].item()) for key in key_list] + [float(train_sample_count)],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                train_sample_count = max(int(metric_tensor[-1].item()), 1)
                for idx, key in enumerate(key_list):
                    train_losses[key] = metric_tensor[idx].to(device=device, dtype=torch.float32)

            denom = max(int(train_sample_count), 1)
            for loss_name in train_losses.keys():
                train_losses[loss_name] = train_losses[loss_name] / float(denom)

            train_losses_log = {key: float(value.detach().cpu().item()) for key, value in train_losses.items()}

            robust_rel_l2 = 0.5 * (aligned_metrics["rel_l2"] + shifted_metrics["rel_l2"])
            robust_loss = 0.5 * (aligned_metrics["loss"] + shifted_metrics["loss"])
            checkpoint_extra_metrics = {
                "test_aligned_metrics": aligned_metrics,
                "test_shifted_metrics": shifted_metrics,
                "test_robust_rel_l2": robust_rel_l2,
            }
            if uncertainty_balancer is not None:
                checkpoint_extra_metrics["task_weighting_state_dict"] = uncertainty_balancer.state_dict()
            if external_gradnorm is not None:
                checkpoint_extra_metrics["external_gradnorm_state_dict"] = external_gradnorm.state_dict()

            if robust_rel_l2 < best_robust_rel_l2 and is_main_process():
                best_robust_rel_l2 = robust_rel_l2
                torch.save(
                    build_training_checkpoint(
                        epoch=ep,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        loss=robust_loss,
                        rel_l2_loss=robust_rel_l2,
                        surface_fields=fields["surface"],
                        volume_fields=fields["volume"],
                        global_step=global_step,
                        best_robust_rel_l2=best_robust_rel_l2,
                        extra_metrics=checkpoint_extra_metrics,
                    ),
                    "checkpoints/" + model_checkpoint_name + "_best.pt",
                )

            if is_main_process():
                torch.save(
                    build_training_checkpoint(
                        epoch=ep,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        loss=robust_loss,
                        rel_l2_loss=robust_rel_l2,
                        surface_fields=fields["surface"],
                        volume_fields=fields["volume"],
                        global_step=global_step,
                        best_robust_rel_l2=best_robust_rel_l2,
                        extra_metrics=checkpoint_extra_metrics,
                    ),
                    "checkpoints/" + model_checkpoint_name + "_last.pt",
                )

            t2 = default_timer()
            if is_main_process():
                print(
                    f"epoch: {ep}, t2-t1 (epoch time): {t2 - t1:.5f}, "
                    f"train loss: {train_losses_log['loss']:.5f}, "
                    f"aligned rel_l2: {aligned_metrics['rel_l2']:.5f}, "
                    f"shifted rel_l2: {shifted_metrics['rel_l2']:.5f}, "
                    f"robust rel_l2: {robust_rel_l2:.5f}"
                )

            if run is not None:
                wandb_dict = {
                    "lr": scheduler.get_last_lr()[0],
                    "test/robust_rel_l2": robust_rel_l2,
                    "test/robust_loss": robust_loss,
                    "train/prediction_consistency_weight": pred_consistency_weight,
                }
                wandb_dict.update({f"train/{key}": value for key, value in train_losses_log.items()})
                wandb_dict.update({f"test_aligned/{key}": value for key, value in aligned_metrics.items()})
                wandb_dict.update({f"test_shifted/{key}": value for key, value in shifted_metrics.items()})
                add_all_field_metrics(wandb_dict, "train", fields["surface"], fields["volume"], metric_values=train_losses_log)
                add_all_field_metrics(wandb_dict, "test_aligned", fields["surface"], fields["volume"], metric_values=aligned_metrics)
                add_all_field_metrics(wandb_dict, "test_shifted", fields["surface"], fields["volume"], metric_values=shifted_metrics)
                add_canonical_field_metrics(wandb_dict, "train", fields["surface"], fields["volume"], metric_values=train_losses_log)
                add_canonical_field_metrics(wandb_dict, "test_aligned", fields["surface"], fields["volume"], metric_values=aligned_metrics)
                add_canonical_field_metrics(wandb_dict, "test_shifted", fields["surface"], fields["volume"], metric_values=shifted_metrics)
                wandb_dict["meta/training_surface_signals"] = ",".join(fields["surface"])
                wandb_dict["meta/training_volume_signals"] = ",".join(fields["volume"])
                wandb.log(wandb_dict, step=global_step)

    except KeyboardInterrupt:
        if is_main_process():
            print("\nTraining interrupted by user (Ctrl+C). Saving current state and exiting cleanly...")
        try:
            emergency_state = {
                "epoch": locals().get("ep", -1),
                "global_step": global_step,
                "best_robust_rel_l2": best_robust_rel_l2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
            if uncertainty_balancer is not None:
                emergency_state["task_weighting_state_dict"] = uncertainty_balancer.state_dict()
            if external_gradnorm is not None:
                emergency_state["external_gradnorm_state_dict"] = external_gradnorm.state_dict()
            if is_main_process():
                torch.save(emergency_state, "checkpoints/" + model_checkpoint_name + "_last.pt")
                print("Saved the latest checkpoint before exiting.")
        except Exception as exc:
            if is_main_process():
                print(f"Could not save an emergency checkpoint: {exc}")
    finally:
        if run is not None:
            best_ckpt = os.path.join("checkpoints", model_checkpoint_name + "_best.pt")
            last_ckpt = os.path.join("checkpoints", model_checkpoint_name + "_last.pt")
            if os.path.isfile(best_ckpt) or os.path.isfile(last_ckpt):
                artifact = wandb.Artifact("model", type="model")
                if os.path.isfile(best_ckpt):
                    artifact.add_file(best_ckpt)
                if os.path.isfile(last_ckpt):
                    artifact.add_file(last_ckpt)
                run.log_artifact(artifact)
            run.finish()

        # Drop references held by the last prefetch iterator before asking
        # NCCL to release its CUDA communicators.  This is especially
        # important with persistent workers and multi-GPU SATLOSS runs.
        train_pbar = None
        train_batch_source = None
        train_loader = None
        test_loader = None
        batch = None
        geo_mesh = None
        surf_mesh = None
        surf_data = None
        vol_mesh = None
        vol_data = None
        params = None
        geo_log_density = None
        primary_view_geo = None
        primary_view_density = None
        secondary_view_geo = None
        secondary_view_density = None
        fused_geo = None
        fused_surf_mesh = None
        fused_vol_mesh = None
        fused_params = None
        fused_density = None
        fused_out = None
        fused_surf = None
        fused_vol = None
        y1_surf = None
        y2_surf = None
        y1_vol = None
        y2_vol = None
        y1_surf_f = None
        y2_surf_f = None
        y1_vol_f = None
        y2_vol_f = None
        weighted_total_loss = None
        supervised_primary = None
        supervised_secondary = None
        pred_consistency = None
        train_model = None
        model = None
        optimizer = None
        scheduler = None
        scaler = None
        uncertainty_balancer = None
        external_gradnorm = None
        gc.collect()
        cleanup_distributed()

~~~

### MSPT model implementation


~~~python
"""MSPT adapter for the SMART DrivAerML surface/volume interface.

The attention, pooled-supernode update, block ordering, and point restoration
follow the official unstructured MSPT implementation.  The only adapter logic
is concatenating geometry support points with surface/volume query points and
splitting the final pointwise head back into SMART's two outputs.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn.flash_attn_interface import flash_attn_func
except Exception:
    flash_attn_func = None


ACTIVATIONS = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}

_BALLTREE_FALLBACK_WARNED = False


def _rotate_half(x):
    first = x[..., ::2]
    second = x[..., 1::2]
    return torch.stack((-second, first), dim=-1).reshape_as(x)


def _rope_cache(seq_len, head_dim, device, dtype, base=10000.0):
    if head_dim % 2 != 0:
        raise ValueError(f"MSPT RoPE requires an even head dimension, got {head_dim}")
    half_dim = head_dim // 2
    frequency_index = torch.arange(half_dim, device=device, dtype=torch.float32)
    inverse_frequency = base ** (-frequency_index / half_dim)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    frequencies = torch.einsum("i,j->ij", positions, inverse_frequency)
    cos = torch.stack((frequencies.cos(), frequencies.cos()), dim=-1).reshape(seq_len, head_dim)
    sin = torch.stack((frequencies.sin(), frequencies.sin()), dim=-1).reshape(seq_len, head_dim)
    return cos.to(dtype=dtype)[None, None], sin.to(dtype=dtype)[None, None]


def _apply_rope(q, k, cos, sin):
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _partition_spatial_tree(points, num_chunks):
    """Fallback spatial partition with the same contiguous-patch contract."""
    if num_chunks <= 0 or num_chunks & (num_chunks - 1):
        raise ValueError("MSPT spatial partition requires a power-of-two patch count.")
    groups = [torch.arange(points.shape[0], device=points.device)]
    while len(groups) < num_chunks:
        next_groups = []
        for group in groups:
            if group.numel() <= 1:
                raise ValueError("Not enough points to build the requested MSPT patches.")
            group_points = points[group]
            spread = group_points.max(dim=0).values - group_points.min(dim=0).values
            split_dim = int(torch.argmax(spread).item())
            order = torch.argsort(group_points[:, split_dim], stable=True)
            midpoint = group.numel() // 2
            next_groups.extend([group[order[:midpoint]], group[order[midpoint:]]])
        groups = next_groups
    return torch.cat(groups, dim=0)


def _partition_balltree(points, num_chunks):
    """Use the official balltree-erwin partitioner when installed.

    The repository's dependency is optional in this project.  The deterministic
    spatial-tree fallback preserves MSPT's patch locality and shape contract
    when balltree-erwin is unavailable.
    """
    global _BALLTREE_FALLBACK_WARNED
    try:
        from balltree import partition_balltree

        batch_index = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
        target_level = max(0, math.ceil(math.log2(num_chunks)))
        partition = partition_balltree(points, batch_index, target_level).long()
        if partition.numel() >= points.shape[0]:
            return partition[: points.shape[0]].to(device=points.device)
    except ImportError:
        pass

    if not _BALLTREE_FALLBACK_WARNED:
        warnings.warn(
            "balltree-erwin is unavailable; MSPT is using its deterministic "
            "spatial-tree patch fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        _BALLTREE_FALLBACK_WARNED = True
    return _partition_spatial_tree(points, num_chunks)


class MSPTMLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", residual=True):
        super().__init__()
        if act not in ACTIVATIONS:
            raise ValueError(f"Unsupported MSPT activation: {act}")
        activation = ACTIVATIONS[act]
        self.residual = bool(residual)
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), activation()) for _ in range(int(n_layers))]
        )
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            update = layer(x)
            x = x + update if self.residual else update
        return self.linear_post(x)


class ChunkedGlobalPoolAttention(nn.Module):
    """Official MSPT parallelized multi-scale attention block."""

    def __init__(
        self,
        dim,
        heads=8,
        V=16,
        Q=1,
        dropout=0.1,
        pool="mean",
        use_rope=False,
        rope_base=10000.0,
        use_flash_attn=False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.V = int(V)
        self.Q = int(Q)
        self.pool = str(pool)
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self.use_flash_attn = bool(use_flash_attn and flash_attn_func is not None)
        if self.pool == "linear":
            self.pool_proj = nn.Linear(self.dim, self.Q * self.dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.heads,
            dropout=dropout,
            batch_first=True,
        )
        if self.use_rope or self.use_flash_attn:
            if self.dim % self.heads != 0:
                raise ValueError(f"MSPT attention dimension {self.dim} must divide heads {self.heads}")
            self.head_dim = self.dim // self.heads
        self.norm = nn.LayerNorm(self.dim)
        self.ff = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.GELU(),
            nn.Linear(4 * self.dim, self.dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _pad_to_multiple(x, multiple, dim=1):
        length = x.size(dim)
        pad_len = (multiple - (length % multiple)) % multiple
        if pad_len == 0:
            return x, 0
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        return torch.cat([x, x.new_zeros(pad_shape)], dim=dim), pad_len

    def _pool(self, chunks):
        batch_size, _num_chunks, seq_len, _dim = chunks.shape
        if self.pool == "mean":
            if self.Q == 1:
                pooled = chunks.mean(dim=2, keepdim=True)
            else:
                k = min(self.Q, seq_len)
                norms = chunks.norm(dim=-1)
                order = torch.argsort(norms, dim=2, descending=True)
                running_sum = chunks.sum(dim=2)
                counts = torch.full(
                    (batch_size, self.V),
                    seq_len,
                    device=chunks.device,
                    dtype=chunks.dtype,
                )
                means = []
                for q_index in range(k):
                    means.append((running_sum / counts.unsqueeze(-1)).unsqueeze(2))
                    if q_index == k - 1:
                        break
                    selected_index = order[:, :, q_index].unsqueeze(-1).unsqueeze(-1).expand(
                        -1, -1, 1, chunks.shape[-1]
                    )
                    selected = torch.gather(chunks, 2, selected_index).squeeze(2)
                    running_sum = running_sum - selected
                    counts = counts - 1
                pooled = torch.cat(means, dim=2)
                if k < self.Q:
                    pooled = torch.cat(
                        [
                            pooled,
                            chunks.new_zeros(batch_size, self.V, self.Q - k, self.dim),
                        ],
                        dim=2,
                    )
        elif self.pool == "max":
            k = min(self.Q, seq_len)
            pooled, _ = chunks.topk(k=k, dim=2)
            if k < self.Q:
                pooled = torch.cat(
                    [pooled, chunks.new_zeros(batch_size, self.V, self.Q - k, self.dim)], dim=2
                )
        elif self.pool == "linear":
            pooled = self.pool_proj(chunks.mean(dim=2)).view(batch_size, self.V, self.Q, self.dim)
        else:
            raise ValueError(f"Unsupported MSPT pooling mode: {self.pool}")
        if pooled.size(2) == 1:
            pooled = pooled.expand(batch_size, self.V, self.Q, self.dim)
        return pooled

    def forward(self, features, prev_supernodes=None):
        batch_size, num_points, _ = features.shape
        x, pad_len = self._pad_to_multiple(features, self.V, dim=1)
        padded_points = x.size(1)
        seq_len = padded_points // self.V
        chunks = x.view(batch_size, self.V, seq_len, self.dim)

        pooled = self._pool(chunks)
        global_tokens = pooled.reshape(batch_size, self.V * self.Q, self.dim)
        if prev_supernodes is not None:
            if prev_supernodes.shape != global_tokens.shape:
                raise ValueError(
                    f"MSPT supernode shape {tuple(prev_supernodes.shape)} does not match "
                    f"{tuple(global_tokens.shape)}"
                )
            global_tokens = global_tokens + prev_supernodes.to(
                device=global_tokens.device, dtype=global_tokens.dtype
            )

        expanded_global = global_tokens.unsqueeze(1).expand(-1, self.V, -1, -1)
        sequence = torch.cat([chunks, expanded_global], dim=2)
        sequence = sequence.view(batch_size * self.V, seq_len + self.V * self.Q, self.dim)
        residual = sequence
        sequence = self.norm(sequence)
        attention_out = self._self_attention(sequence)
        sequence = residual + attention_out
        sequence = sequence + self.ff(self.norm(sequence))
        sequence = sequence.view(batch_size, self.V, seq_len + self.V * self.Q, self.dim)

        point_features = sequence[:, :, :seq_len, :].reshape(batch_size, padded_points, self.dim)
        if pad_len > 0:
            point_features = point_features[:, :-pad_len, :]
        supernodes = sequence[:, :, -self.V * self.Q :, :].mean(dim=1)
        return point_features, supernodes

    def _self_attention(self, sequence):
        if not (self.use_rope or self.use_flash_attn):
            attention_out, _ = self.attn(sequence, sequence, sequence, need_weights=False)
            return attention_out

        batch_size, sequence_length, _ = sequence.shape
        qkv = F.linear(sequence, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, sequence_length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            cos, sin = _rope_cache(
                sequence_length,
                self.head_dim,
                sequence.device,
                sequence.dtype,
                base=self.rope_base,
            )
            q, k = _apply_rope(q, k, cos, sin)
        dropout_probability = self.attn.dropout if self.training else 0.0
        if self.use_flash_attn and sequence.is_cuda:
            output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_probability,
                softmax_scale=None,
                causal=False,
            )
        else:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_probability,
            )
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, self.dim)
        return F.linear(output, self.attn.out_proj.weight, self.attn.out_proj.bias)


class MSPTBlock(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout,
        act="gelu",
        mlp_ratio=1,
        last_layer=False,
        out_dim=1,
        V=32,
        Q=1,
        attn_pool="mean",
        use_checkpoint=True,
        use_rope=False,
        rope_base=10000.0,
        use_flash_attn=False,
    ):
        super().__init__()
        self.last_layer = bool(last_layer)
        self.use_checkpoint = bool(use_checkpoint)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = ChunkedGlobalPoolAttention(
            dim=hidden_dim,
            heads=num_heads,
            V=V,
            Q=Q,
            dropout=dropout,
            pool=attn_pool,
            use_rope=use_rope,
            rope_base=rope_base,
            use_flash_attn=use_flash_attn,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MSPTMLP(
            hidden_dim,
            hidden_dim * int(mlp_ratio),
            hidden_dim,
            n_layers=0,
            act=act,
            residual=False,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, features, supernodes=None):
        if supernodes is None:
            supernodes = features.new_zeros(
                features.shape[0], self.Attn.V * self.Attn.Q, self.Attn.dim
            )
        attention_input = self.ln_1(features)

        def attention_forward(inputs, previous_supernodes):
            return self.Attn(inputs, previous_supernodes)

        if self.training and self.use_checkpoint:
            attention_out, supernodes = checkpoint(
                attention_forward,
                attention_input,
                supernodes,
                use_reentrant=False,
            )
            features = features + attention_out
            features = features + checkpoint(
                self.mlp,
                self.ln_2(features),
                use_reentrant=False,
            )
        else:
            attention_out, supernodes = self.Attn(attention_input, supernodes)
            features = features + attention_out
            features = features + self.mlp(self.ln_2(features))

        if self.last_layer:
            return self.mlp2(self.ln_3(features)), supernodes
        return features, supernodes


class MSPT(nn.Module):
    """Unstructured MSPT adapted for SMART's surface and volume queries."""

    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        num_blocks=6,
        n_hidden=256,
        num_heads=8,
        dropout=0.1,
        activation="gelu",
        mlp_ratio=1,
        V=32,
        Q=1,
        attn_pool="mean",
        chunking_mode="balltree",
        use_checkpoint=True,
        use_rope=False,
        rope_base=10000.0,
        use_flash_attn=False,
    ):
        super().__init__()
        if parameter_channels:
            raise ValueError("MSPT DrivAerML adapter does not use parameter channels.")
        if n_hidden % num_heads != 0:
            raise ValueError(f"n_hidden={n_hidden} must be divisible by num_heads={num_heads}")
        if V <= 0 or V & (V - 1):
            raise ValueError("MSPT V must be a positive power of two.")
        self.spatial_dim = int(spatial_dim)
        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.V = int(V)
        self.Q = int(Q)
        self.chunking_mode = str(chunking_mode).lower()
        if self.chunking_mode not in {"linear", "balltree"}:
            raise ValueError("MSPT chunking_mode must be 'linear' or 'balltree'.")
        self.subsampled_geometry_points = 0

        self.preprocess = MSPTMLP(
            self.spatial_dim,
            int(n_hidden) * 2,
            int(n_hidden),
            n_layers=0,
            act=activation,
            residual=False,
        )
        self.blocks = nn.ModuleList(
            [
                MSPTBlock(
                    num_heads=num_heads,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=activation,
                    mlp_ratio=mlp_ratio,
                    last_layer=index == int(num_blocks) - 1,
                    out_dim=self.surface_channels + self.volume_channels,
                    V=V,
                    Q=Q,
                    attn_pool=attn_pool,
                    use_checkpoint=use_checkpoint,
                    use_rope=use_rope,
                    rope_base=rope_base,
                    use_flash_attn=use_flash_attn,
                )
                for index in range(int(num_blocks))
            ]
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter((1.0 / n_hidden) * torch.rand(n_hidden))

    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _partition(self, positions):
        batch_size, num_points, _ = positions.shape
        if self.chunking_mode == "linear":
            # The official linear path leaves preprocessing unpadded; the
            # attention block pads hidden features only after this MLP.
            identity = torch.arange(num_points, device=positions.device)
            permutation = identity.unsqueeze(0).expand(batch_size, -1)
            return positions, permutation, num_points

        padded_points = math.ceil(num_points / self.V) * self.V
        pad_len = padded_points - num_points
        if pad_len:
            positions = torch.cat(
                [positions, positions.new_zeros(batch_size, pad_len, positions.shape[-1])], dim=1
            )

        permutations = []
        for batch_index in range(batch_size):
            permutations.append(_partition_balltree(positions[batch_index], self.V))
        permutation = torch.stack(permutations, dim=0)
        inverse = torch.empty_like(permutation)
        inverse.scatter_(1, permutation, torch.arange(padded_points, device=positions.device).expand(batch_size, -1))
        chunked = torch.gather(positions, 1, permutation.unsqueeze(-1).expand(-1, -1, positions.shape[-1]))
        return chunked, inverse, num_points

    @staticmethod
    def _restore(features, inverse, original_points):
        restored = torch.gather(features, 1, inverse.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
        return restored[:, :original_points]

    def forward(
        self,
        geo,
        surf_query_pos,
        vol_query_pos,
        params=None,
        geo_log_density=None,
        return_latent=False,
    ):
        del geo_log_density
        if params is not None:
            raise ValueError("MSPT DrivAerML adapter does not use parameter channels.")
        geometry_count = int(geo.shape[1])
        surface_count = int(surf_query_pos.shape[1])
        all_positions = torch.cat([geo, surf_query_pos, vol_query_pos], dim=1)
        chunked_positions, inverse, original_points = self._partition(all_positions)
        features = self.preprocess(chunked_positions)
        features = features + self.placeholder.view(1, 1, -1)

        supernodes = None
        for block in self.blocks:
            features, supernodes = block(features, supernodes)
        outputs = self._restore(features, inverse, original_points)
        query_outputs = outputs[:, geometry_count:]
        pred_surf = query_outputs[:, :surface_count, : self.surface_channels]
        pred_vol = query_outputs[:, surface_count:, self.surface_channels :]
        if return_latent:
            return pred_surf, pred_vol, query_outputs
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        return self.forward(
            geo,
            surf_query_pos,
            vol_query_pos,
            params=params,
            geo_log_density=geo_log_density,
        )

~~~

### MSPT vanilla entry point


~~~python
import hydra
from omegaconf import DictConfig

from models.mspt import MSPT
from utils.surface_volume_trainer import run_surface_volume_training


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_mspt")
def main(cfg: DictConfig):
    run_surface_volume_training(cfg, MSPT, accepts_geo_log_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")

~~~

### MSPT SATLOSS6 entry point


~~~python
import hydra
from omegaconf import DictConfig

from models.mspt import MSPT
from train_consistency_common import run_consistency_training


class MSPTSATLOSS6(MSPT):
    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        return super().forward(
            geo,
            surf_query_pos,
            vol_query_pos,
            params=params,
            geo_log_density=geo_log_density,
            return_latent=return_latent,
        )


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_mspt_satloss6")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=MSPTSATLOSS6, model_requires_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")

~~~
