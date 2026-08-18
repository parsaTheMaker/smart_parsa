# SMART Vanilla Pipeline: Complete Embedded Backup

This document is a source-level backup of the canonical DrivAerML vanilla SMART
pipeline. It embeds the relevant configuration and implementation modules so a
reader does not need to open another repository file to understand the data
contract, preprocessing, loader, targets, model, loss, optimizer, validation,
checkpointing, and launch protocol. The code blocks below are copied verbatim
from the implementation at the time this backup was created.

## Pipeline Contract

The vanilla path is a one-view geometry-conditioned operator. The raw H5 files
are converted into memory-mapped NPY case folders. Training samples geometry,
surface query points, and volume query points independently, normalizes
coordinates with train-set global bounds, standardizes targets with train-set
channel statistics, predicts seven surface and four volume channels, and
optimizes surface Relative-L2 plus volume Relative-L2. Geometry sampling is
epoch-seeded in the canonical DrivAerML configuration. No density weighting,
second view, or consistency loss is active in this document's vanilla path.

## Self-Contained Execution Order

The embedded modules execute conceptually in this order:

~~~text
raw boundary_<id>.h5 and volume_<id>_filtered.h5
  -> chunked exact-uniform source conversion
  -> run_<id> persistent NPY arrays and meta.json
  -> deterministic train/test split
  -> train-only position and target statistics
  -> memory-mapped DrivAerMLDataset
  -> epoch-seeded geometry sample and independent query samples
  -> global coordinate normalization and target standardization
  -> SMART geometry encoder
  -> SMART query decoder and 11-channel head
  -> surface RelL2 + volume RelL2
  -> AdamW, cosine schedule, AMP, validation
  -> best and last checkpoints with full optimizer state
~~~

## Complete Embedded Implementation

The following sections contain the complete implementation in dependency order.

### Configuration

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

### Preprocessing

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
### Dataset Registry

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
### DrivAerML Loader, Statistics, and Sampling

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
### SMART Network

~~~python
"""SMART: Scalable Mesh-free Aerodynamic Simulations from Raw Geometries using a Transformer-based Surrogate Model

This module contains the implementation of the encoder and decoder blocks used in SMART, as well as the complete SMART model.
Designed for simulating time-independent PDEs over complex 3D geometries, SMART leverages a Transformer-based architecture
to perform simulations using solely inexpensive geometry meshes, eliminating the need for costly surface and volumetric
meshing during inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ModulatedPositionalEmbedding(nn.Module):
    """Embedding layer that applies a modulated sine-cosine positional embedding to the spatial positions. This means 
    that the sine-cosine functions are shifted and scaled based on learned parameters from an MLP for each position.
    This allows the model to adaptively adjust the positional embeddings based on the data to emphasize or suppress
    high-frequency variations.

    The implementation follows the original Transformer positional embedding (https://arxiv.org/abs/1706.03762) and
    the class is based on the PositionalEncoding class from the PyTorch tutorial 'Language Modeling with nn.Transformer and torchtext'.
    
    Args:
        dim: Dimensionality of the embedded positions.
        spatial_dim: The spatial dimensionality of the positions (e.g., 2 for 2D positions, 3 for 3D positions). Defaults to 3.
        max_seq_length: Max sequence length. Defaults to 10000 as suggested in the original Transformer paper.
    """

    def __init__(self, dim, spatial_dim=3, max_seq_length=10000):
        super().__init__()
        self.dim = dim
        self.spatial_dim = spatial_dim
        
        # Compute dimensions per spatial dimension
        max_dim_per_spatial_dim = dim // spatial_dim
        dim_per_spatial_dim = max_dim_per_spatial_dim & ~1 # This is equal to (max_dim_per_spatial_dim // 2) * 2
        self.dim_per_spatial_dim = dim_per_spatial_dim
        
        # Compute the total padding
        self.total_padding = dim - (dim_per_spatial_dim * spatial_dim)
        self.register_buffer("padding", torch.zeros(1, 1, self.total_padding))
        
        # Compute the div_term for sine-cosine embedding
        div_term = torch.exp(torch.arange(0, dim_per_spatial_dim, 2) * (-math.log(max_seq_length) / dim_per_spatial_dim))
        self.register_buffer("div_term", div_term)
        
        # Modulation MLP
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim_per_spatial_dim * spatial_dim * 2))

    def compute_embedding(self, pos, shift_sin=None, scale_sin=None, shift_cos=None, scale_cos=None):
        # Following UPT (https://arxiv.org/abs/2402.12365) and compute positional embeddings in float32 to avoid numerical instabilities
        with torch.autocast(device_type=str(pos.device).split(":")[0], enabled=False):
            pos = pos.float()
            sin_cos_arg = pos[..., None] @ self.div_term[None, ...]
            
            embedding = torch.zeros((*sin_cos_arg.shape[:-1], self.dim_per_spatial_dim), device=sin_cos_arg.device, dtype=sin_cos_arg.dtype)
            # Apply shift and scale to embedding if provided
            if shift_sin is not None and scale_sin is not None and shift_cos is not None and scale_cos is not None:
                embedding[..., 0::2] = scale_sin * torch.sin(sin_cos_arg + shift_sin)
                embedding[..., 1::2] = scale_cos * torch.cos(sin_cos_arg + shift_cos)
            else:
                embedding[..., 0::2] = torch.sin(sin_cos_arg)
                embedding[..., 1::2] = torch.cos(sin_cos_arg)
            
        # Rearrange spatial dimensions
        embedding = rearrange(embedding, "b n spatial_dim d -> b n (spatial_dim d)")
        
        # Apply padding if necessary
        if self.total_padding > 0: embedding = torch.concat([embedding, self.padding.expand(*embedding.shape[:-1], -1)], dim=-1)
        
        return embedding
        
    def forward(self, pos):
        """Embeds the positions, normalized to [0, max_seq_length], using modulated sine-cosine positional embeddings.

        Args:
            pos: Normalized positions with shape (batch size, number points, spatial_dim).

        Returns:
            Embedded positions with shape (batch size, number points, dim).
        """
        initial_embedding = self.compute_embedding(pos)
        
        # Apply modulation MLP for shift and scaling
        shift_scale = self.mlp(initial_embedding)
        shift_sin, scale_sin, shift_cos, scale_cos = torch.unbind(rearrange(shift_scale, "b n (d shift_scale spatial_dim) -> b n spatial_dim d shift_scale", shift_scale=4, spatial_dim=self.spatial_dim), -1)
        
        embedding = self.compute_embedding(pos, shift_sin=shift_sin, scale_sin=scale_sin, shift_cos=shift_cos, scale_cos=scale_cos)
        
        return embedding


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE; https://arxiv.org/abs/2104.09864) for spatial positions.
    
    Args:
        dim: Dimensionality of the features to be embedded.
        spatial_dim: The spatial dimensionality of the positions (e.g., 2 for 2D positions, 3 for 3D positions). Defaults to 3.
        max_seq_length: Max sequence length. Defaults to 10000 as suggested in the original RoPE paper.
    """
    
    def __init__(self, dim, spatial_dim, max_seq_length=10000.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for rotary embeddings"
        
        self.dim = dim
        self.spatial_dim = spatial_dim
        
        # Compute dimensions per spatial dimension
        max_dim_per_spatial_dim = dim // spatial_dim
        dim_per_spatial_dim = max_dim_per_spatial_dim & ~1 # This is equal to (max_dim_per_spatial_dim // 2) * 2
        
        # Compute the padding
        self.total_padding = dim - (dim_per_spatial_dim * spatial_dim)
        self.register_buffer("padding", torch.zeros(1, 1, self.total_padding // 2))
        
        # Compute the div_term for sine-cosine embedding
        div_term = torch.exp(torch.arange(0, dim_per_spatial_dim, 2) * (-math.log(max_seq_length) / dim_per_spatial_dim))
        self.register_buffer("div_term", div_term)

    def forward(self, x, pos):
        """Applies RoPE to the features x based on the positions pos.
        
        Args:
            x: Features to apply RoPE to with shape (batch size, number points, dim).
            pos: Normalized positions with shape (batch size, number points, spatial_dim).
            
        Returns:
            Features with RoPE applied to with shape (batch size, number points, dim).
        """
        # Following UPT (https://arxiv.org/abs/2402.12365) and compute positional embeddings in float32 to avoid numerical instabilities
        with torch.autocast(device_type=str(pos.device).split(":")[0], enabled=False):
            pos = pos.float()
            theta = pos[..., None] @ self.div_term[None, ...]
        
        theta = rearrange(theta, "b n spatial_dim d -> b n (spatial_dim d)")
        
        # Add padding
        theta = torch.concat([theta, self.padding.expand(*theta.shape[:-1], -1)], dim=-1)
        
        # Apply rotation matrix in complex space following Llama 3 implementation
        # (https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py#L65)
        rotation = torch.polar(torch.ones_like(theta), theta)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        embedded = torch.view_as_real(x_complex * rotation[:, None, ...]).flatten(3)
        
        return embedded.type_as(x)
    

class CrossAttention(nn.Module):
    """Computes multi-head cross-attention (https://arxiv.org/abs/1706.03762) between the query and key/value sequences. It
    optionally applies Rotary Positional Embedding (RoPE; https://arxiv.org/abs/2104.09864) to both the query and key features.

    Args:
        dim: Dimensionality of the query and key/value features.
        num_heads: Number of attention heads. Defaults to 8.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        dropout: Dropout rate. Defaults to 0.1.
    """

    def __init__(self, dim, num_heads=8, spatial_dim=3, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Pre-layer normalization
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        
        # Projections
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(dim=dim // num_heads, spatial_dim=spatial_dim)
        
        self.dropout = dropout

    def forward(self, q, kv, q_pos=None, kv_pos=None):
        """Applies pre-norm and computes cross-attention between q and kv.

        Args:
            q: Queries with shape (batch size, number query tokens, dim).
            kv: Key/value with shape (batch size, number key/value tokens, dim).
            q_pos (optional): Positions for the queries with shape (batch size, number query tokens, spatial_dim).
            kv_pos (optional): Positions for the key/value with shape (batch size, num key/value tokens, spatial_dim).

        Returns:
            Updated queries that attend to kv with shape (batch size, number query tokens, dim).
        """
        # Apply layer normalization
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        
        # Linear projections
        q = self.q(q)
        kv = self.kv(kv)
            
        # Split heads and keys/values
        q_heads = rearrange(q, "b q (h d) -> b h q d", h=self.num_heads, d=self.head_dim)
        k_heads, v_heads = torch.tensor_split(rearrange(kv, "b kv (h d) -> b h kv d", h=2*self.num_heads, d=self.head_dim), 2, dim=1)
        
        # Apply RoPE if positions are provided
        if q_pos is not None and kv_pos is not None:
            q_heads = self.rope(q_heads, q_pos)
            k_heads = self.rope(k_heads, kv_pos)
            
        # Compute attention using PyTorch's scaled_dot_product_attention
        x = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=(self.dropout if self.training else 0.0))
        
        # Merge heads and output projection
        x = rearrange(x, "b h q d -> b q (h d)")
        x = self.out_proj(x)
        
        return x


class Modulator(nn.Module):
    """Modulator module for FiLM-like modulation (https://github.com/ethanjperez/film) of features based on conditioning parameters.
    
    Args:
        dim: Dimensionality of the features to be modulated.
        cond_dim: Dimensionality of the conditioning parameters.
        hidden_dim: Dimensionality of the hidden layer in the modulation MLP. Default is 128.
        use_residual: If true, use residual connection in modulation. Default is False.
    """
    
    def __init__(self, dim, cond_dim, hidden_dim=128, use_residual=False):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim * 2))
        self.use_residual = use_residual
        # Keep the original direct FiLM behavior as the default.  SHIFT-Crash
        # opts into a bounded residual mode through its isolated adapter.
        self.conditioning_mode = "direct"
        self.conditioning_residual_scale = 1.0
        self.conditioning_shift_scale = 1.0

    def configure_conditioning(self, mode="direct", residual_scale=1.0, shift_scale=1.0):
        mode = str(mode).lower().strip()
        if mode not in {"direct", "residual", "bounded_residual"}:
            raise ValueError(f"Unsupported conditioning mode: {mode!r}")
        residual_scale = float(residual_scale)
        shift_scale = float(shift_scale)
        if residual_scale < 0.0 or shift_scale < 0.0:
            raise ValueError("Conditioning scales must be non-negative.")
        self.conditioning_mode = mode
        self.conditioning_residual_scale = residual_scale
        self.conditioning_shift_scale = shift_scale

    def forward(self, x, params):
        """Modulates the features x based on the conditioning parameters params.
        
        Args:
            x: Features to be modulated with shape (batch size, number points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
        
        Returns:
            Modulated features with shape (batch size, number points, dim).
        """
        scale, shift = torch.tensor_split(self.mlp(params), 2, dim=-1)
        # The conditioner produces one feature-wise vector per sample, while
        # encoder/decoder activations may contain an additional point/latent
        # axis.  Align that vector with the feature axis before broadcasting.
        # Without this, a batch of two and 512 latent points compares the
        # batch dimension against the latent-point dimension.
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(-2)
            shift = shift.unsqueeze(-2)
        if self.conditioning_mode == "bounded_residual":
            scale = torch.tanh(scale) * self.conditioning_residual_scale
            shift = torch.tanh(shift) * self.conditioning_shift_scale
            return x + scale * x + shift
        if self.conditioning_mode == "residual" or self.use_residual:
            return x + scale * x + shift
        return scale * x + shift


class SimulationParamModulatedMLP(nn.Module):
    """MLP with FiLM-like modulation (https://github.com/ethanjperez/film) based on simulation parameters.
    
    Args:
        dim: Dimensionality of the input and output features.
        hidden_dim: Dimensionality of the hidden layer of the MLP.
        cond_dim: Dimensionality of the conditioning parameters.
        cond_hidden_dim: Width of the conditioning bottleneck.
        dropout: Dropout rate. Default is 0.1.
        use_residual: If true, use residual connection in modulation. Default is False.
    """

    def __init__(self, dim, hidden_dim, cond_dim, cond_hidden_dim=128, dropout=0.1, use_residual=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.non_linearity = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.modulator = Modulator(hidden_dim, cond_dim, hidden_dim=cond_hidden_dim, use_residual=use_residual)
        

    def forward(self, x, params):        
        """Processes the features x with an MLP, modulated based on the conditioning parameters params.
        
        Args:
            x: Features with shape (batch size, number points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
        
        Returns:
            Processed features with shape (batch size, number points, dim).
        """
        x = self.modulator(self.non_linearity(self.linear1(self.norm(x))), params)
        x = self.linear2(x)
        
        return x


class PlainMLP(nn.Module):
    """Plain multi-layer perceptron (MLP) **without** modulation
    
    Args:
        dim: Dimensionality of the input and output features.
        hidden_dim: Dimensionality of the hidden layer of the MLP.
        dropout: Dropout rate. Default is 0.1.
    """

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.non_linearity = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        

    def forward(self, x, params):
        """Processes the features x with an MLP without modulation.
            
        Args:
            x: Features with shape (batch size, number points, dim).
            params: Conditioning parameters will be ignored.
        
        Returns:
            Processed features with shape (batch size, number points, dim).
        """
        x = self.linear2(self.non_linearity(self.linear1(self.norm(x))))
        
        return x


class EncoderBlock(nn.Module):
    """The encoder block updates the latent geometry by attending to a cross-attended version of itself. This means that
    first, the latent geometry attends to a subsampled version of the input geometry to integrate geometric information,
    and then it attends to this cross-attended version of itself to refine the latent representation.

    Args:
        dim: Dimensionality of the features.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout rate. Defaults to 0.1.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        cond_dim: Dimensionality of the conditioning parameters for the MLP. Defaults to 2.
        conditioning_hidden_dim: Width of the conditioning bottleneck. Defaults to 128.
        residual_update_scale: Multiplier applied to each residual update. Defaults to 1.
        normalize_residuals: If true, normalize block outputs after residual updates.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.1,
        spatial_dim=3,
        cond_dim=2,
        conditioning_hidden_dim=128,
        residual_update_scale=1.0,
        normalize_residuals=False,
    ):
        super().__init__()
        residual_update_scale = float(residual_update_scale)
        if not math.isfinite(residual_update_scale) or residual_update_scale <= 0.0:
            raise ValueError("residual_update_scale must be a finite positive number.")
        self.residual_update_scale = residual_update_scale
        self.normalize_residuals = bool(normalize_residuals)
        self.geo_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.cross_output_norm = nn.LayerNorm(dim, eps=1e-6) if self.normalize_residuals else nn.Identity()
        self.output_norm = nn.LayerNorm(dim, eps=1e-6) if self.normalize_residuals else nn.Identity()
        
        # Pointwise MLP
        if cond_dim > 0:
            self.mlp = SimulationParamModulatedMLP(
                dim=dim,
                hidden_dim=dim * 4,
                cond_dim=cond_dim,
                cond_hidden_dim=conditioning_hidden_dim,
                dropout=dropout,
            )
        else:
            self.mlp = PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)
       
    def forward(self, latent_geometry, subsampled_geometry, params, latent_geometry_pos=None, subsampled_geometry_pos=None):
        """Updates the latent geometry by attending to a cross-attended version of itself that first attends to a subsampled
        version of the input geometry.

        Args:
            latent_geometry: Latent geometry with shape (batch size, number latent points, dim).
            subsampled_geometry: Subsampled input geometry with shape (batch size, number subsampled points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
            latent_geometry_pos (optional): Positions of the latent geometry for the positional embeddings with shape (batch size, number latent points, spatial_dim). Defaults to None.
            subsampled_geometry_pos (optional): Positions of the subsampled input geometry for the positional embeddings with shape (batch size, number subsampled points, spatial_dim) . Defaults to None.

        Returns:
            tuple: A tuple containing:
                - Updated latent geometry with shape (batch size, number latent points, dim).
                - Latent geometry after geometry cross-attention and before cross-attention and MLP with shape (batch size, number latent points, dim).
        """
        # First cross-attention with the subsampled geometry
        latent_geometry_cross = latent_geometry + self.residual_update_scale * self.attn_dropout(
            self.geo_attn(q=latent_geometry, kv=subsampled_geometry, q_pos=latent_geometry_pos, kv_pos=subsampled_geometry_pos)
        )
        latent_geometry_cross = self.cross_output_norm(latent_geometry_cross)

        # Update the initial latent geometry by attending to the cross-attended version of itself
        latent_geometry_self = latent_geometry + self.residual_update_scale * self.attn_dropout(
            self.cross_attn(q=latent_geometry, kv=latent_geometry_cross, q_pos=latent_geometry_pos, kv_pos=latent_geometry_pos)
        )
        
        # Pointwise MLP
        latent_geometry_mlp = latent_geometry_self + self.residual_update_scale * self.mlp(latent_geometry_self, params)
        
        return self.output_norm(latent_geometry_mlp), latent_geometry_cross


class DecoderBlock(nn.Module):
    """The decoder block attends to the latent geometry of the corresponding encoder block to produce predictions
    of physical quantities at query positions.

    Args:
        dim: Dimensionality of the features.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout rate. Defaults to 0.1.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        cond_dim: Dimensionality of the conditioning parameters for the MLP. Defaults to 2.
        conditioning_hidden_dim: Width of the conditioning bottleneck. Defaults to 128.
        residual_update_scale: Multiplier applied to each residual update. Defaults to 1.
        normalize_residuals: If true, normalize block outputs after residual updates.
        shared_attn: Shared cross-attention module from the corresponding encoder block. Defaults to None.
        shared_mlp: Shared MLP module from the corresponding encoder block. Defaults to None.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.1,
        spatial_dim=3,
        cond_dim=2,
        conditioning_hidden_dim=128,
        residual_update_scale=1.0,
        normalize_residuals=False,
        shared_attn=None,
        shared_mlp=None,
    ):
        super().__init__()
        residual_update_scale = float(residual_update_scale)
        if not math.isfinite(residual_update_scale) or residual_update_scale <= 0.0:
            raise ValueError("residual_update_scale must be a finite positive number.")
        self.residual_update_scale = residual_update_scale
        self.normalize_residuals = bool(normalize_residuals)
        self.attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim) if shared_attn is None else shared_attn
        self.attn_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(dim, eps=1e-6) if self.normalize_residuals else nn.Identity()
        
        # Pointwise MLP
        if cond_dim > 0:
            self.mlp = (
                SimulationParamModulatedMLP(
                    dim=dim,
                    hidden_dim=dim * 4,
                    cond_dim=cond_dim,
                    cond_hidden_dim=conditioning_hidden_dim,
                    dropout=dropout,
                )
                if shared_mlp is None
                else shared_mlp
            )
        else:
            self.mlp = PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout) if shared_mlp is None else shared_mlp
       
    def forward(self, queries, latent_geometry, params, queries_pos=None, latent_geometry_pos=None):
        """Updates the queries by attending to the latent geometry of the corresponding encoder block.

        Args:
            queries: Features of the query positions with shape (batch size, number query points, dim).
            latent_geometry: Latent geometry with shape (batch size, number latent points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
            queries_pos (optional): Positions of the query positions for the positional embeddings with shape (batch size, number query points, spatial_dim). Defaults to None.
            latent_geometry_pos (optional): Positions of the latent geometry for the positional embeddings with shape (batch size, number latent points, spatial_dim). Defaults to None.
        
        Returns:
            Updated queries with shape (batch size, number query points, dim).
        """
        # Cross-attention with the latent geometry
        queries = queries + self.residual_update_scale * self.attn_dropout(
            self.attn(q=queries, kv=latent_geometry, q_pos=queries_pos, kv_pos=latent_geometry_pos)
        )
        
        # Pointwise MLP
        queries = queries + self.residual_update_scale * self.mlp(queries, params)
        
        return self.output_norm(queries)


def sample_geometry(geometry, num_samples, with_replacement=False):
    """Samples points from the input geometry.

    Args:
        geometry: Input geometry with shape (batch size, number points, spatial_dim).
        num_samples: Number of points to sample.
    
    Returns:
        Sampled input geometry with shape (batch size, num_samples, spatial_dim).
    """
    n_points = int(geometry.shape[1])
    batch_size = int(geometry.shape[0])
    if num_samples <= 0:
        return geometry
    if with_replacement:
        idx = torch.randint(0, n_points, (batch_size, num_samples), device=geometry.device, dtype=torch.long)
    else:
        if num_samples >= n_points:
            return geometry
        idx = torch.stack(
            [torch.randperm(n_points, device=geometry.device)[:num_samples] for _ in range(batch_size)],
            dim=0,
        )
    sampled_geometry = torch.gather(geometry, 1, idx.unsqueeze(-1).expand(-1, -1, geometry.shape[-1]))
    return sampled_geometry


def sample_geometry_indices(geometry, num_samples, with_replacement=False):
    """Return the point indices used by ``sample_geometry``."""
    n_points = int(geometry.shape[1])
    batch_size = int(geometry.shape[0])
    if num_samples <= 0 or num_samples >= n_points:
        return torch.arange(n_points, device=geometry.device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
    if with_replacement:
        return torch.randint(0, n_points, (batch_size, num_samples), device=geometry.device, dtype=torch.long)
    return torch.stack(
        [torch.randperm(n_points, device=geometry.device)[:num_samples] for _ in range(batch_size)],
        dim=0,
    )


def gather_point_values(values, indices):
    """Gather [B,N] or [B,N,C] node attributes using [B,K] indices."""
    if values is None:
        return None
    if values.ndim == 2:
        return torch.gather(values, 1, indices)
    return torch.gather(values, 1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]))

   
class SMART(nn.Module):
    """SMART model for simulating time-independent PDEs over complex 3D geometries.
    
    Args:
        spatial_dim: Number of spatial dimensions. Default is 3.
        surface_channels: Number of output channels for surface predictions. Default is 1.
        volume_channels: Number of output channels for volume predictions. Default is 3.
        parameter_channels: Number of conditioning parameter channels. Default is 2.
        latent_dim: Dimensionality of the latent representations. Default is 256.
        latent_geometry_points: Number of points of the latent geometry. Default is 4096.
        subsampled_geometry_points: Number of points in the subsampled geometry for geometry cross-attention. Default is 16384.
        num_encoder_decoder_blocks: Number of encoder-decoder blocks. Default is 8.
        num_heads: Number of attention heads. Default is 8.
        pos_scale_factor: Scaling factor for the positions to use more/less of the dynamic range of the positional embedding. Default is 1000.
        dropout: Dropout rate. Default is 0.0.
        subregion_size: Number of query points to process in each subregion during sequential inference. Default is 262144.
        conditioning_hidden_dim: Width of each FiLM conditioning bottleneck. Default is 128.
        residual_update_scale: Multiplier applied to each residual update. Default is 1.
        normalize_residuals: If true, normalize encoder and decoder block outputs.
    """

    def __init__(self, spatial_dim=3,
                 surface_channels=1,
                 volume_channels=3,
                 parameter_channels=2,
                 latent_dim=256,
                 latent_geometry_points=4096,
                 subsampled_geometry_points=16384,
                 num_encoder_decoder_blocks=8, 
                 num_heads=8,
                 pos_scale_factor=1000,
                 dropout=0.0,
                 subregion_size=262144,
                 subsampled_geometry_with_replacement=False,
                 conditioning_hidden_dim=128,
                 residual_update_scale=1.0,
                 normalize_residuals=False,
                 geometry_feature_channels=0,
                 query_feature_channels=0,
                 part_embedding_size=0,
                 part_embedding_dim=16):
        super(SMART, self).__init__()
        assert surface_channels > 0 and volume_channels > 0, "surface_channels and volume_channels must be positive integers."
        
        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.num_geo = latent_geometry_points
        self.subsampled_geometry_points = subsampled_geometry_points
        self.subsampled_geometry_with_replacement = bool(subsampled_geometry_with_replacement)
        self.pos_scale_factor = pos_scale_factor
        self.conditioning_hidden_dim = int(conditioning_hidden_dim)
        if self.conditioning_hidden_dim <= 0:
            raise ValueError("conditioning_hidden_dim must be positive.")
        self.residual_update_scale = float(residual_update_scale)
        self.normalize_residuals = bool(normalize_residuals)
        self.geometry_feature_channels = int(geometry_feature_channels)
        self.query_feature_channels = int(query_feature_channels)
        self.part_embedding_size = int(part_embedding_size)
        self.part_embedding_dim = int(part_embedding_dim)
        if self.geometry_feature_channels < 0 or self.query_feature_channels < 0:
            raise ValueError("Feature channel counts must be non-negative.")
        if self.geometry_feature_channels != self.query_feature_channels:
            raise ValueError("Geometry and query feature widths must match for shared SMART feature encoding.")
        if self.part_embedding_size < 0 or self.part_embedding_dim <= 0:
            raise ValueError("Invalid part embedding configuration.")
        self.point_feature_encoder = None
        if self.geometry_feature_channels > 0 or self.part_embedding_size > 0:
            if self.geometry_feature_channels <= 0 or self.part_embedding_size <= 0:
                raise ValueError("Both continuous feature channels and part embedding size are required together.")
            self.point_feature_norm = nn.LayerNorm(self.geometry_feature_channels, eps=1.0e-6)
            self.point_feature_projection = nn.Sequential(
                nn.Linear(self.geometry_feature_channels, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
            self.part_embedding = nn.Embedding(self.part_embedding_size, self.part_embedding_dim, padding_idx=0)
            self.part_projection = nn.Linear(self.part_embedding_dim, latent_dim, bias=False)
            self.point_feature_scale = nn.Parameter(torch.tensor(0.1))
            self.part_feature_scale = nn.Parameter(torch.tensor(0.1))
            self.point_feature_encoder = nn.Identity()
        self.pos_encoder = ModulatedPositionalEmbedding(latent_dim, spatial_dim)
        
        # Encoder and decoder blocks
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    spatial_dim=spatial_dim,
                    cond_dim=parameter_channels,
                    conditioning_hidden_dim=self.conditioning_hidden_dim,
                    residual_update_scale=self.residual_update_scale,
                    normalize_residuals=self.normalize_residuals,
                )
                for _ in range(num_encoder_decoder_blocks)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    spatial_dim=spatial_dim,
                    cond_dim=parameter_channels,
                    conditioning_hidden_dim=self.conditioning_hidden_dim,
                    residual_update_scale=self.residual_update_scale,
                    normalize_residuals=self.normalize_residuals,
                    shared_attn=self.encoder_blocks[i].cross_attn,
                    shared_mlp=self.encoder_blocks[i].mlp,
                )
                for i in range(num_encoder_decoder_blocks)
            ]
        )
        
        # Final MLP
        self.mlp = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(),
                                 nn.Linear(128, 64), nn.GELU(),
                                 nn.Linear(64, surface_channels+volume_channels))
        
        # Subregion size for inference
        self.subregion_size = subregion_size
    
    def initialize_weights(self):
        self.apply(self._init_weights)

    # Weight initialization from Transolver
    # (https://github.com/thuml/Transolver/blob/a11be9c4f7db1885e4b08c68432bc31799492ec9/Car-Design-ShapeNetCar/models/Transolver.py#L168)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _encode_point_features(self, features, part_ids):
        if self.point_feature_encoder is None:
            return None
        if features is None or part_ids is None:
            raise ValueError("Configured SMART point features require both continuous features and part IDs.")
        if features.shape[-1] != self.geometry_feature_channels:
            raise ValueError(
                f"Expected {self.geometry_feature_channels} continuous point features, got {features.shape[-1]}."
            )
        part_ids = part_ids.long().clamp(0, self.part_embedding_size - 1)
        continuous = self.point_feature_projection(self.point_feature_norm(features.float()))
        categorical = self.part_projection(self.part_embedding(part_ids))
        return self.point_feature_scale * continuous + self.part_feature_scale * categorical

    def encode(self, geo, params, geometry_features=None, geometry_part_ids=None, return_final=False):
        # Prepare positions by scaling
        geo = geo * self.pos_scale_factor
        
        # Sample the initial latent geometry
        latent_idx = sample_geometry_indices(geo, self.num_geo)
        latent_geo_pos = gather_point_values(geo, latent_idx)
        latent_geo_emb = self.pos_encoder(latent_geo_pos)
        if self.point_feature_encoder is not None:
            latent_geo_emb = latent_geo_emb + self._encode_point_features(
                gather_point_values(geometry_features, latent_idx),
                gather_point_values(geometry_part_ids, latent_idx),
            )
        
        # Apply encoder blocks
        intermediate_latent_geometries = []
        for block in self.encoder_blocks:
            # Subsample the geometry for geometry cross-attention
            sub_idx = sample_geometry_indices(
                geo, self.subsampled_geometry_points, with_replacement=self.subsampled_geometry_with_replacement
            )
            sub_geo_pos = gather_point_values(geo, sub_idx)
            sub_geo_emb = self.pos_encoder(sub_geo_pos)
            if self.point_feature_encoder is not None:
                sub_geo_emb = sub_geo_emb + self._encode_point_features(
                    gather_point_values(geometry_features, sub_idx),
                    gather_point_values(geometry_part_ids, sub_idx),
                )
            
            # Apply encoder block
            latent_geo_emb, e_ca = block(latent_geo_emb, sub_geo_emb, params, latent_geometry_pos=latent_geo_pos, subsampled_geometry_pos=sub_geo_pos)
            
            # Store for decoder
            intermediate_latent_geometries.append(e_ca)
        
        if return_final:
            return intermediate_latent_geometries, latent_geo_pos, latent_geo_emb
        return intermediate_latent_geometries, latent_geo_pos
    
    def decode_features(
        self,
        intermediate_latent_geometries,
        latent_geo_pos,
        params,
        query_pos,
        query_features=None,
        query_part_ids=None,
    ):
        # Prepare positions by scaling
        query_pos = query_pos * self.pos_scale_factor
        query_emb = self.pos_encoder(query_pos)
        if self.point_feature_encoder is not None:
            query_emb = query_emb + self._encode_point_features(query_features, query_part_ids)
        
        # Apply decoder blocks
        for e_ca, block in zip(intermediate_latent_geometries, self.decoder_blocks):
            query_emb = block(query_emb, e_ca, params, queries_pos=query_pos, latent_geometry_pos=latent_geo_pos)
        
        return query_emb

    def decode(self, intermediate_latent_geometries, latent_geo_pos, params, query_pos, query_features=None, query_part_ids=None):
        query_emb = self.decode_features(
            intermediate_latent_geometries,
            latent_geo_pos,
            params,
            query_pos,
            query_features=query_features,
            query_part_ids=query_part_ids,
        )
        pred = self.mlp(query_emb)
        return pred

    def forward(
        self,
        geo,
        surf_query_pos,
        vol_query_pos,
        params,
        geometry_features=None,
        query_features=None,
        geometry_part_ids=None,
        query_part_ids=None,
    ):
        """Forward method for SMART model.
        
        Args:
            geo: Input geometry with shape (batch size, number points, spatial_dim).
            surf_query_pos: Surface query positions with shape (batch size, number surface query points, spatial_dim).
            vol_query_pos: Volume query positions with shape (batch size, number volume query points, spatial_dim).
            params: Conditioning parameters with shape (batch size, cond_dim). If not used, pass None.
        
        Returns:
            tuple: A tuple containing:
                - Surface predictions with shape (batch size, number surface query points, surface_channels).
                - Volume predictions with shape (batch size, number volume query points, volume_channels).
        """
        # Encode
        intermediate_latent_geometries, latent_geo_pos = self.encode(
            geo, params, geometry_features=geometry_features, geometry_part_ids=geometry_part_ids
        )
        
        # Prepare query positions by concatenating surface and volume query positions
        query_pos = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        
        # Decode
        pred = self.decode(
            intermediate_latent_geometries,
            latent_geo_pos,
            params,
            query_pos,
            query_features=query_features,
            query_part_ids=query_part_ids,
        )
        
        # Split surface and volume predictions
        pred_surf = pred[:, :surf_query_pos.shape[1], 0:self.surface_channels]
        pred_vol = pred[:, surf_query_pos.shape[1]:, self.surface_channels:]
        
        return pred_surf, pred_vol
    
    @torch.inference_mode()
    def inference(
        self,
        geo,
        surf_query_pos,
        vol_query_pos,
        params,
        geometry_features=None,
        query_features=None,
        geometry_part_ids=None,
        query_part_ids=None,
        volume_query_features=None,
        volume_query_part_ids=None,
    ):
        """Sequential inference method to handle large number of query points that may not fit into GPU memory.
        
        Args:
            geo: Input geometry with shape (batch size, number points, spatial_dim).
            surf_query_pos: Surface query positions with shape (batch size, number surface query points, spatial_dim).
            vol_query_pos: Volume query positions with shape (batch size, number volume query points, spatial_dim).
            params: Conditioning parameters with shape (batch size, cond_dim). If not used, pass None.
        
        Returns:
            tuple: A tuple containing:
                - Surface predictions with shape (batch size, number surface query points, surface_channels).
                - Volume predictions with shape (batch size, number volume query points, volume_channels).
        """
        # Encode
        intermediate_latent_geometries, latent_geo_pos = self.encode(
            geo, params, geometry_features=geometry_features, geometry_part_ids=geometry_part_ids
        )
        
        # Surface predictions sequentially
        N_surf = surf_query_pos.shape[1]
        y_hat_surf_subregions = []
        for i in range(0, N_surf, self.subregion_size):
            surf_subregion = surf_query_pos[:, i:i+self.subregion_size, :]
            surf_features = None if query_features is None else query_features[:, i:i+self.subregion_size, :]
            surf_part_ids = None if query_part_ids is None else query_part_ids[:, i:i+self.subregion_size]
            y_surf_subregion = self.decode(
                intermediate_latent_geometries,
                latent_geo_pos,
                params,
                surf_subregion,
                query_features=surf_features,
                query_part_ids=surf_part_ids,
            )
            y_hat_surf_subregions.append(y_surf_subregion)
        y_hat_surf = torch.cat(y_hat_surf_subregions, dim=1)

        # Volume predictions sequentially
        N_vol = vol_query_pos.shape[1]
        y_hat_vol_subregions = []
        for i in range(0, N_vol, self.subregion_size):
            vol_subregion = vol_query_pos[:, i:i+self.subregion_size, :]
            vol_features = None if volume_query_features is None else volume_query_features[:, i:i+self.subregion_size, :]
            vol_part_ids = None if volume_query_part_ids is None else volume_query_part_ids[:, i:i+self.subregion_size]
            y_vol_subregion = self.decode(
                intermediate_latent_geometries,
                latent_geo_pos,
                params,
                vol_subregion,
                query_features=vol_features,
                query_part_ids=vol_part_ids,
            )
            y_hat_vol_subregions.append(y_vol_subregion)
        y_hat_vol = torch.cat(y_hat_vol_subregions, dim=1)
        
        # Split surface and volume predictions
        pred_surf = y_hat_surf[:, :, 0:self.surface_channels]
        pred_vol = y_hat_vol[:, :, self.surface_channels:]
        
        return pred_surf, pred_vol

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
### Training Utilities

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
### Vanilla Trainer

~~~python
import hydra
from omegaconf import DictConfig

import os
import torch
import numpy as np
import wandb
from timeit import default_timer
from tqdm.auto import tqdm

# Dataset and loss functions
from data.datasets import get_dataset
from utils.utils import initialize_gpu, initialize_wandb, get_model_checkpoint_name, count_model_params, get_optimizer_scheduler_loss, apply_naca4_auto_point_budget, print_point_budget
from loss.losses import CombinedLoss

# SMART Model
from models.smart.smart import SMART

CANON_SURF_FIELDS = ["pressure", "normal_x", "normal_y"]
CANON_VOL_FIELDS = ["pressure", "sdf", "velocity_x", "velocity_y"]


def init_metric_dict(surface_fields, volume_fields):
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
    return metrics


def accumulate_channel_metrics(metrics, prefix, pred, gt, field_names, rel_l2_loss_fn, batch_size):
    for channel_idx, field_name in enumerate(field_names):
        channel_loss = rel_l2_loss_fn(pred[..., channel_idx:channel_idx + 1], gt[..., channel_idx:channel_idx + 1])
        metrics[f"{prefix}_{field_name}"] += channel_loss.item() * batch_size


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



@hydra.main(version_base="1.2", config_path="config", config_name="car")
def main(cfg: DictConfig):
    # Extract config
    config = cfg.experiment
    wandb_config = cfg.wandb
    
    # Initialize WandB
    run = initialize_wandb(config, wandb_config)
    
    # Set seed and GPU settings
    device = initialize_gpu(config.random_seed, high_precision=False)
    
    # Set gradient norm clipping, precision and amp
    gradient_norm = config.gradient_norm
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(config.precision, torch.float16)
    amp = config.amp
    print(gradient_norm, amp, dtype)

    # Load data
    train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)

    def apply_vanilla_smart_field_subset():
        nonlocal fields, surf_channels, vol_channels
        # Vanilla SMART on NACA4: surface pressure + volume velocity only.
        if config.model_name == "SMART" and config.dataset == "NACA4":
            fields = {"surface": ["pressure"], "volume": ["pressure", "velocity_x", "velocity_y"]}
            # Supervise surface pressure and volume pressure/velocity for apples-to-apples CAT comparison.
            surf_channels = 1
            vol_channels = 3

    apply_vanilla_smart_field_subset()
    print(f"[SMART] training signals -> surface: {fields['surface']} | volume: {fields['volume']}")

    point_info = apply_naca4_auto_point_budget(config, train_data, for_cat=False)
    if point_info is not None:
        print_point_budget("SMART", point_info)
        # Rebuild datasets with the resolved effective point counts.
        train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
        apply_vanilla_smart_field_subset()
        print(f"[SMART] training signals -> surface: {fields['surface']} | volume: {fields['volume']}")

    use_surface_supervision = len(fields["surface"]) > 0
    set_dataset_epoch(train_data, 0)
    set_dataset_epoch(test_data, 0)

    prefetch_factor = int(getattr(config, "prefetch_factor", 2))
    pin_memory = bool(getattr(config, "pin_memory", True))
    dl_common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    if config.num_workers > 0:
        dl_common["prefetch_factor"] = prefetch_factor
        dl_common["persistent_workers"] = True

    train_loader = torch.utils.data.DataLoader(
        train_data,
        shuffle=True,
        **dl_common,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        **dl_common,
    )
    # Move stats to device
    mean_surf = stats[0][:surf_channels].to(device)
    std_surf = stats[1][:surf_channels].to(device)
    if config.model_name == "SMART" and config.dataset == "NACA4" and vol_channels == 2:
        mean_vol = stats[2][2:4].to(device)
        std_vol = stats[3][2:4].to(device)
    elif config.model_name == "SMART" and config.dataset == "NACA4" and vol_channels == 3:
        mean_vol = torch.stack([stats[2][0], stats[2][2], stats[2][3]]).to(device)
        std_vol = torch.stack([stats[3][0], stats[3][2], stats[3][3]]).to(device)
    else:
        mean_vol = stats[2][:vol_channels].to(device)
        std_vol = stats[3][:vol_channels].to(device)
    
    if bool(getattr(config, "inspect_first_sample", False)):
        # Optional sample inspection: useful for debugging, but can add startup I/O.
        sample = train_data[0]
        if params_dim > 0:
            sample_geo_mesh, sample_surf_mesh, sample_surf_data, sample_vol_mesh, sample_vol_data, params = sample
        else:
            sample_geo_mesh, sample_surf_mesh, sample_surf_data, sample_vol_mesh, sample_vol_data = sample
            params = None
        print("Sample geo_mesh shape:", sample_geo_mesh.shape)
        print("Sample surf_mesh shape:", sample_surf_mesh.shape)
        print("Sample surface fields shape:", sample_surf_data.shape, "fields:", fields["surface"])
        print("Sample vol_mesh shape:", sample_vol_mesh.shape)
        print("Sample volume fields shape:", sample_vol_data.shape, "fields:", fields["volume"])
        if params is not None:
            print("Sample params shape:", params.shape)
    
    # Create model
    models = {"SMART": (SMART, {"spatial_dim": spatial_dim, "surface_channels": surf_channels, "volume_channels": vol_channels, "parameter_channels": params_dim})}
    
    if config.model_name in models:
        merged_kwargs = {**models[config.model_name][1], **config.architecture} if "architecture" in config else models[config.model_name][1]
        print(f"Model kwargs: {merged_kwargs}")
        model = models[config.model_name][0](**merged_kwargs).to(device)
    else:
        raise ValueError("Unknown model class name!")
    model = model.to(device)

    print(f"Total parameters: {count_model_params(model)}")
    model_checkpoint_name = get_model_checkpoint_name(config)
    print(f"Checkpoint name: {model_checkpoint_name}")
    
    # Monitor gradients and parameters with wandb
    run.watch(model, log="all")

    # Training and evaluation
    scaler = torch.amp.GradScaler("cuda")
    optimizer, scheduler, loss_fn, rel_l2_loss_fn = get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=1)
    combined_loss_fn = CombinedLoss(loss_fn, fields) if use_surface_supervision else None
        
    # Training loop
    loss_test_min = np.inf
    global_step = 0
    log_every_n_steps = getattr(config, "log_every_n_steps", 10)
    interrupted = False

    try:
        for ep in tqdm(range(config.epochs), desc="Epochs", dynamic_ncols=True):
            t1 = default_timer()
            set_dataset_epoch(train_data, ep)
            set_dataset_epoch(test_data, 0)
            train_losses = init_metric_dict(fields["surface"], fields["volume"])
            test_losses = init_metric_dict(fields["surface"], fields["volume"])

            model.train()
            train_pbar = tqdm(train_loader, desc=f"Train {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
            for batch_idx, batch in enumerate(train_pbar):
                # b, n, c
                if params_dim > 0:
                    geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params = batch
                    params = params.to(device)
                else:
                    geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
                    params = None

                # Move to device
                geo_mesh = geo_mesh.to(device)
                surf_mesh = surf_mesh.to(device)
                surf_data = surf_data.to(device)
                vol_mesh = vol_mesh.to(device)
                vol_data = vol_data.to(device)

                if config.model_name == "SMART" and config.dataset == "NACA4":
                    surf_data = surf_data[..., :1]
                    vol_data = torch.cat([vol_data[..., :1], vol_data[..., 2:4]], dim=-1)

                # Forward pass
                optimizer.zero_grad()

                if amp:
                    with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                        y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)

                        # Rel l2 loss
                        if use_surface_supervision:
                            loss = combined_loss_fn(y_hat_surf, y_hat_vol, surf_data, vol_data)
                        else:
                            loss = loss_fn(y_hat_vol, vol_data)

                        scaler.scale(loss).backward()
                        if gradient_norm is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                else:
                    y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)

                    # Rel l2 loss
                    if use_surface_supervision:
                        loss = combined_loss_fn(y_hat_surf, y_hat_vol, surf_data, vol_data)
                    else:
                        loss = loss_fn(y_hat_vol, vol_data)
                    loss.backward()

                    # Gradient clipping
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)

                    optimizer.step()
                    scheduler.step()

                # Metrics
                batch_size = surf_data.size(0)
                train_losses["loss"] += loss.item() * batch_size
                with torch.no_grad():
                    if use_surface_supervision:
                        surface_loss = rel_l2_loss_fn(y_hat_surf, surf_data)
                    else:
                        surface_loss = torch.tensor(0.0, device=device)
                    volume_loss = rel_l2_loss_fn(y_hat_vol, vol_data)
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
                if batch_idx % log_every_n_steps == 0 or batch_idx == len(train_loader) - 1:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/batch_rel_l2": (surface_loss + volume_loss).item(),
                        "train/batch_rel_l2_surf": surface_loss.item(),
                        "train/batch_rel_l2_vol": volume_loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": ep,
                    }, step=global_step)
                    train_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # Evaluation
            model.eval()
            test_pbar = tqdm(test_loader, desc=f"Eval  {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
            with torch.no_grad():
                for batch in test_pbar:
                    # b, n, c
                    if params_dim > 0:
                        geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params = batch
                        params = params.to(device)
                    else:
                        geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
                        params = None

                    # Move to device
                    geo_mesh = geo_mesh.to(device)
                    surf_mesh = surf_mesh.to(device)
                    surf_data = surf_data.to(device)
                    vol_mesh = vol_mesh.to(device)
                    vol_data = vol_data.to(device)

                    if config.model_name == "SMART" and config.dataset == "NACA4":
                        surf_data = surf_data[..., :1]
                        vol_data = torch.cat([vol_data[..., :1], vol_data[..., 2:4]], dim=-1)

                    # Forward pass
                    if amp:
                        with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                            y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)
                    else:
                        y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)

                    # Denormalize
                    if use_surface_supervision:
                        pred_surf = y_hat_surf[..., :] * std_surf + mean_surf
                        gt_surf = surf_data * std_surf + mean_surf
                    pred_vol = y_hat_vol[..., :] * std_vol + mean_vol
                    gt_vol = vol_data * std_vol + mean_vol

                    # Metrics
                    batch_size = surf_data.size(0)

                    # Combine loss
                    if use_surface_supervision:
                        batch_loss = combined_loss_fn(y_hat_surf, y_hat_vol, surf_data, vol_data)
                        surface_rel_l2 = rel_l2_loss_fn(y_hat_surf, surf_data)
                    else:
                        batch_loss = loss_fn(y_hat_vol, vol_data)
                        surface_rel_l2 = torch.tensor(0.0, device=device)
                    test_losses["loss"] += batch_loss.item() * batch_size

                    volume_rel_l2 = rel_l2_loss_fn(y_hat_vol, vol_data)
                    test_losses["rel_l2_surf"] += surface_rel_l2.item() * batch_size
                    test_losses["rel_l2_vol"] += volume_rel_l2.item() * batch_size
                    test_losses["rel_l2"] += (surface_rel_l2 + volume_rel_l2).item() * batch_size

                    if use_surface_supervision:
                        accumulate_channel_metrics(test_losses, "rel_l2_surf", pred_surf, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size)
                    accumulate_channel_metrics(test_losses, "rel_l2_vol", pred_vol, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size)

                    test_pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

            # Divide by total number of samples to get mean
            for loss_name in train_losses.keys():
                train_losses[loss_name] /= len(train_loader.dataset)
            for loss_name in test_losses.keys():
                test_losses[loss_name] /= len(test_loader.dataset)

            # Store best run
            if test_losses["rel_l2"] < loss_test_min:
                loss_test_min = test_losses["rel_l2"]
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": test_losses["loss"],
                    "rel_l2_loss": test_losses["rel_l2"],
                    "surface_fields": fields["surface"],
                    "volume_fields": fields["volume"],
                    "metric_values": {k: v for k, v in test_losses.items() if k.startswith("rel_l2")},
                    }, "checkpoints/" + model_checkpoint_name + "_best.pt")
            # Store last run
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": test_losses["loss"],
                "rel_l2_loss": test_losses["rel_l2"],
                "surface_fields": fields["surface"],
                "volume_fields": fields["volume"],
                "metric_values": {k: v for k, v in test_losses.items() if k.startswith("rel_l2")},
                }, "checkpoints/" + model_checkpoint_name + "_last.pt")

            t2 = default_timer()
            print(f"epoch: {ep}, t2-t1 (epoch time): {t2-t1:.5f}, train loss: {train_losses['loss']:.5f}, test loss: {test_losses['loss']:.5f}")
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

    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user (Ctrl+C). Saving current state and exiting cleanly...")
        try:
            emergency_state = {
                "epoch": locals().get("ep", -1),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(emergency_state, "checkpoints/" + model_checkpoint_name + "_last.pt")
            print("Saved the latest checkpoint before exiting.")
        except Exception as exc:
            print(f"Could not save an emergency checkpoint: {exc}")
    finally:
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


if __name__ == "__main__":
    main()
    print("Training done.")

~~~
