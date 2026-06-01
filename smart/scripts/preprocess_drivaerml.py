#!/usr/bin/env python3
import argparse
import json
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

    # HDF5 + multiprocessing can stall on some systems. Default to robust streaming mode.
    ok = 0
    status_counts = {"ok": 0, "skipped": 0, "missing": 0}
    try:
        for j in tqdm(jobs, total=len(jobs), desc="Preprocessing runs"):
            rid, status = process_run(j)
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "ok":
                ok += 1
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating immediately.")
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
