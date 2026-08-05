#!/usr/bin/env python3
"""Create a geometry-only DrivAerML domain split using statistical shape analysis.

This script deliberately does not use a neural encoder, CFD fields, targets,
SMART latents, or model errors. Each run is represented as a distribution of
surface points in one common coordinate system. The representation combines:

* projection quantiles across deterministic 3-D directions (a compact,
  permutation-invariant sliced-Wasserstein-style signature),
* binary multiscale voxel occupancy,
* axis/radius/pair-distance quantiles, and
* low-order coordinate moments.

Several statistical clusterers are compared. The selected method is chosen
from geometry-only stability and internal separation, never from CFD results.
The resulting JSON contains both cluster directions so either cluster can be
used for SATLOSS7 training and the other for cross-domain testing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


DEFAULT_ROOT = "/mnt/ssdraid/parsa/drivaerml_preprocessed"
DEFAULT_OUTPUT = "/home/parsa/smart_parsa/results/drivaerml_geometry_statistical_split"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-points", type=int, default=8192)
    parser.add_argument("--projection-count", type=int, default=48)
    parser.add_argument("--projection-quantiles", type=int, default=32)
    parser.add_argument("--pair-samples", type=int, default=4096)
    parser.add_argument("--voxel-resolutions", default="8,16,24")
    parser.add_argument("--feature-replicates", type=int, default=5)
    parser.add_argument("--n-clusters", type=int, default=2)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-runs", type=int, default=0, help="Debug limit; 0 uses all valid runs.")
    parser.add_argument(
        "--split-method",
        choices=["auto", "multiscale_ward", "multiscale_spherical_kmeans", "multiscale_spectral", "sliced_wasserstein_ward", "sliced_wasserstein_spectral"],
        default="auto",
    )
    return parser.parse_args()


def fibonacci_directions(count: int) -> np.ndarray:
    """Deterministic approximately uniform directions on the unit sphere."""
    count = max(6, int(count))
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / float(count)
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden_angle * index
    directions = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])
    return directions.astype(np.float32)


def parse_resolutions(text: str) -> List[int]:
    values = sorted({int(item.strip()) for item in str(text).split(",") if item.strip()})
    if not values or any(value < 2 or value > 64 for value in values):
        raise ValueError("Voxel resolutions must be integers in [2, 64].")
    return values


def discover_runs(root: Path) -> List[int]:
    run_ids = []
    for path in root.glob("run_*/surface_coords.npy"):
        try:
            run_ids.append(int(path.parent.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    run_ids = sorted(set(run_ids))
    if not run_ids:
        raise FileNotFoundError(f"No run_*/surface_coords.npy files found under {root}")
    return run_ids


def load_manifest(root: Path) -> Dict[str, List[int]]:
    path = root / "preprocessed_manifest.json"
    if not path.is_file():
        return {"train_ids": [], "test_ids": []}
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
        "train_ids": [int(item) for item in manifest.get("train_ids", [])],
        "test_ids": [int(item) for item in manifest.get("test_ids", [])],
    }


def load_common_bounds(root: Path, run_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, str]:
    stats_path = root / "position_stats_v2_h5.npy"
    if stats_path.is_file():
        stats = np.asarray(np.load(stats_path), dtype=np.float64)
        if stats.shape == (2, 3) and np.all(np.isfinite(stats)) and np.all(stats[1] > stats[0]):
            return stats[0], stats[1], str(stats_path)

    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    for run_id in run_ids:
        coords = np.load(root / f"run_{run_id}" / "surface_coords.npy", mmap_mode="r")
        minimum = np.minimum(minimum, np.asarray(coords.min(axis=0), dtype=np.float64))
        maximum = np.maximum(maximum, np.asarray(coords.max(axis=0), dtype=np.float64))
    if not np.all(maximum > minimum):
        raise ValueError(f"Invalid common coordinate bounds: min={minimum}, max={maximum}")
    return minimum, maximum, "computed_from_all_surface_coords"


def sample_coordinates(root: Path, run_id: int, sample_points: int, seed: int) -> np.ndarray:
    coords = np.asarray(np.load(root / f"run_{run_id}" / "surface_coords.npy", mmap_mode="r"), dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"run_{run_id} surface_coords.npy has shape {coords.shape}, expected (N, 3)")
    count = min(int(sample_points), int(coords.shape[0]))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(run_id), 918273]))
    if count == coords.shape[0]:
        indices = np.arange(coords.shape[0], dtype=np.int64)
    else:
        indices = rng.choice(coords.shape[0], size=count, replace=False)
    return np.asarray(coords[indices], dtype=np.float64)


def normalize_coordinates(coords: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    scaled = (coords - minimum[None, :]) / np.maximum(maximum - minimum, 1.0e-12)[None, :]
    return np.clip(2.0 * scaled - 1.0, -1.0, 1.0)


def quantile_block(values: np.ndarray, quantile_count: int) -> np.ndarray:
    quantiles = np.linspace(0.02, 0.98, int(quantile_count), dtype=np.float64)
    return np.quantile(values, quantiles, axis=0).reshape(-1)


def occupancy_block(points: np.ndarray, resolution: int) -> np.ndarray:
    scaled = np.clip((points + 1.0) * 0.5, 0.0, np.nextafter(1.0, 0.0))
    indices = np.floor(scaled * int(resolution)).astype(np.int32)
    flat = np.ravel_multi_index(indices.T, (resolution, resolution, resolution))
    occupied = np.zeros((resolution**3,), dtype=np.float64)
    occupied[np.unique(flat)] = 1.0
    return occupied


def geometry_signature(
    points: np.ndarray,
    directions: np.ndarray,
    projection_quantiles: int,
    pair_samples: int,
    voxel_resolutions: Sequence[int],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    blocks: Dict[str, np.ndarray] = {}
    blocks["axis_quantiles"] = quantile_block(points, projection_quantiles)
    radius = np.linalg.norm(points, axis=1, keepdims=True)
    blocks["radius_quantiles"] = quantile_block(radius, projection_quantiles)
    projections = points @ directions.T
    blocks["projection_quantiles"] = quantile_block(projections, projection_quantiles)

    rng = np.random.default_rng(int(seed) + 77331)
    first = rng.integers(0, points.shape[0], size=int(pair_samples), dtype=np.int64)
    second = rng.integers(0, points.shape[0], size=int(pair_samples), dtype=np.int64)
    pair_distances = np.linalg.norm(points[first] - points[second], axis=1, keepdims=True)
    blocks["pair_distance_quantiles"] = quantile_block(pair_distances, projection_quantiles)

    centered = points - points.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / max(1, points.shape[0] - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues /= max(float(eigenvalues.sum()), 1.0e-12)
    blocks["moments"] = np.concatenate(
        [points.mean(axis=0), points.std(axis=0), np.diag(covariance), eigenvalues, [float(np.mean(radius))]],
        axis=0,
    )
    for resolution in voxel_resolutions:
        blocks[f"occupancy_{resolution}"] = occupancy_block(points, int(resolution))

    full = np.concatenate(list(blocks.values()), axis=0).astype(np.float64, copy=False)
    return full, blocks


def block_scale(features: np.ndarray, block_ranges: Mapping[str, Tuple[int, int]]) -> Tuple[np.ndarray, Dict[str, object]]:
    scaled = np.asarray(features, dtype=np.float64).copy()
    block_report = {}
    for name, (start, stop) in block_ranges.items():
        block = scaled[:, start:stop]
        mean = block.mean(axis=0)
        std = block.std(axis=0)
        std[std < 1.0e-10] = 1.0
        block[:] = np.clip((block - mean[None, :]) / std[None, :], -8.0, 8.0)
        block[:] /= math.sqrt(max(1, stop - start))
        block_report[name] = {"start": start, "stop": stop, "dimension": stop - start}
    return scaled, block_report


def make_feature_matrix(
    root: Path,
    run_ids: Sequence[int],
    minimum: np.ndarray,
    maximum: np.ndarray,
    sample_points: int,
    projection_count: int,
    projection_quantiles: int,
    pair_samples: int,
    voxel_resolutions: Sequence[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]], List[Dict[str, object]]]:
    directions = fibonacci_directions(projection_count)
    all_features = []
    all_sw_features = []
    block_ranges: Dict[str, Tuple[int, int]] = {}
    summaries = []
    first_blocks = None
    for run_id in run_ids:
        points = normalize_coordinates(sample_coordinates(root, run_id, sample_points, seed), minimum, maximum)
        full, blocks = geometry_signature(
            points,
            directions,
            projection_quantiles,
            pair_samples,
            voxel_resolutions,
            seed=int(seed) + int(run_id),
        )
        if first_blocks is None:
            cursor = 0
            for name, values in blocks.items():
                block_ranges[name] = (cursor, cursor + int(values.shape[0]))
                cursor += int(values.shape[0])
            first_blocks = list(blocks)
        all_features.append(full)
        all_sw_features.append(blocks["projection_quantiles"])
        extent = points.max(axis=0) - points.min(axis=0)
        summaries.append(
            {
                "run_id": int(run_id),
                "sampled_points": int(points.shape[0]),
                "extent_xyz": extent.tolist(),
                "centroid_xyz": points.mean(axis=0).tolist(),
                "radius_mean": float(np.linalg.norm(points, axis=1).mean()),
                "radius_std": float(np.linalg.norm(points, axis=1).std()),
            }
        )
    features = np.stack(all_features, axis=0)
    scaled, block_report = block_scale(features, block_ranges)
    sw_features = np.stack(all_sw_features, axis=0)
    sw_scaled, _ = block_scale(sw_features, {"projection_quantiles": (0, sw_features.shape[1])})
    for item in summaries:
        item["feature_dimension"] = int(scaled.shape[1])
    return scaled, sw_scaled, block_ranges, summaries


def rbf_affinity(features: np.ndarray) -> np.ndarray:
    distances = squareform(pdist(features, metric="euclidean"))
    nonzero = distances[distances > 0.0]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0
    scale = max(scale, 1.0e-12)
    affinity = np.exp(-0.5 * (distances / scale) ** 2)
    np.fill_diagonal(affinity, 1.0)
    return affinity


def cluster_labels(name: str, features: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    if name.endswith("ward"):
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(features)
    if name.endswith("spherical_kmeans"):
        normalized = normalize(features)
        return KMeans(n_clusters=n_clusters, n_init=50, random_state=int(seed), algorithm="lloyd").fit_predict(normalized)
    if name.endswith("spectral"):
        affinity = rbf_affinity(features)
        return SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="cluster_qr",
            random_state=int(seed),
        ).fit_predict(affinity)
    raise ValueError(f"Unknown clusterer {name}")


def candidate_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    counts = np.bincount(labels.astype(np.int64), minlength=int(labels.max()) + 1)
    if len(np.unique(labels)) < 2:
        return {"silhouette": -1.0, "calinski_harabasz": 0.0, "davies_bouldin": math.inf, "balance": 0.0}
    return {
        "silhouette": float(silhouette_score(features, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
        "balance": float(np.min(counts) / max(np.max(counts), 1)),
    }


def representative_ids(features: np.ndarray, labels: np.ndarray, run_ids: Sequence[int]) -> Dict[str, List[int]]:
    result = {}
    for cluster_id in sorted(np.unique(labels).tolist()):
        member_idx = np.flatnonzero(labels == cluster_id)
        centroid = features[member_idx].mean(axis=0)
        distance = np.linalg.norm(features[member_idx] - centroid[None, :], axis=1)
        order = member_idx[np.argsort(distance)]
        result[str(int(cluster_id))] = [int(run_ids[index]) for index in order[:5]]
    return result


def save_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def save_cluster_plot(path: Path, embedding: np.ndarray, labels: np.ndarray, run_ids: Sequence[int], title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    colors = ["#1f77b4", "#d62728"]
    for cluster_id in sorted(np.unique(labels).tolist()):
        mask = labels == cluster_id
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=34, alpha=0.85, color=colors[int(cluster_id) % 2], label=f"cluster {cluster_id} (n={int(mask.sum())})")
        for x, y, run_id in zip(embedding[mask, 0], embedding[mask, 1], np.asarray(run_ids)[mask]):
            ax.annotate(str(int(run_id)), (x, y), fontsize=5, alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    voxel_resolutions = parse_resolutions(args.voxel_resolutions)
    run_ids = discover_runs(root)
    if int(args.max_runs) > 0:
        run_ids = run_ids[: int(args.max_runs)]
    if len(run_ids) < 4:
        raise ValueError("At least four runs are required for a stable two-cluster diagnostic.")
    manifest = load_manifest(root)
    minimum, maximum, bounds_source = load_common_bounds(root, run_ids)
    print(f"Geometry runs: {len(run_ids)}")
    print(f"Common bounds source: {bounds_source}; min={minimum.tolist()}, max={maximum.tolist()}")

    feature_replicates = max(2, int(args.feature_replicates))
    feature_sets = []
    sw_sets = []
    block_ranges = None
    summaries = None
    for replicate in range(feature_replicates):
        print(f"Building statistical geometry signatures {replicate + 1}/{feature_replicates}")
        features, sw_features, block_ranges, summaries = make_feature_matrix(
            root,
            run_ids,
            minimum,
            maximum,
            int(args.sample_points),
            int(args.projection_count),
            int(args.projection_quantiles),
            int(args.pair_samples),
            voxel_resolutions,
            seed=int(args.seed) + replicate * 100003,
        )
        feature_sets.append(features)
        sw_sets.append(sw_features)

    candidate_inputs = {
        "multiscale": ["multiscale_ward", "multiscale_spherical_kmeans", "multiscale_spectral"],
        "sliced_wasserstein": ["sliced_wasserstein_ward", "sliced_wasserstein_spectral"],
    }
    candidate_names = candidate_inputs["multiscale"] + candidate_inputs["sliced_wasserstein"]
    candidate_results: Dict[str, Dict[str, object]] = {}
    for candidate in candidate_names:
        labels_by_replicate = []
        metrics_by_replicate = []
        for replicate in range(feature_replicates):
            source = sw_sets[replicate] if candidate.startswith("sliced_wasserstein") else feature_sets[replicate]
            labels = cluster_labels(candidate, source, int(args.n_clusters), int(args.seed) + replicate)
            labels_by_replicate.append(labels)
            metrics_by_replicate.append(candidate_metrics(source, labels))
        base_labels = labels_by_replicate[0]
        stability = [adjusted_rand_score(base_labels, labels) for labels in labels_by_replicate[1:]]
        base_metrics = metrics_by_replicate[0]
        quality_score = (
            0.55 * float(np.mean(stability))
            + 0.30 * ((float(base_metrics["silhouette"]) + 1.0) * 0.5)
            + 0.15 * float(base_metrics["balance"])
        )
        candidate_results[candidate] = {
            "base_labels": base_labels.astype(int).tolist(),
            "base_metrics": base_metrics,
            "replicate_metrics": metrics_by_replicate,
            "replicate_ari_to_base": [float(value) for value in stability],
            "mean_stability_ari": float(np.mean(stability)),
            "quality_score": float(quality_score),
            "cluster_sizes": [int(np.sum(base_labels == cluster_id)) for cluster_id in range(int(args.n_clusters))],
        }

    if args.split_method == "auto":
        selected_method = max(
            candidate_results,
            key=lambda name: (
                float(candidate_results[name]["quality_score"]),
                float(candidate_results[name]["mean_stability_ari"]),
                float(candidate_results[name]["base_metrics"]["silhouette"]),
            ),
        )
    else:
        selected_method = str(args.split_method)
    selected_labels = np.asarray(candidate_results[selected_method]["base_labels"], dtype=np.int64)
    selected_features = sw_sets[0] if selected_method.startswith("sliced_wasserstein") else feature_sets[0]

    pca_components = min(16, selected_features.shape[0] - 1, selected_features.shape[1])
    pca = PCA(n_components=max(2, pca_components), random_state=int(args.seed))
    pca_embedding = pca.fit_transform(selected_features)
    tsne_perplexity = min(float(args.perplexity), max(2.0, (len(run_ids) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        random_state=int(args.seed),
    )
    tsne_embedding = tsne.fit_transform(pca_embedding)
    save_cluster_plot(output / "geometry_clusters_pca.png", pca_embedding[:, :2], selected_labels, run_ids, f"Statistical geometry clusters: {selected_method} (PCA)")
    save_cluster_plot(output / "geometry_clusters_tsne.png", tsne_embedding, selected_labels, run_ids, f"Statistical geometry clusters: {selected_method} (t-SNE visualization only)")

    train_set = set(manifest.get("train_ids", []))
    test_set = set(manifest.get("test_ids", []))
    cluster_rows = []
    cluster_summary = {}
    for cluster_id in sorted(np.unique(selected_labels).tolist()):
        member_indices = np.flatnonzero(selected_labels == cluster_id)
        member_ids = [int(run_ids[index]) for index in member_indices]
        cluster_summary[str(int(cluster_id))] = {
            "count": len(member_ids),
            "run_ids": member_ids,
            "original_manifest_train_count": int(sum(run_id in train_set for run_id in member_ids)),
            "original_manifest_test_count": int(sum(run_id in test_set for run_id in member_ids)),
            "representative_run_ids": representative_ids(selected_features, selected_labels, run_ids)[str(int(cluster_id))],
            "mean_extent_xyz": np.mean([summaries[index]["extent_xyz"] for index in member_indices], axis=0).tolist(),
            "std_extent_xyz": np.std([summaries[index]["extent_xyz"] for index in member_indices], axis=0).tolist(),
            "mean_centroid_xyz": np.mean([summaries[index]["centroid_xyz"] for index in member_indices], axis=0).tolist(),
            "mean_radius": float(np.mean([summaries[index]["radius_mean"] for index in member_indices])),
        }
        for index in member_indices:
            cluster_rows.append(
                {
                    "run_id": int(run_ids[index]),
                    "cluster_id": int(cluster_id),
                    "original_manifest_split": "train" if run_ids[index] in train_set else "test" if run_ids[index] in test_set else "unknown",
                    "pca_x": float(pca_embedding[index, 0]),
                    "pca_y": float(pca_embedding[index, 1]),
                    "tsne_x": float(tsne_embedding[index, 0]),
                    "tsne_y": float(tsne_embedding[index, 1]),
                }
            )
    cluster_rows.sort(key=lambda row: row["run_id"])
    save_csv(output / "geometry_cluster_assignments.csv", cluster_rows, list(cluster_rows[0]))

    manifest_labels = []
    manifest_label_mask = []
    for run_id in run_ids:
        if run_id in train_set:
            manifest_labels.append(0)
            manifest_label_mask.append(True)
        elif run_id in test_set:
            manifest_labels.append(1)
            manifest_label_mask.append(True)
        else:
            manifest_labels.append(-1)
            manifest_label_mask.append(False)
    manifest_label_mask = np.asarray(manifest_label_mask, dtype=bool)
    if manifest_label_mask.any() and len(np.unique(np.asarray(manifest_labels)[manifest_label_mask])) == 2:
        manifest_overlap = {
            "runs_with_manifest_assignment": int(manifest_label_mask.sum()),
            "train_runs_in_geometry_split": int(sum(run_id in train_set for run_id in run_ids)),
            "test_runs_in_geometry_split": int(sum(run_id in test_set for run_id in run_ids)),
            "adjusted_rand_vs_original_manifest": float(
                adjusted_rand_score(
                    np.asarray(manifest_labels, dtype=np.int64)[manifest_label_mask],
                    selected_labels[manifest_label_mask],
                )
            ),
        }
    else:
        manifest_overlap = {"runs_with_manifest_assignment": int(manifest_label_mask.sum())}

    split_json = {
        "description": "Geometry-only statistical domain split for cross-domain SATLOSS7 generalization.",
        "data_root": str(root),
        "run_count": len(run_ids),
        "seed": int(args.seed),
        "bounds_source": bounds_source,
        "common_min_xyz": minimum.tolist(),
        "common_max_xyz": maximum.tolist(),
        "statistical_representation": {
            "sample_points_per_run": int(args.sample_points),
            "feature_replicates": feature_replicates,
            "projection_count": int(args.projection_count),
            "projection_quantiles": int(args.projection_quantiles),
            "pair_samples": int(args.pair_samples),
            "voxel_resolutions": voxel_resolutions,
            "no_neural_encoder": True,
            "no_cfd_fields_or_targets": True,
        },
        "selection_rule": "0.55 mean replicate ARI stability + 0.30 normalized silhouette + 0.15 cluster balance",
        "selected_method": selected_method,
        "candidate_results": candidate_results,
        "clusters": cluster_summary,
        "train_cluster_0_test_cluster_1": {"train_ids": cluster_summary["0"]["run_ids"], "test_ids": cluster_summary["1"]["run_ids"]},
        "train_cluster_1_test_cluster_0": {"train_ids": cluster_summary["1"]["run_ids"], "test_ids": cluster_summary["0"]["run_ids"]},
        "original_manifest_overlap_is_diagnostic_only": True,
        "original_manifest_overlap": manifest_overlap,
        "run_summaries": summaries,
    }
    with (output / "geometry_domain_split.json").open("w", encoding="utf-8") as handle:
        json.dump(split_json, handle, indent=2)

    with (output / "method_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump({"selected_method": selected_method, "candidate_results": candidate_results}, handle, indent=2)
    print(f"Selected statistical split: {selected_method}")
    print(f"Cluster sizes: {[cluster_summary[str(i)]['count'] for i in range(int(args.n_clusters))]}")
    print(f"Wrote split: {output / 'geometry_domain_split.json'}")
    print(f"Wrote assignments: {output / 'geometry_cluster_assignments.csv'}")


if __name__ == "__main__":
    main()
