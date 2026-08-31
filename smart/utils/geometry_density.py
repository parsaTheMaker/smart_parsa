from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys


def _load_conda_libstdcpp() -> None:
    """Load the active Conda C++ runtime before optional kNN extensions.

    Some cluster shells resolve an older system ``libstdc++`` first. The
    optional torch-cluster and scikit-learn backends then cannot load even
    though the active Conda environment contains a compatible runtime.
    """
    lib_dir = Path(sys.prefix) / "lib"
    for candidate in (lib_dir / "libstdc++.so.6.0.34", lib_dir / "libstdc++.so.6"):
        if not candidate.is_file():
            continue
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
        break


_load_conda_libstdcpp()

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


def _sklearn_knn_jobs() -> int:
    """Honor an explicit per-process kNN worker limit when one is configured.

    Dataset preprocessing often evaluates many independent clouds in separate
    processes.  ``n_jobs=-1`` in every process then oversubscribes the host.
    Keeping the default preserves existing behavior, while preprocessing
    launchers can set ``SMART_KNN_N_JOBS=1`` and parallelize across cases.
    """
    try:
        return int(os.environ.get("SMART_KNN_N_JOBS", "-1"))
    except ValueError:
        return -1


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
            nbrs = NearestNeighbors(
                n_neighbors=k_cur + 1,
                algorithm="kd_tree",
                n_jobs=_sklearn_knn_jobs(),
            )
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
