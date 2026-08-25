#!/usr/bin/env python3
"""Create fast, geometry-only decimated VTPs for DrivAerML surfaces.

The default method is VTK's topology-preserving DecimatePro. The decimations
are chained (5x, then 10x, 20x, and 40x), which makes the expensive first pass
run only on the full mesh and keeps later passes fast. A faster quadric
clustering method is available explicitly. Isotropic remeshing uses PyACVD's
compiled ACVD surface clustering to produce a uniform triangle mesh. The
requested target edge length is converted to an area-equivalent target vertex
count, avoiding the unbounded edge-operation cost of explicit remeshing.

Outputs are written as readable, uncompressed inline-binary VTP files:

    <output-dir>/run_<id>/drivaer_<id>_faces_div5.vtp

The requested factors are approximate because voxel occupancy depends on the
surface geometry. A JSON profile records the actual point/triangle counts,
achieved factor, validation results, and timings for every output.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import queue
import re
import signal
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


RUN_FILE_RE = re.compile(r"^drivaer_(\d+)\.vtp$")

# These divisions were calibrated on the 17.7M-triangle DrivAerML run-1 mesh.
# They produce approximately 5x, 10x, 20x, and 40x fewer triangles while
# keeping the domain aspect ratio. Other factors are scaled from the closest
# calibrated entry.
CALIBRATED_DIVISIONS = {
    5: (1024, 512, 384),
    10: (724, 362, 272),
    20: (450, 225, 169),
    40: (306, 153, 115),
}


def parse_factors(text: str) -> list[int]:
    factors = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        factor = int(item)
        if factor <= 1:
            raise ValueError(f"Face-division factors must be > 1, got {factor}.")
        factors.append(factor)
    factors = sorted(set(factors))
    if not factors:
        raise ValueError("--factors must contain at least one integer.")
    return factors


def parse_divisions(text: str) -> tuple[int, int, int]:
    values = tuple(int(item.strip()) for item in str(text).split(","))
    if len(values) != 3 or any(value < 8 for value in values):
        raise ValueError("Divisions must be three integers >= 8, e.g. 450,225,169.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/mnt/ssdraid/parsa/drivaerml_surface_vtp"),
        help="Root containing run_<id>/drivaer_<id>.vtp files.",
    )
    parser.add_argument(
        "--input-glob",
        default="run_*/drivaer_*.vtp",
        help=(
            "Pattern below --input-dir used to discover input VTPs. The default preserves "
            "the DrivAerML layout; for toy surfaces use 'case_*/toy_case_*_surface.vtp'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/ssdraid/parsa/drivaerml_surface_vtp_decimated"),
        help="Root for decimated VTPs and decimation_summary.json.",
    )
    parser.add_argument("--factors", default="5,10,20,40", help="Approximate face-division factors.")
    parser.add_argument(
        "--method",
        choices=(
            "decimate_pro",
            "quadric_clustering",
            "voxel_quadric_clustering",
            "quadric_decimation",
            "isotropic_remeshing",
            "isotropic_gpu",
        ),
        default="decimate_pro",
        help=(
            "Topology-preserving DecimatePro by default; "
            "voxel_quadric_clustering uses a uniform VTK grid; "
            "isotropic_remeshing uses compiled PyACVD/ACVD remeshing."
        ),
    )
    parser.add_argument(
        "--feature-angle",
        type=float,
        default=30.0,
        help="Feature angle used by the protected decimators (default: 30 degrees).",
    )
    parser.add_argument(
        "--divisions-5",
        default="1024,512,384",
        help="Voxel divisions for factor 5, calibrated for DrivAerML.",
    )
    parser.add_argument(
        "--divisions-10",
        default="724,362,272",
        help="Voxel divisions for factor 10, calibrated for DrivAerML.",
    )
    parser.add_argument(
        "--divisions-20",
        default="450,225,169",
        help="Voxel divisions for factor 20, calibrated for DrivAerML.",
    )
    parser.add_argument(
        "--divisions-40",
        default="306,153,115",
        help="Voxel divisions for factor 40, calibrated for DrivAerML.",
    )
    parser.add_argument(
        "--isotropic-iterations",
        type=int,
        default=5,
        help="ACVD cluster optimization iterations per output (default: 5).",
    )
    parser.add_argument(
        "--isotropic-iso-tries",
        type=int,
        default=3,
        help="ACVD isolated-cluster repair attempts (default: 3).",
    )
    parser.add_argument(
        "--isotropic-seed-triangle-multiplier",
        type=float,
        default=1.25,
        help=(
            "Pre-reduce with VTK to this multiple of the target triangle count "
            "before ACVD (default: 1.25)."
        ),
    )
    parser.add_argument(
        "--gpu-device",
        default="cuda:0",
        help="Single CUDA device for --method isotropic_gpu (default: cuda:0).",
    )
    parser.add_argument(
        "--gpu-devices",
        default=None,
        help=(
            "Comma-separated CUDA devices for --method isotropic_gpu, one fixed worker per device, "
            "for example cuda:0,cuda:1. Overrides --gpu-device."
        ),
    )
    parser.add_argument(
        "--gpu-adjustments",
        type=int,
        default=3,
        help="CUDA voxel-size feedback steps for --method isotropic_gpu (default: 3).",
    )
    parser.add_argument(
        "--gpu-face-chunk-size",
        type=int,
        default=1_048_576,
        help="Faces processed per CUDA remapping chunk (default: 1048576).",
    )
    parser.add_argument(
        "--runs",
        default=None,
        help="Optional comma-separated run ids, e.g. 1,10,34. Defaults to every input VTP.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many input VTPs.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent geometry workers. Each worker uses about 2.5 GB peak RAM with DecimatePro; default: 1.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing decimated outputs.")
    parser.add_argument(
        "--validate-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Check finite points, triangle-only cells, and positive triangle areas (default: disabled).",
    )
    parser.add_argument(
        "--validate-topology",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also count boundary/non-manifold edges (default: disabled).",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed input.")
    return parser.parse_args()


def require_vtk():
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("VTK Python bindings are required in the active environment.") from exc
    return vtk, vtk_to_numpy


def discover_inputs(input_dir: Path, runs: str | None, limit: int | None, input_glob: str) -> list[Path]:
    if runs:
        run_ids = []
        for item in runs.split(","):
            item = item.strip()
            if item:
                run_ids.append(int(item))
        paths = [input_dir / f"run_{run_id}" / f"drivaer_{run_id}.vtp" for run_id in sorted(set(run_ids))]
    else:
        paths = []
        for path in sorted(input_dir.glob(str(input_glob))):
            if str(input_glob) != "run_*/drivaer_*.vtp" or RUN_FILE_RE.fullmatch(path.name):
                paths.append(path)
    paths = [path for path in paths if path.is_file()]
    if limit is not None:
        paths = paths[: max(0, int(limit))]
    if not paths:
        raise FileNotFoundError(f"No VTP inputs matching {input_glob!r} found under {input_dir}.")
    return paths


def read_polydata(vtk, path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetPoints() is None or polydata.GetNumberOfPolys() == 0:
        raise RuntimeError(f"Input VTP is empty or has no polygons: {path}")
    return polydata


def is_triangle_only(vtk, polydata) -> bool:
    polys = polydata.GetPolys()
    if polys is None or polys.GetNumberOfCells() != polydata.GetNumberOfPolys():
        return False
    if hasattr(polys, "IsHomogeneous"):
        return bool(polys.IsHomogeneous() == 3)
    if polys.GetNumberOfCells() == 0:
        return False
    return all(polydata.GetCellType(index) == vtk.VTK_TRIANGLE for index in (0, polys.GetNumberOfCells() - 1))


def triangulate_if_needed(vtk, polydata):
    if is_triangle_only(vtk, polydata):
        return polydata
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polydata)
    if hasattr(triangle_filter, "PreservePolysOff"):
        triangle_filter.PreservePolysOff()
    triangle_filter.Update()
    output = triangle_filter.GetOutput()
    if output is None or output.GetNumberOfPolys() == 0:
        raise RuntimeError("TriangleFilter produced no polygons.")
    return output


def geometry_only(vtk, polydata):
    output = vtk.vtkPolyData()
    output.SetPoints(polydata.GetPoints())
    output.SetPolys(polydata.GetPolys())
    output.BuildCells()
    return output


def remove_invalid_triangles(vtk, vtk_to_numpy, polydata):
    """Drop zero-area cells and unused invalid vertices from a triangle mesh.

    Some ACVD releases occasionally leave coincident-vertex triangles after
    clustering highly graded meshes.  These cells have zero measure and carry
    no surface geometry.  Removing only those cells is safer than accepting an
    invalid VTP or applying a broad smoothing/cleaning pass.
    """
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    polydata = triangulate_if_needed(vtk, polydata)
    points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=np.float64)
    raw_cells = np.asarray(vtk_to_numpy(polydata.GetPolys().GetData()), dtype=np.int64)
    if raw_cells.size == 0 or raw_cells.size % 4 != 0 or not np.all(raw_cells[::4] == 3):
        raise RuntimeError("Cannot repair a non-triangle mesh.")
    faces = raw_cells.reshape(-1, 4)[:, 1:]
    valid_indices = np.logical_and(faces >= 0, faces < points.shape[0]).all(axis=1)
    finite_faces = np.zeros(faces.shape[0], dtype=bool)
    finite_faces[valid_indices] = np.isfinite(points[faces[valid_indices]]).all(axis=(1, 2))
    triangles = np.zeros((faces.shape[0], 3, 3), dtype=np.float64)
    triangles[finite_faces] = points[faces[finite_faces]]
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    finite_points = points[np.isfinite(points).all(axis=1)]
    span = float(np.linalg.norm(finite_points.max(axis=0) - finite_points.min(axis=0))) if finite_points.size else 1.0
    area_tolerance = max(1.0e-16, (max(span, 1.0e-8) ** 2) * 1.0e-12)
    keep = finite_faces & (double_area > 2.0 * area_tolerance)
    if not np.any(keep):
        raise RuntimeError("Degenerate-triangle repair would remove the complete surface.")
    kept_faces = faces[keep]
    used = np.unique(kept_faces.reshape(-1))
    remap = np.full(points.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    compact_faces = remap[kept_faces]
    compact_points = points[used].astype(np.float32, copy=False)
    output = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(compact_points), deep=True))
    output.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    offsets = np.arange(0, 3 * compact_faces.shape[0] + 1, 3, dtype=np.int64)
    cells.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(compact_faces.reshape(-1)), deep=True),
    )
    output.SetPolys(cells)
    output.BuildCells()
    return output, int((~keep).sum())


def surface_area(vtk, polydata) -> float:
    """Return total area without allocating one scalar per source triangle."""
    mass_properties = vtk.vtkMassProperties()
    mass_properties.SetInputData(polydata)
    mass_properties.Update()
    area = float(mass_properties.GetSurfaceArea())
    if not np.isfinite(area) or area <= 0.0:
        raise RuntimeError(f"Invalid source surface area: {area!r}")
    return area


def divisions_for_factor(factor: int, calibrated: dict[int, tuple[int, int, int]]) -> tuple[int, int, int]:
    if factor in calibrated:
        return calibrated[factor]
    nearest_factor = min(calibrated, key=lambda value: abs(math.log(float(value)) - math.log(float(factor))))
    scale = math.sqrt(float(nearest_factor) / float(factor))
    return tuple(max(8, int(round(value * scale))) for value in calibrated[nearest_factor])


def run_quadric_clustering(vtk, polydata, divisions: tuple[int, int, int], feature_angle: float):
    cluster = vtk.vtkQuadricClustering()
    cluster.SetInputData(polydata)
    cluster.SetNumberOfDivisions(*divisions)
    # Automatic adjustment would silently change the requested divisions.
    cluster.AutoAdjustNumberOfDivisionsOff()
    cluster.UseFeatureEdgesOn()
    cluster.UseFeaturePointsOn()
    cluster.SetFeaturePointsAngle(float(feature_angle))
    cluster.UseInputPointsOn()
    cluster.Update()
    output = cluster.GetOutput()
    if output is None or output.GetNumberOfPolys() == 0:
        raise RuntimeError(f"Quadric clustering produced no polygons for divisions={divisions}.")
    return geometry_only(vtk, output)


def run_isotropic_remeshing(
    vtk,
    polydata,
    source_triangles: int,
    target_edge_length: float,
    target_triangles: float,
    args,
):
    """Run fast ACVD uniform surface remeshing.

    ACVD optimizes a uniform Voronoi clustering on the input surface and
    reconstructs a triangle mesh from the clusters. For a near-regular closed
    triangle mesh, F ~= 2V, so the area-derived target edge length is mapped
    to approximately half as many target vertices. This keeps the requested
    geometric target while avoiding an expensive full-edge sweep.
    """
    try:
        import pyacvd
        import pyvista as pv
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "PyACVD and PyVista are required for --method isotropic_remeshing; "
            "install pyacvd in the active environment."
        ) from exc

    if target_edge_length <= 0.0 or target_triangles <= 0.0:
        raise ValueError("Isotropic target edge length and triangle count must be positive.")
    target_vertices = max(4, int(round(float(target_triangles) / 2.0)))
    seed_triangles = min(
        int(source_triangles),
        max(target_vertices * 2, int(round(float(target_triangles) * args.isotropic_seed_triangle_multiplier))),
    )
    if seed_triangles < int(source_triangles):
        seed = run_quadric_decimation_to_triangles(vtk, polydata, seed_triangles)
    else:
        seed = polydata
    surface = pv.wrap(seed)
    clusterer = pyacvd.Clustering(surface)
    clusterer.cluster(
        target_vertices,
        maxiter=int(args.isotropic_iterations),
        iso_try=int(args.isotropic_iso_tries),
    )
    remeshed = clusterer.create_mesh(moveclus=True, flipnorm=True, clean=True)
    if remeshed is None or remeshed.n_cells == 0:
        raise RuntimeError("PyACVD returned no triangles.")
    output = geometry_only(vtk, remeshed)
    del remeshed, clusterer, surface, seed
    gc.collect()
    return output


def _gpu_voxel_remesh(torch, points, faces, voxel_size, face_chunk_size):
    """CUDA voxel-centroid remeshing for one voxel size.

    All expensive point assignment, centroid accumulation, face remapping, and
    duplicate-face removal remain on CUDA. The returned tensors are still on
    CUDA so the caller can adjust the voxel size without round-tripping data.
    """
    bounds_min = points.amin(dim=0)
    cell_index = torch.floor((points - bounds_min) / float(voxel_size)).to(torch.int64)
    _unique_cells, point_cluster = torch.unique(cell_index, dim=0, return_inverse=True)
    cluster_count = int(_unique_cells.shape[0])
    if cluster_count < 4:
        raise RuntimeError("CUDA voxel remeshing collapsed the surface to fewer than four cells.")
    del cell_index, _unique_cells

    centroids = torch.zeros((cluster_count, 3), device=points.device, dtype=torch.float32)
    centroids.index_add_(0, point_cluster, points)
    counts = torch.bincount(point_cluster, minlength=cluster_count).to(dtype=torch.float32)
    centroids = centroids / counts.clamp_min(1.0).unsqueeze(1)

    # A former implementation encoded each sorted triangle as a base-N int64
    # key.  DrivAerML can retain more than 2.1M clusters, for which N**3
    # overflows int64.  Keep the three indices instead: torch.unique(dim=0)
    # is exact and remains entirely on CUDA.
    face_chunks = []
    num_faces = int(faces.shape[0])
    for start in range(0, num_faces, int(face_chunk_size)):
        mapped = point_cluster[faces[start:start + int(face_chunk_size)]]
        mapped = torch.sort(mapped, dim=1).values
        valid = (mapped[:, 0] < mapped[:, 1]) & (mapped[:, 1] < mapped[:, 2])
        mapped = mapped[valid]
        if mapped.numel() > 0:
            face_chunks.append(torch.unique(mapped, dim=0))
        del mapped, valid
    if not face_chunks:
        raise RuntimeError("CUDA voxel remeshing produced no non-degenerate faces.")
    remapped_faces = torch.unique(torch.cat(face_chunks, dim=0), dim=0)
    del face_chunks, point_cluster, counts
    # Voxel vertex merging can make more than two source triangles share an
    # edge. Remove only the excess incident triangles on CUDA so the output
    # does not contain non-manifold edges before it is handed back to VTK.
    num_output_vertices = cluster_count
    edge_pairs = torch.stack(
        (
            remapped_faces[:, [0, 1]],
            remapped_faces[:, [1, 2]],
            remapped_faces[:, [0, 2]],
        ),
        dim=1,
    ).reshape(-1, 2)
    edge_pairs = torch.sort(edge_pairs, dim=1).values
    edge_keys = edge_pairs[:, 0] * num_output_vertices + edge_pairs[:, 1]
    order = torch.argsort(edge_keys)
    sorted_keys = edge_keys[order]
    unique_edge_keys, edge_counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    edge_starts = torch.cumsum(edge_counts, dim=0) - edge_counts
    edge_rank = torch.arange(sorted_keys.numel(), device=faces.device) - torch.repeat_interleave(
        edge_starts, edge_counts
    )
    allowed_occurrences = edge_rank < 2
    face_ids = torch.arange(remapped_faces.shape[0], device=faces.device).repeat_interleave(3)[order]
    keep_faces = torch.ones(remapped_faces.shape[0], device=faces.device, dtype=torch.int8)
    keep_faces.scatter_reduce_(
        0,
        face_ids,
        allowed_occurrences.to(torch.int8),
        reduce="amin",
        include_self=True,
    )
    keep_faces = keep_faces.bool()
    remapped_faces = remapped_faces[keep_faces]
    del (
        edge_pairs,
        edge_keys,
        order,
        sorted_keys,
        unique_edge_keys,
        edge_counts,
        edge_starts,
        edge_rank,
        allowed_occurrences,
        face_ids,
        keep_faces,
    )
    if remapped_faces.shape[0] == 0:
        raise RuntimeError("CUDA manifold cleanup removed every triangle.")
    used_clusters, compact_faces = torch.unique(
        remapped_faces.reshape(-1), sorted=True, return_inverse=True
    )
    output_points = centroids[used_clusters]
    output_faces = compact_faces.reshape(-1, 3)
    del remapped_faces, used_clusters, compact_faces, centroids
    return output_points, output_faces


def run_isotropic_gpu(vtk, vtk_to_numpy, polydata, target_edge_length: float, target_triangles: float, args):
    """GPU-native uniform surface remeshing using CUDA voxel centroids."""
    try:
        import torch
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PyTorch and VTK NumPy bindings are required for isotropic_gpu.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("--method isotropic_gpu requires a CUDA-capable PyTorch runtime.")
    if args.gpu_adjustments <= 0 or args.gpu_face_chunk_size <= 0:
        raise ValueError("GPU adjustment and face chunk sizes must be positive.")

    device = torch.device(args.gpu_device)
    if device.type != "cuda":
        raise ValueError(f"--gpu-device must be CUDA, got {args.gpu_device!r}.")
    try:
        torch.cuda.get_device_properties(device)
    except Exception as exc:
        raise RuntimeError(f"Cannot access CUDA device {args.gpu_device!r}.") from exc

    points_np = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=np.float32)
    raw_cells = np.asarray(vtk_to_numpy(polydata.GetPolys().GetData()))
    if raw_cells.size == 0 or raw_cells.size % 4 != 0 or not np.all(raw_cells[::4] == 3):
        raise RuntimeError("isotropic_gpu requires triangle-only input.")
    faces_np = np.ascontiguousarray(raw_cells.reshape(-1, 4)[:, 1:], dtype=np.int64)
    del raw_cells
    points = torch.as_tensor(points_np, device=device, dtype=torch.float32)
    faces = torch.as_tensor(faces_np, device=device, dtype=torch.int64)
    del points_np, faces_np

    target_triangles = float(target_triangles)
    voxel_size = float(target_edge_length)
    output_points = output_faces = None
    for _attempt in range(int(args.gpu_adjustments)):
        if output_points is not None:
            del output_points, output_faces
        output_points, output_faces = _gpu_voxel_remesh(
            torch,
            points,
            faces,
            voxel_size,
            args.gpu_face_chunk_size,
        )
        actual_triangles = max(int(output_faces.shape[0]), 1)
        if _attempt + 1 < int(args.gpu_adjustments):
            correction = float(np.clip(math.sqrt(actual_triangles / target_triangles), 0.65, 1.55))
            if abs(actual_triangles / target_triangles - 1.0) < 0.04:
                break
            voxel_size *= correction

    output_points_cpu = output_points.detach().cpu().numpy().astype(np.float32, copy=False)
    output_faces_cpu = output_faces.detach().cpu().numpy().astype(np.int64, copy=False)
    del output_points, output_faces, points, faces
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    result = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(output_points_cpu, deep=True))
    result.SetPoints(vtk_points)
    id_dtype = np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32
    connectivity = np.empty((output_faces_cpu.shape[0], 4), dtype=id_dtype)
    connectivity[:, 0] = 3
    connectivity[:, 1:] = output_faces_cpu
    cells = vtk.vtkCellArray()
    cells.SetCells(
        int(output_faces_cpu.shape[0]),
        numpy_to_vtkIdTypeArray(connectivity.reshape(-1), deep=True),
    )
    result.SetPolys(cells)
    result.BuildCells()
    del output_points_cpu, output_faces_cpu, connectivity
    return geometry_only(vtk, result)


def run_quadric_decimation_to_triangles(vtk, polydata, target_triangles: int):
    """Fast compiled VTK seed reduction used before ACVD clustering."""
    source_triangles = int(polydata.GetNumberOfPolys())
    target_triangles = max(4, min(int(target_triangles), source_triangles))
    if target_triangles >= source_triangles:
        return geometry_only(vtk, polydata)
    decimator = vtk.vtkQuadricDecimation()
    decimator.SetInputData(polydata)
    decimator.SetTargetReduction(1.0 - float(target_triangles) / float(source_triangles))
    decimator.VolumePreservationOn()
    if hasattr(decimator, "WeighBoundaryConstraintsByLengthOn"):
        decimator.WeighBoundaryConstraintsByLengthOn()
    decimator.Update()
    output = decimator.GetOutput()
    if output is None or output.GetNumberOfPolys() == 0:
        raise RuntimeError(f"Quadric seed reduction produced no polygons for target={target_triangles}.")
    return geometry_only(vtk, output)


def run_quadric_decimation(vtk, polydata, factor: int):
    target_triangles = max(4, int(round(polydata.GetNumberOfPolys() / float(factor))))
    return run_quadric_decimation_to_triangles(vtk, polydata, target_triangles)


def run_decimate_pro(vtk, polydata, target_reduction: float, feature_angle: float):
    """Topology-preserving edge-collapse pass.

    Chaining passes is important here: reducing the full 17M-triangle mesh
    directly to every requested target would repeat the expensive full-mesh
    operation four times.
    """
    decimator = vtk.vtkDecimatePro()
    decimator.SetInputData(polydata)
    decimator.SetTargetReduction(float(target_reduction))
    decimator.PreserveTopologyOn()
    decimator.BoundaryVertexDeletionOff()
    decimator.SetFeatureAngle(float(feature_angle))
    decimator.Update()
    output = decimator.GetOutput()
    if output is None or output.GetNumberOfPolys() == 0:
        raise RuntimeError(f"DecimatePro produced no polygons for reduction={target_reduction}.")
    return geometry_only(vtk, output)


def validate_mesh(vtk, vtk_to_numpy, polydata, check_topology: bool) -> dict:
    points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()))
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RuntimeError("Output contains non-finite or malformed point coordinates.")

    raw_cells = np.asarray(vtk_to_numpy(polydata.GetPolys().GetData()))
    if raw_cells.size == 0 or raw_cells.size % 4 != 0 or not np.all(raw_cells[::4] == 3):
        raise RuntimeError("Output contains non-triangle polygon cells.")

    cell_sizes = vtk.vtkCellSizeFilter()
    cell_sizes.SetInputData(polydata)
    cell_sizes.ComputeAreaOn()
    cell_sizes.ComputeLengthOff()
    cell_sizes.ComputeVolumeOff()
    cell_sizes.ComputeVertexCountOff()
    cell_sizes.Update()
    area_array = cell_sizes.GetOutput().GetCellData().GetArray(cell_sizes.GetAreaArrayName())
    areas = np.asarray(vtk_to_numpy(area_array), dtype=np.float64)
    if areas.size == 0 or not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise RuntimeError("Output contains non-finite or zero-area triangles.")

    result = {
        "points_finite": True,
        "triangle_only": True,
        "positive_area": True,
        "min_triangle_area": float(areas.min()),
        "max_triangle_area": float(areas.max()),
        "boundary_edges": None,
        "nonmanifold_edges": None,
    }
    if check_topology:
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(polydata)
        edges.BoundaryEdgesOn()
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.NonManifoldEdgesOff()
        edges.Update()
        result["boundary_edges"] = int(edges.GetOutput().GetNumberOfLines())

        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(polydata)
        edges.BoundaryEdgesOff()
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.NonManifoldEdgesOn()
        edges.Update()
        result["nonmanifold_edges"] = int(edges.GetOutput().GetNumberOfLines())
    return result


def write_vtp(vtk, polydata, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if partial_path.exists():
        partial_path.unlink()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(partial_path))
    writer.SetInputData(polydata)
    # Inline base64 binary is readable by older ParaView versions and avoids
    # the appended-data XML parser path that caused earlier VTP failures.
    writer.SetDataModeToBinary()
    if hasattr(writer, "SetHeaderTypeToUInt32"):
        writer.SetHeaderTypeToUInt32()
    writer.SetCompressor(None)
    if writer.Write() != 1:
        raise RuntimeError(f"VTK failed to write {output_path}.")
    partial_path.replace(output_path)


def process_input(vtk, vtk_to_numpy, input_path: Path, args, factors, calibrated) -> list[dict]:
    read_t0 = time.perf_counter()
    original = triangulate_if_needed(vtk, read_polydata(vtk, input_path))
    source_points = int(original.GetNumberOfPoints())
    source_triangles = int(original.GetNumberOfPolys())
    read_seconds = time.perf_counter() - read_t0
    records = []
    current = original
    current_factor = 1
    source_surface_area = None
    if args.method in {"isotropic_remeshing", "isotropic_gpu"}:
        source_surface_area = surface_area(vtk, original)

    # ACVD is fastest at the coarse targets. Process those first so a long
    # full-mesh input reports useful progress before the finest target runs.
    factor_sequence = (
        sorted(factors, reverse=True)
        if args.method in {"isotropic_remeshing", "isotropic_gpu"}
        else factors
    )
    for factor in factor_sequence:
        output_path = args.output_dir / input_path.parent.name / f"{input_path.stem}_faces_div{factor}.vtp"
        if output_path.is_file() and not args.overwrite:
            if args.method == "decimate_pro":
                # Reuse an existing intermediate when continuing a chained run.
                current = read_polydata(vtk, output_path)
                current_factor = factor
            records.append(
                {
                    "input": str(input_path),
                    "output": str(output_path),
                    "factor_requested": factor,
                    "status": "skipped_existing",
                }
            )
            continue

        t0 = time.perf_counter()
        divisions = None
        target_edge_length = None
        target_vertex_count = None
        isotropic_seed_triangles = None
        if args.method in {"quadric_clustering", "voxel_quadric_clustering"}:
            divisions = divisions_for_factor(factor, calibrated)
            decimated = run_quadric_clustering(vtk, original, divisions, args.feature_angle)
        elif args.method == "decimate_pro":
            target_reduction = 1.0 - float(current_factor) / float(factor)
            decimated = run_decimate_pro(vtk, current, target_reduction, args.feature_angle)
        elif args.method == "isotropic_remeshing":
            target_triangles = max(4.0, float(source_triangles) / float(factor))
            # For an equilateral triangle mesh, A = sqrt(3) * e^2 / 4.
            # This gives the explicit remesher a geometry-aware edge target;
            # the achieved count is recorded because boundaries/features can
            # move the final ratio away from the requested approximation.
            target_edge_length = math.sqrt(
                4.0 * float(source_surface_area) / (math.sqrt(3.0) * target_triangles)
            )
            target_vertex_count = max(4, int(round(target_triangles / 2.0)))
            isotropic_seed_triangles = min(
                source_triangles,
                max(
                    target_vertex_count * 2,
                    int(round(target_triangles * args.isotropic_seed_triangle_multiplier)),
                ),
            )
            print(
                f"[run {input_path.parent.name}] factor={factor} "
                f"backend=pyacvd seed_triangles={isotropic_seed_triangles:,} "
                f"target_vertices={target_vertex_count:,} starting",
                flush=True,
            )
            decimated = run_isotropic_remeshing(
                vtk,
                original,
                source_triangles,
                target_edge_length,
                target_triangles,
                args,
            )
        elif args.method == "isotropic_gpu":
            target_triangles = max(4.0, float(source_triangles) / float(factor))
            target_edge_length = math.sqrt(
                4.0 * float(source_surface_area) / (math.sqrt(3.0) * target_triangles)
            )
            print(
                f"[run {input_path.parent.name}] factor={factor} "
                f"backend=cuda target_edge={target_edge_length:.6g} starting",
                flush=True,
            )
            decimated = run_isotropic_gpu(
                vtk,
                vtk_to_numpy,
                original,
                target_edge_length,
                target_triangles,
                args,
            )
        else:
            decimated = run_quadric_decimation(vtk, original, factor)
        decimate_seconds = time.perf_counter() - t0

        repaired_triangles = 0
        if getattr(args, "repair_degenerate_triangles", False):
            decimated, repaired_triangles = remove_invalid_triangles(vtk, vtk_to_numpy, decimated)
        validation = {}
        if args.validate_output:
            validation = validate_mesh(vtk, vtk_to_numpy, decimated, args.validate_topology)
            nonmanifold_edges = validation.get("nonmanifold_edges")
            if nonmanifold_edges:
                print(
                    f"[warning] {output_path.name}: VTK reports "
                    f"{nonmanifold_edges} non-manifold edges after decimation.",
                    flush=True,
                )
        write_t0 = time.perf_counter()
        write_vtp(vtk, decimated, output_path)
        write_seconds = time.perf_counter() - write_t0
        output_points = int(decimated.GetNumberOfPoints())
        output_triangles = int(decimated.GetNumberOfPolys())
        record = {
            "input": str(input_path),
            "output": str(output_path),
            "method": args.method,
            "factor_requested": factor,
            "divisions": list(divisions) if divisions is not None else None,
            "source_surface_area": source_surface_area,
            "target_edge_length": target_edge_length,
            "target_vertex_count": target_vertex_count,
            "isotropic_seed_triangles": isotropic_seed_triangles,
            "chained_from_factor": current_factor if args.method == "decimate_pro" else None,
            "source_points": source_points,
            "source_triangles": source_triangles,
            "output_points": output_points,
            "output_triangles": output_triangles,
            "achieved_triangle_division": source_triangles / max(output_triangles, 1),
            "read_seconds": read_seconds,
            "decimate_seconds": decimate_seconds,
            "write_seconds": write_seconds,
            "status": "ok",
            "repaired_degenerate_triangles": repaired_triangles,
            "validation": validation,
        }
        records.append(record)
        print(
            f"[run {input_path.parent.name}] factor={factor} "
            f"triangles={source_triangles:,}->{output_triangles:,} "
            f"actual={record['achieved_triangle_division']:.2f}x "
            f"decimate={decimate_seconds:.1f}s write={write_seconds:.1f}s"
            + (
                f" target_edge={target_edge_length:.6g}"
                if target_edge_length is not None
                else ""
            ),
            flush=True,
        )
        if args.method == "decimate_pro":
            current = decimated
            current_factor = factor
        del decimated
        gc.collect()
    return records


def process_worker(payload):
    """Process one geometry in a fresh process so VTK state is not shared."""
    input_path, args, factors, calibrated = payload
    try:
        vtk, vtk_to_numpy = require_vtk()
        return process_input(vtk, vtk_to_numpy, input_path, args, factors, calibrated)
    except Exception as exc:
        return [{"input": str(input_path), "status": "failed", "error": repr(exc)}]


def process_gpu_worker(gpu_device, input_paths, args, factors, calibrated, result_queue):
    """Run one fixed worker per CUDA device without sharing a GPU between workers."""
    args.gpu_device = gpu_device
    try:
        import torch

        torch.cuda.set_device(torch.device(gpu_device))
        vtk, vtk_to_numpy = require_vtk()
    except Exception as exc:
        for input_path in input_paths:
            result_queue.put(
                {
                    "input": str(input_path),
                    "records": [{"input": str(input_path), "status": "failed", "error": repr(exc)}],
                }
            )
        return

    for input_path in input_paths:
        try:
            records = process_input(vtk, vtk_to_numpy, input_path, args, factors, calibrated)
        except Exception as exc:
            records = [{"input": str(input_path), "status": "failed", "error": repr(exc)}]
        for record in records:
            record.setdefault("gpu_device", gpu_device)
        result_queue.put({"input": str(input_path), "records": records})


def ignore_sigint_in_worker():
    # The parent owns Ctrl+C handling and terminates the pool explicitly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main() -> int:
    args = parse_args()
    if args.feature_angle <= 0.0 or args.feature_angle >= 180.0:
        raise ValueError("--feature-angle must be in (0, 180).")
    if args.isotropic_iterations <= 0:
        raise ValueError("--isotropic-iterations must be positive.")
    if args.isotropic_iso_tries <= 0:
        raise ValueError("--isotropic-iso-tries must be positive.")
    if args.isotropic_seed_triangle_multiplier < 1.0:
        raise ValueError("--isotropic-seed-triangle-multiplier must be at least 1.0.")
    if args.gpu_adjustments <= 0:
        raise ValueError("--gpu-adjustments must be positive.")
    if args.gpu_face_chunk_size <= 0:
        raise ValueError("--gpu-face-chunk-size must be positive.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    gpu_devices = []
    if args.gpu_devices:
        gpu_devices = [item.strip() for item in str(args.gpu_devices).split(",") if item.strip()]
    elif args.method == "isotropic_gpu":
        gpu_devices = [str(args.gpu_device).strip()]
    if args.method == "isotropic_gpu":
        if not gpu_devices or any(not item.startswith("cuda") for item in gpu_devices):
            raise ValueError("--gpu-devices/--gpu-device must contain CUDA devices, e.g. cuda:0,cuda:1.")
        if len(set(gpu_devices)) != len(gpu_devices):
            raise ValueError("--gpu-devices must not contain duplicate CUDA devices.")
        if args.workers != len(gpu_devices):
            print(
                f"[gpu] Using one fixed worker per CUDA device: workers={len(gpu_devices)} "
                f"(requested {args.workers}).",
                flush=True,
            )
            args.workers = len(gpu_devices)
    worker_caps = {
        "isotropic_remeshing": 4,
        "voxel_quadric_clustering": 2,
    }
    worker_cap = worker_caps.get(args.method)
    if worker_cap is not None and args.workers > worker_cap:
        print(
            f"[memory] Capping {args.method} workers from {args.workers} to {worker_cap}.",
            flush=True,
        )
        args.workers = worker_cap
    factors = parse_factors(args.factors)
    calibrated = {
        5: parse_divisions(args.divisions_5),
        10: parse_divisions(args.divisions_10),
        20: parse_divisions(args.divisions_20),
        40: parse_divisions(args.divisions_40),
    }
    inputs = discover_inputs(args.input_dir, args.runs, args.limit, args.input_glob)
    vtk, _vtk_to_numpy = require_vtk()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Method: {args.method}; factors: {factors}; inputs: {len(inputs)}")
    print(f"VTK version: {vtk.vtkVersion.GetVTKVersion()}")
    print(
        f"Processing with {args.workers} worker(s); each worker loads one input once "
        "and chains all requested factors."
    )
    if args.method == "isotropic_gpu":
        print(f"CUDA workers: {', '.join(gpu_devices)} (one process pinned to each device)")

    all_records = []
    failed = []
    wall_t0 = time.perf_counter()
    if args.workers == 1:
        for input_path in tqdm(inputs, desc="VTP geometries", dynamic_ncols=True):
            try:
                all_records.extend(process_input(vtk, _vtk_to_numpy, input_path, args, factors, calibrated))
            except Exception as exc:
                record = {"input": str(input_path), "status": "failed", "error": repr(exc)}
                all_records.append(record)
                failed.append(record)
                print(f"[failed] {input_path}: {exc}", flush=True)
                if args.fail_fast:
                    break
    elif args.method == "isotropic_gpu":
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        assignments = [inputs[index::len(gpu_devices)] for index in range(len(gpu_devices))]
        processes = [
            context.Process(
                target=process_gpu_worker,
                args=(gpu_device, assigned_inputs, args, factors, calibrated, result_queue),
            )
            for gpu_device, assigned_inputs in zip(gpu_devices, assignments)
            if assigned_inputs
        ]
        for process in processes:
            process.start()
        remaining = len(inputs)
        try:
            with tqdm(total=len(inputs), desc="VTP geometries", dynamic_ncols=True) as progress:
                while remaining:
                    try:
                        message = result_queue.get(timeout=1.0)
                    except queue.Empty:
                        if not any(process.is_alive() for process in processes):
                            raise RuntimeError("All GPU workers exited before reporting every geometry.")
                        continue
                    records = message["records"]
                    all_records.extend(records)
                    newly_failed = [record for record in records if record.get("status") == "failed"]
                    if newly_failed:
                        failed.extend(newly_failed)
                        if args.fail_fast:
                            raise RuntimeError(newly_failed[0].get("error", "GPU worker failed."))
                    remaining -= 1
                    progress.update(1)
        except KeyboardInterrupt:
            print("\n[interrupt] Ctrl+C received; terminating GPU workers...", flush=True)
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise SystemExit(130)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            result_queue.close()
            result_queue.join_thread()
    else:
        context = mp.get_context("spawn")
        payloads = [(input_path, args, factors, calibrated) for input_path in inputs]
        pool = context.Pool(
            processes=args.workers,
            initializer=ignore_sigint_in_worker,
            maxtasksperchild=1,
        )
        try:
            progress = tqdm(
                pool.imap_unordered(process_worker, payloads, chunksize=1),
                total=len(payloads),
                desc="VTP geometries",
                dynamic_ncols=True,
            )
            for records in progress:
                all_records.extend(records)
                newly_failed = [record for record in records if record.get("status") == "failed"]
                if newly_failed:
                    failed.extend(newly_failed)
                    if args.fail_fast:
                        pool.terminate()
                        pool.join()
                        pool = None
                        break
        except KeyboardInterrupt:
            print("\n[interrupt] Ctrl+C received; terminating VTP workers...", flush=True)
            pool.terminate()
            pool.join()
            pool = None
            raise SystemExit(130)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "method": args.method,
        "factors": factors,
        "feature_angle": args.feature_angle,
        "isotropic_iterations": args.isotropic_iterations,
        "isotropic_iso_tries": args.isotropic_iso_tries,
        "isotropic_seed_triangle_multiplier": args.isotropic_seed_triangle_multiplier,
        "gpu_device": args.gpu_device,
        "gpu_devices": gpu_devices if args.method == "isotropic_gpu" else [],
        "gpu_adjustments": args.gpu_adjustments,
        "gpu_face_chunk_size": args.gpu_face_chunk_size,
        "isotropic_backend": "pyacvd_acvd" if args.method == "isotropic_remeshing" else (
            "cuda_voxel_centroid" if args.method == "isotropic_gpu" else None
        ),
        "validate_output": bool(args.validate_output),
        "validate_topology": bool(args.validate_topology),
        "wall_seconds": time.perf_counter() - wall_t0,
        "records": all_records,
        "failed_count": len(failed),
    }
    summary_path = args.output_dir / "decimation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary: {summary_path}")
    print(f"Finished in {summary['wall_seconds'] / 3600.0:.2f} h; failures={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
