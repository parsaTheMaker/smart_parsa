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


def discover_inputs(input_dir: Path, runs: str | None, limit: int | None) -> list[Path]:
    if runs:
        run_ids = []
        for item in runs.split(","):
            item = item.strip()
            if item:
                run_ids.append(int(item))
        paths = [input_dir / f"run_{run_id}" / f"drivaer_{run_id}.vtp" for run_id in sorted(set(run_ids))]
    else:
        paths = []
        for path in sorted(input_dir.glob("run_*/drivaer_*.vtp")):
            if RUN_FILE_RE.fullmatch(path.name):
                paths.append(path)
    paths = [path for path in paths if path.is_file()]
    if limit is not None:
        paths = paths[: max(0, int(limit))]
    if not paths:
        raise FileNotFoundError(f"No DrivAerML VTP inputs found under {input_dir}.")
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
    if args.method == "isotropic_remeshing":
        source_surface_area = surface_area(vtk, original)

    # ACVD is fastest at the coarse targets. Process those first so a long
    # full-mesh input reports useful progress before the finest target runs.
    factor_sequence = sorted(factors, reverse=True) if args.method == "isotropic_remeshing" else factors
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
        else:
            decimated = run_quadric_decimation(vtk, original, factor)
        decimate_seconds = time.perf_counter() - t0

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
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
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
    inputs = discover_inputs(args.input_dir, args.runs, args.limit)
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
        "isotropic_backend": "pyacvd_acvd",
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
