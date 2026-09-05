#!/usr/bin/env python3
"""Quantitatively audit geometric preservation of remeshed VTP surfaces.

Distances are symmetric, point-to-triangle surface distances evaluated with
VTK's compiled distance filter.  Normal deviation uses nearest sampled surface
vertices and is explicitly labelled as an orientation-agnostic approximation.
The script deliberately separates mesh distortion from density redistribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("drivaerml", "pump", "heat_exchanger"), required=True)
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing the original VTPs.")
    parser.add_argument("--remesh-dir", type=Path, required=True, help="Root containing method/case/remeshed.vtp.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", default="voxel,quadric,feature")
    parser.add_argument("--factors", default="5,10")
    parser.add_argument("--max-cases", type=int, default=0, help="0 validates every available remeshed case.")
    parser.add_argument("--distance-samples", type=int, default=50000)
    parser.add_argument("--normal-samples", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_vtk():
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("VTK is required for remesh validation.") from exc
    return vtk, vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray


def read_polydata(vtk, path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Unable to read nonempty VTP: {path}")
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(poly)
    triangle.PassVertsOn()
    triangle.PassLinesOff()
    triangle.Update()
    return triangle.GetOutput()


def sampled_vertex_polydata(vtk, numpy_to_vtk, numpy_to_vtkIdTypeArray, points: np.ndarray):
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    poly.SetPoints(vtk_points)
    packed = np.empty(points.shape[0] * 2, dtype=np.int64)
    packed[0::2] = 1
    packed[1::2] = np.arange(points.shape[0], dtype=np.int64)
    verts = vtk.vtkCellArray()
    verts.SetCells(points.shape[0], numpy_to_vtkIdTypeArray(packed, deep=True))
    poly.SetVerts(verts)
    return poly


def sample_indices(count: int, budget: int, seed: int) -> np.ndarray:
    if count <= budget:
        return np.arange(count, dtype=np.int64)
    return np.random.default_rng(seed).choice(count, size=budget, replace=False)


def point_to_surface_distances(vtk, vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray, points: np.ndarray, target):
    samples = sampled_vertex_polydata(vtk, numpy_to_vtk, numpy_to_vtkIdTypeArray, points)
    distance = vtk.vtkDistancePolyDataFilter()
    distance.SetInputData(0, samples)
    distance.SetInputData(1, target)
    distance.SignedDistanceOff()
    distance.ComputeSecondDistanceOff()
    distance.Update()
    values = distance.GetOutput().GetPointData().GetArray("Distance")
    if values is None:
        raise RuntimeError("VTK distance filter did not produce a Distance array.")
    return np.abs(np.asarray(vtk_to_numpy(values), dtype=np.float64))


def point_normals(vtk, vtk_to_numpy, poly):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOff()
    normals.Update()
    output = normals.GetOutput()
    data = output.GetPointData().GetNormals()
    if data is None:
        raise RuntimeError("Could not compute point normals.")
    return output, np.asarray(vtk_to_numpy(data), dtype=np.float64)


def sampled_normal_deviation(vtk, vtk_to_numpy, source, target, budget: int, seed: int) -> np.ndarray:
    source_with_normals, source_normals = point_normals(vtk, vtk_to_numpy, source)
    target_with_normals, target_normals = point_normals(vtk, vtk_to_numpy, target)
    source_points = np.asarray(vtk_to_numpy(source_with_normals.GetPoints().GetData()), dtype=np.float64)
    target_points = np.asarray(vtk_to_numpy(target_with_normals.GetPoints().GetData()), dtype=np.float64)
    indices = sample_indices(len(source_points), budget, seed)
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(target_with_normals)
    locator.BuildLocator()
    nearest = np.fromiter((locator.FindClosestPoint(source_points[index]) for index in indices), dtype=np.int64, count=len(indices))
    source_unit = source_normals[indices] / np.maximum(np.linalg.norm(source_normals[indices], axis=1, keepdims=True), 1.0e-12)
    target_unit = target_normals[nearest] / np.maximum(np.linalg.norm(target_normals[nearest], axis=1, keepdims=True), 1.0e-12)
    # Global winding may legitimately differ after simplification. This reports
    # local orientation mismatch independent of a global sign flip.
    cosine = np.clip(np.abs(np.einsum("ij,ij->i", source_unit, target_unit)), 0.0, 1.0)
    return np.degrees(np.arccos(cosine))


def topology_counts(vtk, poly) -> dict[str, int]:
    def count(kind: str) -> int:
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(poly)
        edges.BoundaryEdgesOff(); edges.NonManifoldEdgesOff(); edges.FeatureEdgesOff(); edges.ManifoldEdgesOff()
        if kind == "boundary":
            edges.BoundaryEdgesOn()
        else:
            edges.NonManifoldEdgesOn()
        edges.ColoringOff()
        edges.Update()
        return int(edges.GetOutput().GetNumberOfCells())
    return {"boundary_edges": count("boundary"), "nonmanifold_edges": count("nonmanifold")}


def surface_area(vtk, poly) -> float:
    mass = vtk.vtkMassProperties()
    mass.SetInputData(poly)
    mass.Update()
    return float(mass.GetSurfaceArea())


def original_path(dataset: str, source_dir: Path, remesh: Path) -> Path:
    case_name = remesh.parent.name
    stem = re.sub(r"_faces_div\d+$", "", remesh.stem)
    if dataset == "drivaerml":
        return source_dir / case_name / f"{stem}.vtp"
    if dataset == "pump":
        return source_dir / case_name / f"{stem}.vtp"
    if dataset == "heat_exchanger":
        return source_dir / case_name / f"{stem}.vtp"
    raise AssertionError(dataset)


def validate_one(payload: tuple[str, str, str, str, int, int, int]) -> dict[str, object]:
    dataset, source_text, remesh_text, method, factor, distance_samples, normal_samples = payload
    vtk, vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray = require_vtk()
    source_path, remesh_path = Path(source_text), Path(remesh_text)
    source = read_polydata(vtk, source_path)
    output = read_polydata(vtk, remesh_path)
    source_points = np.asarray(vtk_to_numpy(source.GetPoints().GetData()), dtype=np.float64)
    output_points = np.asarray(vtk_to_numpy(output.GetPoints().GetData()), dtype=np.float64)
    seed = abs(hash((str(remesh_path), factor))) % (2**31 - 1)
    source_sample = source_points[sample_indices(len(source_points), distance_samples, seed)]
    output_sample = output_points[sample_indices(len(output_points), distance_samples, seed + 1)]
    forward = point_to_surface_distances(vtk, vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray, source_sample, output)
    reverse = point_to_surface_distances(vtk, vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray, output_sample, source)
    all_distance = np.concatenate((forward, reverse))
    normal_forward = sampled_normal_deviation(vtk, vtk_to_numpy, source, output, normal_samples, seed + 2)
    normal_reverse = sampled_normal_deviation(vtk, vtk_to_numpy, output, source, normal_samples, seed + 3)
    all_normals = np.concatenate((normal_forward, normal_reverse))
    bounds = source.GetBounds()
    diagonal = float(np.linalg.norm(np.asarray([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]], dtype=np.float64)))
    source_area, output_area = surface_area(vtk, source), surface_area(vtk, output)
    source_topology, output_topology = topology_counts(vtk, source), topology_counts(vtk, output)
    return {
        "dataset": dataset,
        "case": remesh_path.parent.name,
        "method": method,
        "factor": factor,
        "source_vtp": str(source_path),
        "remesh_vtp": str(remesh_path),
        "source_vertices": int(source.GetNumberOfPoints()),
        "remesh_vertices": int(output.GetNumberOfPoints()),
        "source_triangles": int(source.GetNumberOfPolys()),
        "remesh_triangles": int(output.GetNumberOfPolys()),
        "source_area": source_area,
        "remesh_area": output_area,
        "area_change_percent": 100.0 * (output_area - source_area) / max(source_area, 1.0e-12),
        "bounding_box_diagonal": diagonal,
        "symmetric_chamfer_mean": float(all_distance.mean()),
        "symmetric_chamfer_p95": float(np.quantile(all_distance, 0.95)),
        "symmetric_hausdorff_p99_sampled": float(np.quantile(all_distance, 0.99)),
        "symmetric_hausdorff_max_sampled": float(all_distance.max()),
        "chamfer_mean_percent_bbox_diagonal": 100.0 * float(all_distance.mean()) / max(diagonal, 1.0e-12),
        "normal_deviation_mean_degrees": float(all_normals.mean()),
        "normal_deviation_p95_degrees": float(np.quantile(all_normals, 0.95)),
        "source_boundary_edges": source_topology["boundary_edges"],
        "remesh_boundary_edges": output_topology["boundary_edges"],
        "source_nonmanifold_edges": source_topology["nonmanifold_edges"],
        "remesh_nonmanifold_edges": output_topology["nonmanifold_edges"],
        "distance_samples_per_direction": min(distance_samples, len(source_points), len(output_points)),
        "normal_samples_per_direction": min(normal_samples, len(source_points), len(output_points)),
    }


def main() -> int:
    args = parse_args()
    methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    factors = {int(item.strip()) for item in args.factors.split(",") if item.strip()}
    candidates = []
    for path in sorted(args.remesh_dir.glob("*/*/*.vtp")):
        method = path.parent.parent.name
        match = re.search(r"_faces_div(\d+)\.vtp$", path.name)
        if method not in methods or match is None or int(match.group(1)) not in factors:
            continue
        source = original_path(args.dataset, args.source_dir, path)
        if source.is_file():
            candidates.append((source, path, method, int(match.group(1))))
    case_names = sorted({item[1].parent.name for item in candidates})
    if args.max_cases > 0:
        if args.max_cases > len(case_names):
            raise ValueError(f"Requested {args.max_cases} cases, but only {len(case_names)} complete remeshed cases exist.")
        selected_cases = set(np.random.default_rng(args.seed).choice(case_names, size=args.max_cases, replace=False).tolist())
        candidates = [item for item in candidates if item[1].parent.name in selected_cases]
    else:
        selected_cases = set(case_names)
    if not candidates:
        raise FileNotFoundError("No matching original/remeshed VTP pairs found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.dataset, str(source), str(remesh), method, factor, args.distance_samples, args.normal_samples) for source, remesh, method, factor in candidates]
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        future_map = {pool.submit(validate_one, job): job for job in jobs}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Validating remesh geometry"):
            job = future_map[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append({"source": job[1], "remesh": job[2], "error": repr(exc)})
    records.sort(key=lambda item: (str(item["case"]), str(item["method"]), int(item["factor"])))
    with (args.output_dir / "remesh_geometry_per_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else ["dataset"])
        writer.writeheader(); writer.writerows(records)
    summaries = []
    for method in sorted(methods):
        for factor in sorted(factors):
            subset = [record for record in records if record["method"] == method and record["factor"] == factor]
            if not subset:
                continue
            summaries.append({
                "method": method, "factor": factor, "cases": len(subset),
                "chamfer_mean_percent_bbox_diagonal_mean": float(np.mean([row["chamfer_mean_percent_bbox_diagonal"] for row in subset])),
                "chamfer_mean_percent_bbox_diagonal_p95": float(np.quantile([row["chamfer_mean_percent_bbox_diagonal"] for row in subset], 0.95)),
                "hausdorff_p99_percent_bbox_diagonal_mean": float(np.mean([100.0 * row["symmetric_hausdorff_p99_sampled"] / max(row["bounding_box_diagonal"], 1.0e-12) for row in subset])),
                "absolute_area_change_percent_mean": float(np.mean([abs(row["area_change_percent"]) for row in subset])),
                "normal_deviation_mean_degrees_mean": float(np.mean([row["normal_deviation_mean_degrees"] for row in subset])),
                "normal_deviation_p95_degrees_mean": float(np.mean([row["normal_deviation_p95_degrees"] for row in subset])),
                "mean_triangle_reduction": float(np.mean([row["source_triangles"] / max(row["remesh_triangles"], 1) for row in subset])),
                "topology_changed_cases": int(sum(
                    row["source_boundary_edges"] != row["remesh_boundary_edges"] or row["source_nonmanifold_edges"] != row["remesh_nonmanifold_edges"]
                    for row in subset
                )),
            })
    with (args.output_dir / "remesh_geometry_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]) if summaries else ["method"])
        writer.writeheader(); writer.writerows(summaries)
    (args.output_dir / "remesh_geometry_validation.json").write_text(json.dumps({
        "dataset": args.dataset,
        "source_dir": str(args.source_dir),
        "remesh_dir": str(args.remesh_dir),
        "methods": sorted(methods), "factors": sorted(factors),
        "distance_definition": "symmetric sampled point-to-triangle surface distance via vtkDistancePolyDataFilter",
        "normal_definition": "orientation-agnostic nearest-vertex normal deviation; approximate diagnostic",
        "validated_cases": sorted(selected_cases), "records": len(records), "failures": failures, "summary": summaries,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(records)} remeshes with {len(failures)} failures; outputs in {args.output_dir}")
    if failures:
        raise RuntimeError(f"{len(failures)} geometry-validation jobs failed; inspect remesh_geometry_validation.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
