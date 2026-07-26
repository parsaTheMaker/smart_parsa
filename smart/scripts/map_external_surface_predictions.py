#!/usr/bin/env python3
"""Map external SMART predictions from point clouds onto supplied VTK surfaces.

The prediction VTK files contain values evaluated at the external surface
query cloud.  This script transfers those values to the points of the matching
``.vtp`` surface while preserving the supplied surface geometry and arrays.
Coordinates remain in the original physical frame: no centering, scaling, or
per-case normalization is performed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "CFD_audi" / "new_cfds"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "external_surface_smart_vs_satloss6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="cKDTree query workers; -1 uses all available workers.",
    )
    parser.add_argument(
        "--bbox-tolerance",
        type=float,
        default=1.0e-3,
        help="Maximum allowed absolute bounding-box mismatch in physical coordinates.",
    )
    parser.add_argument(
        "--point-tolerance",
        type=float,
        default=1.0e-5,
        help="Pointwise tolerance for direct coordinate/order matching.",
    )
    return parser.parse_args()


def read_polydata(path: Path, xml: bool) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader() if xml else vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    if hasattr(reader, "ReadAllScalarsOn"):
        reader.ReadAllScalarsOn()
    reader.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(reader.GetOutput())
    if output.GetNumberOfPoints() == 0:
        raise ValueError(f"No points were read from {path}")
    return output


def write_polydata(path: Path, polydata: vtk.vtkPolyData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToZLib()
    if writer.Write() != 1:
        raise IOError(f"Could not write {path}")


def point_array(polydata: vtk.vtkPolyData, name: str) -> np.ndarray:
    array = polydata.GetPointData().GetArray(name)
    if array is None:
        raise KeyError(f"Point-data array {name!r} is missing")
    values = np.asarray(vtk_to_numpy(array))
    if values.shape[0] != polydata.GetNumberOfPoints():
        raise ValueError(f"Array {name!r} has {values.shape[0]} values but the surface has {polydata.GetNumberOfPoints()} points")
    return values


def add_point_array(polydata: vtk.vtkPolyData, name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if values.shape[0] != polydata.GetNumberOfPoints():
        raise ValueError(f"Array {name!r} does not match the target point count")
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    vtk_array = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
    vtk_array.SetName(name)
    point_data = polydata.GetPointData()
    point_data.RemoveArray(name)
    point_data.AddArray(vtk_array)


def coordinates(polydata: vtk.vtkPolyData) -> np.ndarray:
    return np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=np.float32)


def bbox(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.min(axis=0), values.max(axis=0)


def make_mapping(
    source_points: np.ndarray,
    target_points: np.ndarray,
    bbox_tolerance: float,
    point_tolerance: float,
    workers: int,
) -> tuple[np.ndarray, dict[str, object]]:
    source_min, source_max = bbox(source_points)
    target_min, target_max = bbox(target_points)
    bbox_delta = np.maximum(np.abs(source_min - target_min), np.abs(source_max - target_max))
    if float(bbox_delta.max()) > float(bbox_tolerance):
        raise ValueError(
            "Prediction and supplied surface are not in the same physical coordinate frame: "
            f"bbox_delta={bbox_delta.tolist()} exceeds tolerance {bbox_tolerance}."
        )

    if source_points.shape == target_points.shape:
        max_point_delta = float(np.max(np.abs(source_points - target_points)))
        if max_point_delta <= float(point_tolerance):
            return np.arange(target_points.shape[0], dtype=np.int64), {
                "method": "direct_index",
                "max_nearest_distance": max_point_delta,
                "bbox_delta": bbox_delta.tolist(),
            }

    tree = cKDTree(source_points)
    distances, indices = tree.query(target_points, k=1, workers=workers)
    distances = np.asarray(distances, dtype=np.float64)
    if not np.isfinite(distances).all():
        raise ValueError("The nearest-neighbor mapping produced non-finite distances")
    return np.asarray(indices, dtype=np.int64), {
        "method": "cKDTree",
        "max_nearest_distance": float(distances.max()),
        "mean_nearest_distance": float(distances.mean()),
        "p99_nearest_distance": float(np.percentile(distances, 99.0)),
        "bbox_delta": bbox_delta.tolist(),
    }


def mapped_values(values: np.ndarray, target_to_source: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return values[target_to_source]


def surface_copy_with_fields(
    surface: vtk.vtkPolyData,
    fields: dict[str, np.ndarray],
) -> vtk.vtkPolyData:
    output = vtk.vtkPolyData()
    output.DeepCopy(surface)
    for name, values in fields.items():
        add_point_array(output, name, values)
    return output


def find_cases(input_dir: Path, results_dir: Path) -> list[tuple[Path, Path]]:
    cases = []
    for case_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        surfaces = sorted(case_dir.glob("*.vtp"))
        result_dir = results_dir / case_dir.name
        if not surfaces:
            print(f"[skip] {case_dir.name}: no .vtp surface found")
            continue
        if not result_dir.is_dir():
            print(f"[skip] {case_dir.name}: result directory is missing: {result_dir}")
            continue
        if len(surfaces) > 1:
            raise ValueError(f"{case_dir}: expected one .vtp surface, found {len(surfaces)}")
        cases.append((surfaces[0], result_dir))
    if not cases:
        raise FileNotFoundError("No input surface/result case pairs were found")
    return cases


def process_case(surface_path: Path, result_dir: Path, args: argparse.Namespace) -> None:
    surface = read_polydata(surface_path, xml=True)
    target_points = coordinates(surface)

    smart_path = result_dir / "smart_surface_pressure.vtk"
    satloss6_path = result_dir / "smart_satloss6_surface_pressure.vtk"
    for path in (smart_path, satloss6_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction VTK: {path}")

    smart = read_polydata(smart_path, xml=False)
    satloss6 = read_polydata(satloss6_path, xml=False)
    smart_points = coordinates(smart)
    satloss6_points = coordinates(satloss6)
    if smart_points.shape != satloss6_points.shape or not np.allclose(smart_points, satloss6_points, atol=args.point_tolerance, rtol=0.0):
        raise ValueError(f"{result_dir.name}: SMART and SATLOSS6 prediction clouds do not share the same coordinates")

    target_to_source, mapping_info = make_mapping(
        smart_points,
        target_points,
        bbox_tolerance=args.bbox_tolerance,
        point_tolerance=args.point_tolerance,
        workers=args.workers,
    )
    if mapping_info["method"] == "cKDTree" and float(mapping_info["max_nearest_distance"]) > args.bbox_tolerance:
        raise ValueError(
            f"{result_dir.name}: nearest surface mapping is too far: "
            f"max distance={mapping_info['max_nearest_distance']:.6g}"
        )

    smart_pred = mapped_values(point_array(smart, "pred_pressure"), target_to_source).reshape(-1)
    satloss6_pred = mapped_values(point_array(satloss6, "pred_pressure"), target_to_source).reshape(-1)

    gt = None
    gt_path = surface_path.parent / "surface_pMeanTrim.npy"
    if gt_path.is_file():
        raw_gt = np.asarray(np.load(gt_path), dtype=np.float32).reshape(-1)
        if raw_gt.shape[0] != smart_points.shape[0]:
            raise ValueError(f"{gt_path}: expected {smart_points.shape[0]} values, got {raw_gt.shape[0]}")
        gt = mapped_values(raw_gt, target_to_source).reshape(-1)
    elif surface.GetPointData().GetArray("pressure_gt") is not None:
        gt = point_array(surface, "pressure_gt").reshape(-1).astype(np.float32, copy=False)
    if gt is None:
        raise FileNotFoundError(f"No surface_pMeanTrim.npy or pressure_gt array found for {surface_path}")

    common = {
        "pressure_gt": gt,
        "smart_pred_pressure": smart_pred,
        "smart_abs_error_pressure": np.abs(smart_pred - gt),
        "satloss6_pred_pressure": satloss6_pred,
        "satloss6_abs_error_pressure": np.abs(satloss6_pred - gt),
        "satloss6_minus_smart_pressure": satloss6_pred - smart_pred,
        "absolute_model_difference_pressure": np.abs(satloss6_pred - smart_pred),
    }
    smart_fields = {
        "pressure_gt": gt,
        "pressure_pred": smart_pred,
        "pressure_abs_error": np.abs(smart_pred - gt),
        "pred_pressure": smart_pred,
        "abs_error_pressure": np.abs(smart_pred - gt),
        "smart_pred_pressure": smart_pred,
        "smart_abs_error_pressure": np.abs(smart_pred - gt),
    }
    satloss6_fields = {
        "pressure_gt": gt,
        "pressure_pred": satloss6_pred,
        "pressure_abs_error": np.abs(satloss6_pred - gt),
        "pred_pressure": satloss6_pred,
        "abs_error_pressure": np.abs(satloss6_pred - gt),
        "satloss6_pred_pressure": satloss6_pred,
        "satloss6_abs_error_pressure": np.abs(satloss6_pred - gt),
    }

    outputs = {
        result_dir / "smart_surface_pressure_on_surface.vtp": smart_fields,
        result_dir / "smart_satloss6_surface_pressure_on_surface.vtp": satloss6_fields,
        result_dir / "smart_vs_satloss6_surface_pressure_on_surface.vtp": common,
    }
    for output_path, fields in outputs.items():
        write_polydata(output_path, surface_copy_with_fields(surface, fields))

    metadata = {
        "surface_input": str(surface_path.resolve()),
        "smart_prediction_input": str(smart_path.resolve()),
        "satloss6_prediction_input": str(satloss6_path.resolve()),
        "surface_points": int(target_points.shape[0]),
        "prediction_points": int(smart_points.shape[0]),
        "surface_bounds": list(map(float, surface.GetBounds())),
        "prediction_bounds": list(map(float, smart.GetBounds())),
        "mapping": mapping_info,
        "normalization": "none; mapping is performed in the original physical coordinate frame",
        "outputs": [str(path.resolve()) for path in outputs],
    }
    with (result_dir / "surface_mapping_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(
        f"[{result_dir.name}] points={target_points.shape[0]} method={mapping_info['method']} "
        f"max_distance={mapping_info['max_nearest_distance']:.6g}"
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if args.bbox_tolerance <= 0.0 or args.point_tolerance <= 0.0:
        raise ValueError("Tolerances must be positive")
    for surface_path, result_dir in find_cases(input_dir, results_dir):
        process_case(surface_path, result_dir, args)


if __name__ == "__main__":
    main()
