#!/usr/bin/env python3
"""Compute external pressure drag/errors and create slide-ready plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = REPO_ROOT / "CFD_audi" / "new_cfds"
RESULTS_ROOT = REPO_ROOT / "results" / "external_surface_smart_vs_satloss6"

COLORS = {
    "gt": "#20252B",
    "smart": "#1677B8",
    "satloss6": "#D65F02",
}


def read_vtp_fields(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"No points found in {path}")
    points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=np.float32)
    fields = {}
    point_data = polydata.GetPointData()
    for name in ("pressure_gt", "pressure_pred"):
        array = point_data.GetArray(name)
        if array is None:
            raise KeyError(f"{path}: missing point-data array {name!r}")
        fields[name] = np.asarray(vtk_to_numpy(array), dtype=np.float64).reshape(-1)
    return points, fields


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(np.percentile(values, 1)),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
    }


def pointwise_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - ground_truth
    absolute = np.abs(error)
    denominator = max(float(np.linalg.norm(ground_truth)), 1.0e-12)
    correlation = np.corrcoef(ground_truth, prediction)[0, 1]
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "relative_l2": float(np.linalg.norm(error) / denominator),
        "bias_prediction_minus_gt": float(error.mean()),
        "max_absolute_error": float(absolute.max()),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "correlation": float(correlation) if np.isfinite(correlation) else None,
    }


def pressure_drag_x(pressure: np.ndarray, normals: np.ndarray, areas: np.ndarray) -> float:
    """Signed pressure force in x: integral(-p * n_x dA)."""
    return float(np.sum(-pressure * normals[:, 0] * areas, dtype=np.float64))


def case_label(name: str) -> str:
    return name.replace("audiCFD_", "").replace("_", " ").title()


def save_pressure_distribution_plot(case_results: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(case_results), figsize=(18, 8), squeeze=False)
    axes = axes[0]
    for axis, result in zip(axes, case_results):
        values = result["pressure_values"]
        all_values = np.concatenate(list(values.values()))
        bins = np.linspace(float(all_values.min()), float(all_values.max()), 140)
        for key, label in (("gt", "Ground truth"), ("smart", "SMART"), ("satloss6", "SMART-SATLOSS6")):
            axis.hist(
                values[key],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=3.0,
                color=COLORS[key],
                label=label,
            )
            axis.axvline(
                np.mean(values[key]),
                color=COLORS[key],
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
            )
        axis.set_title(case_label(result["case"]), fontsize=23, pad=14, weight="bold")
        axis.set_xlabel("Surface pressure", fontsize=20)
        axis.set_ylabel("Probability density", fontsize=20)
        axis.tick_params(axis="both", labelsize=16)
        axis.grid(axis="y", alpha=0.25, linewidth=1.0)
        axis.legend(fontsize=15, frameon=True)
    fig.suptitle("External surface pressure distribution", fontsize=28, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_error_distribution_plot(case_results: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(case_results), figsize=(18, 8), squeeze=False)
    axes = axes[0]
    for axis, result in zip(axes, case_results):
        errors = result["errors"]
        all_errors = np.concatenate(list(errors.values()))
        upper = float(np.percentile(all_errors, 99.9))
        bins = np.linspace(0.0, max(upper, 1.0e-6), 120)
        for key, label in (("smart", "SMART"), ("satloss6", "SMART-SATLOSS6")):
            axis.hist(
                errors[key],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=3.0,
                color=COLORS[key],
                label=label,
            )
        axis.set_title(case_label(result["case"]), fontsize=23, pad=14, weight="bold")
        axis.set_xlabel("Absolute pressure error", fontsize=20)
        axis.set_ylabel("Probability density", fontsize=20)
        axis.tick_params(axis="both", labelsize=16)
        axis.grid(axis="y", alpha=0.25, linewidth=1.0)
        axis.legend(fontsize=16, frameon=True)
    fig.suptitle("External surface pressure error distribution", fontsize=28, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_drag_plot(case_results: list[dict], output_path: Path) -> None:
    labels = [case_label(result["case"]) for result in case_results]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    fig, axis = plt.subplots(figsize=(16, 9))
    for index, (key, label) in enumerate(
        (("gt", "Ground truth"), ("smart", "SMART"), ("satloss6", "SMART-SATLOSS6"))
    ):
        values = [result["drag_force_x"][key] for result in case_results]
        bars = axis.bar(x + (index - 1) * width, values, width, label=label, color=COLORS[key])
        axis.bar_label(bars, fmt="%.1f", padding=5, fontsize=15)
    axis.axhline(0.0, color="#333333", linewidth=1.2)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Pressure drag force in x", fontsize=21)
    axis.set_title("External pressure drag comparison", fontsize=28, weight="bold", pad=18)
    axis.tick_params(axis="both", labelsize=18)
    axis.grid(axis="y", alpha=0.25, linewidth=1.0)
    axis.legend(fontsize=18, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    output_root = RESULTS_ROOT
    case_results = []
    for case_dir in sorted(path for path in INPUT_ROOT.iterdir() if path.is_dir()):
        result_dir = output_root / case_dir.name
        smart_vtp = result_dir / "smart_surface_pressure_on_surface.vtp"
        satloss6_vtp = result_dir / "smart_satloss6_surface_pressure_on_surface.vtp"
        if not smart_vtp.is_file() or not satloss6_vtp.is_file():
            raise FileNotFoundError(f"Missing mapped surface outputs for {case_dir.name}")

        coords = np.asarray(np.load(case_dir / "surface_coords.npy", mmap_mode="r"), dtype=np.float32)
        areas = np.asarray(np.load(case_dir / "surface_areas.npy", mmap_mode="r"), dtype=np.float64).reshape(-1)
        normals = np.asarray(np.load(case_dir / "surface_normals.npy", mmap_mode="r"), dtype=np.float64)
        ground_truth = np.asarray(np.load(case_dir / "surface_pMeanTrim.npy", mmap_mode="r"), dtype=np.float64).reshape(-1)
        if coords.shape[0] != ground_truth.shape[0] or coords.shape[0] != areas.shape[0] or coords.shape[0] != normals.shape[0]:
            raise ValueError(f"{case_dir.name}: surface arrays do not share the same point count")

        smart_points, smart_fields = read_vtp_fields(smart_vtp)
        satloss6_points, satloss6_fields = read_vtp_fields(satloss6_vtp)
        for name, points in (("SMART", smart_points), ("SATLOSS6", satloss6_points)):
            if points.shape != coords.shape or not np.allclose(points, coords, atol=1.0e-5, rtol=0.0):
                raise ValueError(f"{case_dir.name}: {name} surface VTP coordinates do not match source surface coordinates")

        smart = smart_fields["pressure_pred"]
        satloss6 = satloss6_fields["pressure_pred"]
        if not np.allclose(smart_fields["pressure_gt"], ground_truth, atol=1.0e-3, rtol=0.0):
            raise ValueError(f"{case_dir.name}: mapped SMART ground truth does not match surface_pMeanTrim.npy")
        if not np.allclose(satloss6_fields["pressure_gt"], ground_truth, atol=1.0e-3, rtol=0.0):
            raise ValueError(f"{case_dir.name}: mapped SATLOSS6 ground truth does not match surface_pMeanTrim.npy")

        case_result = {
            "case": case_dir.name,
            "surface_points": int(coords.shape[0]),
            "drag_force_x": {
                "gt": pressure_drag_x(ground_truth, normals, areas),
                "smart": pressure_drag_x(smart, normals, areas),
                "satloss6": pressure_drag_x(satloss6, normals, areas),
            },
            "pointwise_pressure_error": {
                "smart": pointwise_metrics(ground_truth, smart),
                "satloss6": pointwise_metrics(ground_truth, satloss6),
            },
            "pressure_values": {"gt": ground_truth, "smart": smart, "satloss6": satloss6},
            "errors": {"smart": np.abs(smart - ground_truth), "satloss6": np.abs(satloss6 - ground_truth)},
            "pressure_summary": {
                "gt": summarize(ground_truth),
                "smart": summarize(smart),
                "satloss6": summarize(satloss6),
            },
        }
        case_results.append(case_result)

    serializable = []
    for result in case_results:
        serializable.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"pressure_values", "errors"}
            }
        )
    with (output_root / "external_pressure_drag_and_error_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "drag_definition": "pressure-only signed x-force integral: sum(-pressure * outward_normal_x * surface_area)",
                "drag_units": "physical pressure times physical surface area; expected N if pressure is Pa and coordinates are m",
                "pointwise_error_definition": "pressure error over every supplied surface point",
                "cases": serializable,
            },
            handle,
            indent=2,
        )

    save_pressure_distribution_plot(case_results, output_root / "external_pressure_distribution_comparison.png")
    save_error_distribution_plot(case_results, output_root / "external_pressure_error_distribution_comparison.png")
    save_drag_plot(case_results, output_root / "external_pressure_drag_comparison.png")

    for result in case_results:
        print(f"\n{case_label(result['case'])} ({result['surface_points']:,} surface points)")
        print("  pressure drag x:")
        for key, label in (("gt", "GT"), ("smart", "SMART"), ("satloss6", "SATLOSS6")):
            print(f"    {label:10s}: {result['drag_force_x'][key]: .8f}")
        print("  pointwise pressure errors:")
        for key, label in (("smart", "SMART"), ("satloss6", "SATLOSS6")):
            metrics = result["pointwise_pressure_error"][key]
            print(
                f"    {label:10s}: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
                f"rel-L2={metrics['relative_l2']:.6f}, p95={metrics['p95_absolute_error']:.6f}, "
                f"max={metrics['max_absolute_error']:.6f}"
            )
    print(f"\nMetrics: {output_root / 'external_pressure_drag_and_error_metrics.json'}")
    print(f"Plots:  {output_root / 'external_pressure_distribution_comparison.png'}")
    print(f"        {output_root / 'external_pressure_error_distribution_comparison.png'}")
    print(f"        {output_root / 'external_pressure_drag_comparison.png'}")


if __name__ == "__main__":
    main()
