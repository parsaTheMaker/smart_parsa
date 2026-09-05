#!/usr/bin/env python3
"""Decode a full DrivAerML volume cloud for a QEM input and render dense slices.

This publication diagnostic is intentionally independent of the fixed-budget
evaluation.  It retains the trained encoder budget but streams every stored
preprocessed volume query through the decoder, then writes both a portable VTK
cloud and high-resolution pressure and velocity-magnitude slices. It never changes a reported
benchmark metric.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SMART_ROOT.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2  # noqa: E402
from scripts.compare_drivaerml_sampling_invariance import build_model, load_cfg  # noqa: E402
from scripts.compare_shift_endpoint_strategies import load_vtp_points  # noqa: E402


VOLUME_FIELDS = ("pressure", "velocity_x", "velocity_y", "velocity_z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=29)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssdraid/parsa/drivaerml_preprocessed"))
    parser.add_argument("--remesh-root", type=Path, default=Path("/mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4"))
    parser.add_argument("--base-config", default="drivaerml")
    parser.add_argument("--deal-config", default="drivaerml_satloss7_range100")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--deal-checkpoint", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1", help="Base and DeAL device, respectively.")
    parser.add_argument("--encoder-points", type=int, default=131072)
    parser.add_argument("--query-chunk-size", type=int, default=65536)
    parser.add_argument(
        "--native-slice",
        type=Path,
        default=None,
        help="Official DrivAerML y=0 CFD slice carrying native cell fields.",
    )
    parser.add_argument("--error-percentile", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-path", type=Path, required=True)
    return parser.parse_args()


def parse_devices(value: str) -> tuple[torch.device, torch.device]:
    devices = [torch.device(item.strip()) for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one inference device is required.")
    if len(devices) == 1:
        devices *= 2
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were requested but CUDA is unavailable.")
    return devices[0], devices[1]


def normalize_positions(points: np.ndarray, minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
    values = torch.from_numpy(np.array(points, dtype=np.float32, order="C", copy=True))
    return (values - minimum) / torch.clamp(maximum - minimum, min=1.0e-12)


@torch.inference_mode()
def decode_full_volume(
    model: torch.nn.Module,
    device: torch.device,
    geometry_norm: torch.Tensor,
    query_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    chunk_size: int,
    seed: int,
) -> np.ndarray:
    """Encode once and stream every volume query through the trained decoder."""
    if chunk_size <= 0:
        raise ValueError("--query-chunk-size must be positive.")
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context:
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        geometry = geometry_norm.unsqueeze(0).to(device, non_blocking=True)
        output = np.empty((query_norm.shape[0], len(VOLUME_FIELDS)), dtype=np.float32)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            intermediate, latent_pos = model.encode(geometry, None)
            for start in range(0, query_norm.shape[0], chunk_size):
                stop = min(start + chunk_size, query_norm.shape[0])
                query = query_norm[start:stop].unsqueeze(0).to(device, non_blocking=True)
                normalized = model.decode(intermediate, latent_pos, None, query)
                output[start:stop] = normalized[0, :, model.surface_channels :].float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return output * std.reshape(-1).numpy()[None, :] + mean.reshape(-1).numpy()[None, :]


def write_vtk(path: Path, points: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float32), deep=True))
    poly.SetPoints(vtk_points)
    count = points.shape[0]
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(count + 1, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.arange(count, dtype=np.int64), deep=True),
    )
    poly.SetVerts(cells)
    for name, values in arrays.items():
        vtk_array = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
        vtk_array.SetName(name)
        poly.GetPointData().AddArray(vtk_array)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise IOError(f"Could not write {path}")


def write_slice_vtp(
    path: Path,
    poly: vtk.vtkPolyData,
    arrays: dict[str, np.ndarray],
) -> None:
    output = vtk.vtkPolyData()
    output.DeepCopy(poly)
    output.GetPointData().Initialize()
    for name, values in arrays.items():
        vtk_array = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
        vtk_array.SetName(name)
        output.GetPointData().AddArray(vtk_array)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(output)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise IOError(f"Could not write {path}")


def read_native_slice(path: Path) -> tuple[vtk.vtkPolyData, np.ndarray, np.ndarray, np.ndarray]:
    """Read and triangulate the official CFD slice, preserving native fields."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    if reader.GetOutput().GetNumberOfPoints() == 0:
        raise RuntimeError(f"Native slice is empty: {path}")
    cell_to_point = vtk.vtkCellDataToPointData()
    cell_to_point.SetInputConnection(reader.GetOutputPort())
    cell_to_point.PassCellDataOff()
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(cell_to_point.GetOutputPort())
    triangulate.PassLinesOff()
    triangulate.PassVertsOff()
    triangulate.Update()
    poly = vtk.vtkPolyData()
    poly.DeepCopy(triangulate.GetOutput())
    points = np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float32)
    pressure = np.asarray(vtk_to_numpy(poly.GetPointData().GetArray("pMeanTrim")), dtype=np.float32)
    velocity = np.asarray(vtk_to_numpy(poly.GetPointData().GetArray("UMeanTrim")), dtype=np.float32)
    connectivity = np.asarray(vtk_to_numpy(poly.GetPolys().GetConnectivityArray()), dtype=np.int64)
    if connectivity.size % 3:
        raise RuntimeError("Triangulated native slice contains non-triangular cells.")
    triangles = connectivity.reshape(-1, 3)
    if not all(np.isfinite(values).all() for values in (points, pressure, velocity)):
        raise RuntimeError("Native CFD slice contains non-finite values.")
    return poly, points, pressure, velocity


def render_native_slice(
    axis: plt.Axes,
    triangulation: mtri.Triangulation,
    values: np.ndarray,
    *,
    norm: matplotlib.colors.Normalize,
    cmap: str,
    x_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
) -> None:
    axis.tripcolor(
        triangulation,
        values,
        shading="gouraud",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    axis.set_xlim(*x_bounds)
    axis.set_ylim(*z_bounds)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()


def plot_slice(
    path: Path,
    points: np.ndarray,
    triangles: np.ndarray,
    ground_truth: np.ndarray,
    base: np.ndarray,
    deal: np.ndarray,
    error_percentile: float,
) -> int:
    if not 50.0 <= error_percentile < 100.0:
        raise ValueError("--error-percentile must lie in [50, 100).")
    # Expand the former view by 20% on both x sides and by 25% upward, capped
    # at the preprocessed inference domain rather than extrapolating beyond it.
    x_bounds = (-2.0, 5.64)
    z_bounds = (-0.32, 1.7675)
    visible = (
        (points[:, 0] >= x_bounds[0])
        & (points[:, 0] <= x_bounds[1])
        & (points[:, 2] >= z_bounds[0])
        & (points[:, 2] <= z_bounds[1])
    )
    triangulation = mtri.Triangulation(points[:, 0], points[:, 2], triangles)
    field_specs = (
        ("Pressure", ground_truth[:, 0], base[:, 0], deal[:, 0]),
        (
            "Velocity magnitude",
            np.linalg.norm(ground_truth[:, 1:4], axis=1),
            np.linalg.norm(base[:, 1:4], axis=1),
            np.linalg.norm(deal[:, 1:4], axis=1),
        ),
    )
    fig = plt.figure(figsize=(7.15, 5.55))
    grid = fig.add_gridspec(
        4, 12, left=0.008, right=0.995, top=0.965, bottom=0.045,
        hspace=0.20, wspace=0.025,
    )
    panel = 0
    for field_index, (field_label, ground_truth_values, base_values, deal_values) in enumerate(field_specs):
        values = (ground_truth_values, base_values, deal_values)
        errors = (np.abs(base_values - ground_truth_values), np.abs(deal_values - ground_truth_values))
        field_norm = matplotlib.colors.Normalize(
            vmin=float(np.percentile(np.concatenate([value[visible] for value in values]), 1.0)),
            vmax=float(np.percentile(np.concatenate([value[visible] for value in values]), 99.0)),
            clip=True,
        )
        error_norm = matplotlib.colors.Normalize(
            vmin=0.0,
            vmax=max(
                float(
                    np.percentile(
                        np.concatenate([error[visible] for error in errors]),
                        error_percentile,
                    )
                ),
                1.0e-12,
            ),
            clip=True,
        )
        top_axes = (
            fig.add_subplot(grid[2 * field_index, 0:4]),
            fig.add_subplot(grid[2 * field_index, 4:8]),
            fig.add_subplot(grid[2 * field_index, 8:12]),
        )
        bottom_axes = (
            fig.add_subplot(grid[2 * field_index + 1, 0:6]),
            fig.add_subplot(grid[2 * field_index + 1, 6:12]),
        )
        titles = ("Ground truth", "Base", "DeAL", "Base error", "DeAL error")
        for axis, values_here, title, cmap, norm in zip(
            (*top_axes, *bottom_axes),
            (*values, *errors),
            titles,
            ("turbo", "turbo", "turbo", "magma", "magma"),
            (field_norm, field_norm, field_norm, error_norm, error_norm),
        ):
            render_native_slice(
                axis,
                triangulation,
                values_here,
                norm=norm,
                cmap=cmap,
                x_bounds=x_bounds,
                z_bounds=z_bounds,
            )
            axis.set_title(f"({chr(97 + panel)}) {title}", pad=2.2, fontsize=8.6, fontweight="normal")
            panel += 1
        field_bar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=field_norm, cmap="turbo"),
            ax=top_axes, orientation="horizontal", fraction=0.048, pad=0.018, aspect=42,
        )
        field_bar.set_label(field_label, fontsize=8.2, labelpad=1)
        field_bar.ax.tick_params(labelsize=7.3, pad=1, length=2)
        error_bar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=error_norm, cmap="magma"),
            ax=bottom_axes, orientation="horizontal", fraction=0.048, pad=0.015, aspect=42,
        )
        error_bar.set_label(
            f"Absolute {field_label.lower()} error (linear, clipped at p{error_percentile:g})",
            fontsize=8.2,
            labelpad=1,
        )
        error_bar.ax.tick_params(labelsize=7.3, pad=1, length=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=900, bbox_inches="tight", pad_inches=0.012, transparent=True)
    plt.close(fig)
    return int(np.count_nonzero(visible))


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    run_dir = data_root / f"run_{args.run_id}"
    qem_path = args.remesh_root.expanduser().resolve() / "quadric" / f"run_{args.run_id}" / f"drivaer_{args.run_id}_faces_div10.vtp"
    native_slice_path = (
        args.native_slice.expanduser().resolve()
        if args.native_slice is not None
        else Path(
            f"/mnt/ssdraid/parsa/drivaerml_native_fields/run_{args.run_id}/"
            "slices/yNormal_p00000.vtp"
        )
    )
    for path in (run_dir, qem_path, native_slice_path, args.base_checkpoint, args.deal_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    base_device, deal_device = parse_devices(args.devices)
    print(f"Full-volume inference: Base->{base_device}, DeAL->{deal_device}", flush=True)

    dataset = AhmedMLDatasetV2(
        saved_folder=str(data_root), if_test=True, geometry_points=0, surface_points=0,
        volume_points=0, scale_positions=False, require_preprocessed=True,
    )
    volume_points = np.asarray(np.load(run_dir / "volume_coords.npy", mmap_mode="r"), dtype=np.float32)
    volume_gt = np.concatenate(
        (
            np.asarray(np.load(run_dir / "volume_pMeanTrim.npy", mmap_mode="r"), dtype=np.float32)[:, None],
            np.asarray(np.load(run_dir / "volume_UMeanTrim.npy", mmap_mode="r"), dtype=np.float32),
        ),
        axis=1,
    )
    slice_poly, slice_points, slice_pressure, slice_velocity = read_native_slice(native_slice_path)
    slice_gt = np.concatenate((slice_pressure[:, None], slice_velocity), axis=1)
    geometry_source = load_vtp_points(qem_path)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, args.run_id, 901]))
    indices = rng.choice(geometry_source.shape[0], size=min(args.encoder_points, geometry_source.shape[0]), replace=False)
    geometry = np.ascontiguousarray(geometry_source[np.sort(indices)], dtype=np.float32)
    geometry_norm = normalize_positions(geometry, dataset.min_pos.cpu(), dataset.max_pos.cpu())
    volume_norm = normalize_positions(volume_points, dataset.min_pos.cpu(), dataset.max_pos.cpu())
    slice_norm = normalize_positions(slice_points, dataset.min_pos.cpu(), dataset.max_pos.cpu())
    all_queries_norm = torch.cat((volume_norm, slice_norm), dim=0)

    base_cfg = load_cfg(args.base_config)
    deal_cfg = load_cfg(args.deal_config)
    base_model = build_model(base_cfg, str(args.base_checkpoint), base_device, args.query_chunk_size).to(base_device)
    deal_model = build_model(deal_cfg, str(args.deal_checkpoint), deal_device, args.query_chunk_size).to(deal_device)
    mean = dataset.mean_vol_data.detach().cpu().float().reshape(-1)
    std = dataset.std_vol_data.detach().cpu().float().reshape(-1)
    with ThreadPoolExecutor(max_workers=2 if base_device != deal_device else 1) as pool:
        base_future = pool.submit(decode_full_volume, base_model, base_device, geometry_norm, all_queries_norm, mean, std, args.query_chunk_size, args.seed + 101)
        deal_future = pool.submit(decode_full_volume, deal_model, deal_device, geometry_norm, all_queries_norm, mean, std, args.query_chunk_size, args.seed + 101)
        base_all = base_future.result()
        deal_all = deal_future.result()
    volume_count = volume_points.shape[0]
    base, base_slice = base_all[:volume_count], base_all[volume_count:]
    deal, deal_slice = deal_all[:volume_count], deal_all[volume_count:]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_path = output_dir / f"drivaerml_run_{args.run_id}_qem10_full_volume_predictions.vtp"
    write_vtk(
        vtk_path,
        volume_points,
        {
            "ground_truth_pressure": volume_gt[:, 0],
            "base_pressure": base[:, 0],
            "deal_pressure": deal[:, 0],
            "base_absolute_error_pressure": np.abs(base[:, 0] - volume_gt[:, 0]),
            "deal_absolute_error_pressure": np.abs(deal[:, 0] - volume_gt[:, 0]),
            "ground_truth_velocity": volume_gt[:, 1:4],
            "base_velocity": base[:, 1:4],
            "deal_velocity": deal[:, 1:4],
            "ground_truth_velocity_magnitude": np.linalg.norm(volume_gt[:, 1:4], axis=1),
            "base_velocity_magnitude": np.linalg.norm(base[:, 1:4], axis=1),
            "deal_velocity_magnitude": np.linalg.norm(deal[:, 1:4], axis=1),
            "base_absolute_error_velocity_magnitude": np.abs(
                np.linalg.norm(base[:, 1:4], axis=1) - np.linalg.norm(volume_gt[:, 1:4], axis=1)
            ),
            "deal_absolute_error_velocity_magnitude": np.abs(
                np.linalg.norm(deal[:, 1:4], axis=1) - np.linalg.norm(volume_gt[:, 1:4], axis=1)
            ),
        },
    )
    slice_vtp_path = output_dir / f"drivaerml_run_{args.run_id}_qem10_native_y0_slice_predictions.vtp"
    write_slice_vtp(
        slice_vtp_path,
        slice_poly,
        {
            "ground_truth_pressure": slice_gt[:, 0],
            "base_pressure": base_slice[:, 0],
            "deal_pressure": deal_slice[:, 0],
            "base_absolute_error_pressure": np.abs(base_slice[:, 0] - slice_gt[:, 0]),
            "deal_absolute_error_pressure": np.abs(deal_slice[:, 0] - slice_gt[:, 0]),
            "ground_truth_velocity": slice_gt[:, 1:4],
            "base_velocity": base_slice[:, 1:4],
            "deal_velocity": deal_slice[:, 1:4],
            "ground_truth_velocity_magnitude": np.linalg.norm(slice_gt[:, 1:4], axis=1),
            "base_velocity_magnitude": np.linalg.norm(base_slice[:, 1:4], axis=1),
            "deal_velocity_magnitude": np.linalg.norm(deal_slice[:, 1:4], axis=1),
            "base_absolute_error_velocity_magnitude": np.abs(
                np.linalg.norm(base_slice[:, 1:4], axis=1)
                - np.linalg.norm(slice_gt[:, 1:4], axis=1)
            ),
            "deal_absolute_error_velocity_magnitude": np.abs(
                np.linalg.norm(deal_slice[:, 1:4], axis=1)
                - np.linalg.norm(slice_gt[:, 1:4], axis=1)
            ),
        },
    )
    triangles = np.asarray(
        vtk_to_numpy(slice_poly.GetPolys().GetConnectivityArray()), dtype=np.int64
    ).reshape(-1, 3)
    slice_count = plot_slice(
        args.figure_path.expanduser().resolve(),
        slice_points,
        triangles,
        slice_gt,
        base_slice,
        deal_slice,
        args.error_percentile,
    )
    summary = {
        "run_id": int(args.run_id),
        "input": "QEM 10x remesh",
        "encoder_points": int(geometry.shape[0]),
        "volume_queries": int(volume_points.shape[0]),
        "native_slice_vertices": int(slice_points.shape[0]),
        "native_slice_triangles": int(triangles.shape[0]),
        "visible_slice_vertices": slice_count,
        "native_slice_y_range": [float(slice_points[:, 1].min()), float(slice_points[:, 1].max())],
        "native_slice": str(native_slice_path),
        "error_percentile": float(args.error_percentile),
        "query_chunk_size": int(args.query_chunk_size),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "deal_checkpoint": str(args.deal_checkpoint.resolve()),
        "volume_vtp": str(vtk_path),
        "slice_vtp": str(slice_vtp_path),
        "figure": str(args.figure_path.expanduser().resolve()),
    }
    (output_dir / "full_volume_export_summary.json").write_text(__import__("json").dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {vtk_path}, {slice_vtp_path}, and {args.figure_path} "
        f"({volume_points.shape[0]:,} full volume queries; "
        f"{slice_points.shape[0]:,} native slice queries).",
        flush=True,
    )


if __name__ == "__main__":
    main()
