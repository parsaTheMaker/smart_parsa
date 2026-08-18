#!/usr/bin/env python3
"""Export exact analytic Toy-SATLOSS examples as triangular VTP and PNG files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vtk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

from generate_toy_satloss_benchmark import radial_geometry, surface_field


def parse_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_params(case_dir: Path) -> dict:
    metadata = json.loads((case_dir / "case_metadata.json").read_text(encoding="utf-8"))
    params = metadata["parameters"]
    for key in ("rotation", "axes", "harmonics", "bump_direction", "density_axis", "density_center"):
        params[key] = np.asarray(params[key], dtype=np.float32)
    return params


def latitude_longitude_mesh(params: dict, theta_count: int, phi_count: int):
    theta = np.linspace(0.0, np.pi, theta_count + 1, dtype=np.float32)[1:-1]
    phi = np.linspace(0.0, 2.0 * np.pi, phi_count, endpoint=False, dtype=np.float32)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    directions = np.stack(
        [np.sin(theta_grid) * np.cos(phi_grid), np.sin(theta_grid) * np.sin(phi_grid), np.cos(theta_grid)], axis=-1
    ).reshape(-1, 3).astype(np.float32)
    interior, radii, local = radial_geometry(directions, params)
    north, north_radius, north_local = radial_geometry(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), params)
    south, south_radius, south_local = radial_geometry(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), params)
    points = np.concatenate([north, interior, south], axis=0)
    point_local = np.concatenate([north_local, local, south_local], axis=0)
    point_radius = np.concatenate([north_radius, radii, south_radius], axis=0)
    surface_values = surface_field(point_local, point_radius, params)[:, 0]
    native_log_density = (
        params["density_amplitude"] * np.sin(2.0 * np.pi * (directions @ params["density_axis"] + params["density_phase"]))
        + params["density_focus"] * np.exp(10.0 * (directions @ params["density_center"] - 1.0))
    )
    native_log_density = np.concatenate([[native_log_density.mean()], native_log_density, [native_log_density.mean()]]).astype(np.float32)

    north_id, south_id = 0, points.shape[0] - 1
    def ring_id(row, col):
        return 1 + row * phi_count + (col % phi_count)
    faces = []
    for column in range(phi_count):
        faces.append([north_id, ring_id(0, column + 1), ring_id(0, column)])
    for row in range(theta_count - 2):
        for column in range(phi_count):
            a, b = ring_id(row, column), ring_id(row, column + 1)
            c, d = ring_id(row + 1, column), ring_id(row + 1, column + 1)
            faces.extend(([a, b, c], [b, d, c]))
    last_row = theta_count - 2
    for column in range(phi_count):
        faces.append([south_id, ring_id(last_row, column), ring_id(last_row, column + 1)])
    return points, np.asarray(faces, dtype=np.int64), surface_values, native_log_density


def make_polydata(points: np.ndarray, faces: np.ndarray | None, arrays: dict[str, np.ndarray]) -> vtk.vtkPolyData:
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    poly.SetPoints(vtk_points)
    if faces is not None:
        cells = vtk.vtkCellArray()
        connectivity = numpy_to_vtkIdTypeArray(np.ascontiguousarray(faces.reshape(-1)), deep=True)
        offsets = numpy_to_vtkIdTypeArray(np.arange(0, faces.size + 1, 3, dtype=np.int64), deep=True)
        cells.SetData(offsets, connectivity)
        poly.SetPolys(cells)
    else:
        cells = vtk.vtkCellArray()
        ids = np.arange(points.shape[0], dtype=np.int64)
        connectivity = numpy_to_vtkIdTypeArray(ids, deep=True)
        offsets = numpy_to_vtkIdTypeArray(np.arange(points.shape[0] + 1, dtype=np.int64), deep=True)
        cells.SetData(offsets, connectivity)
        poly.SetVerts(cells)
    for name, values in arrays.items():
        vtk_array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
        vtk_array.SetName(name)
        poly.GetPointData().AddArray(vtk_array)
    return poly


def write_vtp(path: Path, polydata: vtk.vtkPolyData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()
    writer.SetCompressorTypeToZLib()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTP: {path}")


def save_png(case_id: int, geometry: np.ndarray, surface_points: np.ndarray, surface_values: np.ndarray, volume: np.ndarray, volume_values: np.ndarray, output_path: Path) -> None:
    rng = np.random.default_rng(42 + case_id)
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    panels = (
        (geometry, None, "Native nonuniform encoder cloud", "#243b53"),
        (surface_points, surface_values, "Analytic surface field", None),
        (volume, volume_values, "Analytic volume field", None),
    )
    for index, (points, values, title, color) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        selected = rng.choice(points.shape[0], min(24_000, points.shape[0]), replace=False)
        if values is None:
            ax.scatter(points[selected, 0], points[selected, 1], points[selected, 2], s=1.2, c=color, alpha=0.72, rasterized=True)
        else:
            draw = ax.scatter(points[selected, 0], points[selected, 1], points[selected, 2], s=1.5, c=values[selected], cmap="coolwarm", rasterized=True)
            fig.colorbar(draw, ax=ax, shrink=0.70, pad=0.02)
        ax.set_title(title, fontsize=15, weight="bold")
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    fig.suptitle(f"Toy-SATLOSS analytic case {case_id:05d}", fontsize=18, weight="bold")
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def save_surface_mesh_png(case_id: int, points: np.ndarray, faces: np.ndarray, values: np.ndarray, output_path: Path) -> None:
    """Render actual triangles; this is intentionally not a point-cloud view."""
    # Render the complete mesh: a thinned regular latitude/longitude grid can
    # create visual striping and incorrectly suggest holes in the surface.
    draw_faces = faces
    face_values = values[draw_faces].mean(axis=1)
    norm = plt.Normalize(float(values.min()), float(values.max()))
    collection = Poly3DCollection(points[draw_faces], linewidths=0.0, alpha=1.0)
    collection.set_facecolor(plt.get_cmap("coolwarm")(norm(face_values)))
    fig = plt.figure(figsize=(8.5, 7.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(collection)
    lo, hi = points.min(axis=0), points.max(axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_title(f"Toy-SATLOSS case {case_id:05d}: analytic surface mesh", fontsize=16, weight="bold")
    ax.set_axis_off()
    mapper = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    mapper.set_array(values)
    fig.colorbar(mapper, ax=ax, shrink=0.68, pad=0.01, label="Manufactured surface field")
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/ssdraid/parsa/toy_satloss_benchmark_v1")
    parser.add_argument("--output-dir", default="/home/parsa/smart_parsa/results/toy_satloss_benchmark/examples")
    parser.add_argument("--case-ids", default="0,1,2,3,4")
    parser.add_argument("--theta-count", type=int, default=192)
    parser.add_argument("--phi-count", type=int, default=384)
    args = parser.parse_args()
    if args.theta_count < 4 or args.phi_count < 8:
        raise ValueError("theta-count must be >=4 and phi-count must be >=8.")
    root, output_dir = Path(args.data_root).expanduser().resolve(), Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_id in parse_ids(args.case_ids):
        case_dir = root / f"case_{case_id:05d}"
        params = load_params(case_dir)
        mesh_points, faces, surface_values, log_density = latitude_longitude_mesh(params, args.theta_count, args.phi_count)
        surface_poly = make_polydata(mesh_points, faces, {"manufactured_surface": surface_values, "native_encoder_log_density": log_density})
        write_vtp(output_dir / f"toy_case_{case_id:05d}_surface.vtp", surface_poly)
        save_surface_mesh_png(
            case_id, mesh_points, faces, surface_values,
            output_dir / f"toy_case_{case_id:05d}_surface_mesh.png",
        )
        geometry = np.asarray(np.load(case_dir / "geometry_coords.npy", mmap_mode="r"), dtype=np.float32)
        volume = np.asarray(np.load(case_dir / "volume_coords.npy", mmap_mode="r"), dtype=np.float32)
        volume_values = np.asarray(np.load(case_dir / "volume_data.npy", mmap_mode="r"), dtype=np.float32)
        volume_poly = make_polydata(volume, None, {"manufactured_volume": volume_values[:, 0]})
        write_vtp(output_dir / f"toy_case_{case_id:05d}_volume_points.vtp", volume_poly)
        save_png(case_id, geometry, mesh_points, surface_values, volume, volume_values[:, 0], output_dir / f"toy_case_{case_id:05d}_overview.png")
        print(f"Exported case {case_id:05d}: {faces.shape[0]:,} triangles")


if __name__ == "__main__":
    main()
