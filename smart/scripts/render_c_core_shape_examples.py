#!/usr/bin/env python3
"""Render the eight fixed-coil C-core examples for visual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image, ImageDraw


def add_fixed_coil(plotter: pv.Plotter) -> None:
    for y_position in np.linspace(-0.035, 0.035, 10):
        winding = pv.ParametricTorus(ringradius=0.0265, crosssectionradius=0.0015)
        winding.rotate_x(90.0, inplace=True)
        winding.translate((-0.0665, y_position, 0.0), inplace=True)
        plotter.add_mesh(
            winding,
            color="#d9472f",
            smooth_shading=True,
            ambient=0.18,
            diffuse=0.74,
            specular=0.38,
            specular_power=30,
        )


def display_name(record: dict[str, object]) -> str:
    profile = str(record["profile"])
    if str(record["name"]).startswith("ellipse_wide"):
        profile = "wide ellipse"
    elif str(record["name"]).startswith("ellipse_tall"):
        profile = "tall ellipse"
    topology = "unified window" if record["window_topology"] == "unified" else "two windows"
    style = "parallel poles" if record["pole_style"] == "parallel" else "focused poles"
    section = f"{record['section_profile']} section"
    return f"{profile.title()} - {topology}, {style}, {section}"


def compact_display_name(record: dict[str, object]) -> str:
    profile = str(record["profile"])
    if str(record["name"]).startswith("ellipse_wide"):
        profile = "wide ellipse"
    elif str(record["name"]).startswith("ellipse_tall"):
        profile = "tall ellipse"
    topology = "one window" if record["window_topology"] == "unified" else "two windows"
    style = "parallel poles" if record["pole_style"] == "parallel" else "focused poles"
    return f"{profile.title()} | {record['section_profile']} section\n{topology} | {style}"


def data_path(root: Path, record: dict[str, object], key: str) -> Path:
    path = Path(str(record[key]))
    return path if path.is_absolute() else root / path


def render_one(root: Path, record: dict[str, object], clim: tuple[float, float]) -> Path:
    surface = pv.read(data_path(root, record, "surface_vtp"))
    solid = pv.read(data_path(root, record, "solid_volume_vtu"))
    center_slice = solid.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0))
    center_slice.point_data["B_magnitude_T"] = np.linalg.norm(
        np.asarray(center_slice.point_data["B_T"], dtype=np.float64), axis=1
    )
    plotter = pv.Plotter(shape=(1, 2), window_size=(2200, 900), off_screen=True, border=False)
    plotter.set_background("white")

    plotter.subplot(0, 0)
    plotter.add_mesh(
        surface,
        color="#bdc5c7",
        smooth_shading=True,
        ambient=0.30,
        diffuse=0.70,
        specular=0.25,
        specular_power=25,
    )
    plotter.add_silhouette(surface, color="#263238", line_width=2.0)
    add_fixed_coil(plotter)
    plotter.add_text(compact_display_name(record), position="upper_left", color="#172126", font_size=14)
    plotter.add_text("identical straight limb and coil", position="lower_left", color="#59666c", font_size=12)
    plotter.camera_position = [(0.29, -0.34, 0.26), (0.005, 0.0, 0.0), (0.0, 0.0, 1.0)]
    plotter.camera.zoom(1.35)

    plotter.subplot(0, 1)
    plotter.add_mesh(
        center_slice,
        scalars="B_magnitude_T",
        cmap="turbo",
        clim=clim,
        interpolate_before_map=True,
        scalar_bar_args={
            "title": "Magnetic flux density |B| (T)",
            "vertical": True,
            "position_x": 0.84,
            "position_y": 0.13,
            "height": 0.70,
            "width": 0.055,
            "title_font_size": 14,
            "label_font_size": 12,
            "n_labels": 5,
            "fmt": "%.2g",
        },
    )
    plotter.add_mesh(center_slice.extract_feature_edges(), color="#182126", line_width=1.7)
    plotter.add_text("Solid mid-plane solution", position="upper_edge", color="#172126", font_size=19)
    plotter.add_text("common linear scale", position="lower_left", color="#59666c", font_size=11)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.30)

    output = root / f"{record['name']}_preview.png"
    plotter.show(screenshot=output, auto_close=True)
    return output


def compose(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    gutter = 28
    canvas = Image.new("RGB", (2 * width + gutter, 4 * height + 3 * gutter), "white")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(images):
        x = (index % 2) * (width + gutter)
        y = (index // 2) * (height + gutter)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline="#d8dfe2", width=2)
    canvas.save(output, compress_level=2)


def midplane_triangles(solid: pv.UnstructuredGrid) -> tuple[np.ndarray, np.ndarray]:
    section = solid.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0)).triangulate()
    triangles = np.asarray(section.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    xyz = np.asarray(section.points, dtype=np.float64)[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    equivalent_edge_mm = np.sqrt(4.0 * area / np.sqrt(3.0)) * 1.0e3
    return xyz[:, :, :2], equivalent_edge_mm


def render_mesh_adaptivity(root: Path, records: list[dict[str, object]]) -> Path:
    panels = []
    all_sizes = []
    for record in records:
        solid = pv.read(data_path(root, record, "solid_volume_vtu"))
        triangles, sizes = midplane_triangles(solid)
        panels.append((record, triangles, sizes))
        all_sizes.append(sizes)

    combined = np.concatenate(all_sizes)
    limits = np.percentile(combined, [0.5, 99.5])
    norm = Normalize(vmin=float(limits[0]), vmax=float(limits[1]), clip=True)
    figure, axes = plt.subplots(4, 2, figsize=(13.2, 17.2), constrained_layout=True)
    collection = None
    for axis, (record, triangles, sizes) in zip(axes.ravel(), panels):
        collection = PolyCollection(
            triangles,
            array=sizes,
            cmap="viridis",
            norm=norm,
            edgecolors=(0.08, 0.10, 0.12, 0.30),
            linewidths=0.10,
            rasterized=True,
        )
        axis.add_collection(collection)
        points = triangles.reshape(-1, 2)
        padding = 0.015
        axis.set_xlim(float(points[:, 0].min() - padding), float(points[:, 0].max() + padding))
        axis.set_ylim(float(points[:, 1].min() - padding), float(points[:, 1].max() + padding))
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.set_title(display_name(record), fontsize=13, color="#172126", pad=4)
    if collection is None:
        raise RuntimeError("No surfaces were available for the mesh adaptivity overview.")
    colorbar = figure.colorbar(collection, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.62, pad=0.015)
    colorbar.set_label("Equivalent mid-plane triangle edge length (mm)", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    output = root / "c_core_mesh_adaptivity_overview.png"
    figure.savefig(output, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output


def render_bz_overview(root: Path, records: list[dict[str, object]]) -> Path:
    """Show the signed out-of-plane field away from the z-symmetry plane."""
    panels = []
    all_values = []
    for record in records:
        solid = pv.read(data_path(root, record, "solid_volume_vtu"))
        section = solid.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.005)).triangulate()
        triangles = np.asarray(section.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
        xy = np.asarray(section.points, dtype=np.float64)[:, :2]
        point_bz = np.asarray(section.point_data["B_T"], dtype=np.float64)[:, 2]
        cell_bz = point_bz[triangles].mean(axis=1)
        panels.append((record, xy[triangles], cell_bz))
        all_values.append(cell_bz)

    limit = float(np.percentile(np.abs(np.concatenate(all_values)), 99.5))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    figure, axes = plt.subplots(4, 2, figsize=(13.2, 17.0), constrained_layout=True)
    collection = None
    for axis, (record, triangles, values) in zip(axes.ravel(), panels):
        collection = PolyCollection(
            triangles,
            array=values,
            cmap="RdBu_r",
            norm=norm,
            edgecolors="none",
            rasterized=True,
        )
        axis.add_collection(collection)
        points = triangles.reshape(-1, 2)
        padding = 0.008
        axis.set_xlim(float(points[:, 0].min() - padding), float(points[:, 0].max() + padding))
        axis.set_ylim(float(points[:, 1].min() - padding), float(points[:, 1].max() + padding))
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.set_title(display_name(record), fontsize=13, color="#172126", pad=4)
    if collection is None:
        raise RuntimeError("No solved sections were available for the Bz overview.")
    colorbar = figure.colorbar(
        collection,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        shrink=0.62,
        pad=0.015,
    )
    colorbar.set_label(r"Out-of-plane magnetic flux density $B_z$ at $z=5$ mm (T)", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    output = root / "c_core_bz_offset_slice_overview.png"
    figure.savefig(output, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/parsa/smart_parsa/results/c_core_shape_solution_examples"),
    )
    args = parser.parse_args()
    summary = json.loads((args.input_dir / "c_core_shape_examples_summary.json").read_text())
    all_values = []
    for record in summary["problems"]:
        solid = pv.read(data_path(args.input_dir, record, "solid_volume_vtu"))
        values = np.linalg.norm(np.asarray(solid.point_data["B_T"], dtype=np.float64), axis=1)
        all_values.append(values[np.isfinite(values)])
    clim = (0.0, float(np.percentile(np.concatenate(all_values), 99.9)))
    previews = [render_one(args.input_dir, record, clim) for record in summary["problems"]]
    overview = args.input_dir / "c_core_shape_examples_overview.png"
    compose(previews, overview)
    mesh_overview = render_mesh_adaptivity(args.input_dir, summary["problems"])
    bz_overview = render_bz_overview(args.input_dir, summary["problems"])
    print(f"Wrote {overview}")
    print(f"Wrote {mesh_overview}")
    print(f"Wrote {bz_overview}")


if __name__ == "__main__":
    main()
