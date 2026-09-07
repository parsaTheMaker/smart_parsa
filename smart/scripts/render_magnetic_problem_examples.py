#!/usr/bin/env python3
"""Render consistent comparison previews for the magnetic problem examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw


PROBLEMS = (
    ("flux_concentrator", "Passive flux concentrator", "B_magnitude_T", "Magnetic flux density |B| (T)"),
    ("c_core", "Current-driven C-core", "B_magnitude_T", "Magnetic flux density |B| (T)"),
    ("magnetic_shield", "Magnetic shielding shell", "B_magnitude_T", "Magnetic flux density |B| (T)"),
    ("eddy_current_plate", "Perforated eddy-current plate", "J_magnitude_A_m2", "Eddy-current density |J| (A/m2)"),
)


def robust_range(mesh: pv.DataSet, scalar: str) -> tuple[float, float]:
    values = np.asarray(mesh.point_data[scalar], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise RuntimeError(f"No finite values found for {scalar}.")
    upper = float(np.percentile(finite, 99.5))
    lower = float(max(0.0, np.percentile(finite, 0.5)))
    if upper <= lower:
        upper = float(finite.max())
    return lower, upper


def render_one(root: Path, name: str, title: str, scalar: str, scalar_title: str) -> Path:
    surface = pv.read(root / f"{name}_material_surface.vtp")
    volume = pv.read(root / f"{name}_full_volume.vtu")
    solution_slice = volume.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0))
    material_slice = solution_slice.threshold((0.5, 1.5), scalars="region_id")
    clim = robust_range(solution_slice, scalar)

    plotter = pv.Plotter(shape=(1, 2), window_size=(2400, 1050), off_screen=True, border=False)
    plotter.set_background("white")

    plotter.subplot(0, 0)
    plotter.add_mesh(
        surface,
        color="#b8c1c4",
        smooth_shading=True,
        ambient=0.28,
        diffuse=0.72,
        specular=0.28,
        specular_power=24,
    )
    plotter.add_silhouette(surface, color="#273238", line_width=2.0)
    if name == "c_core":
        # Show representative winding turns instead of the jagged boundary of
        # the cellwise analytic current-support region used by the FEM solve.
        for y_position in np.linspace(-0.044, 0.044, 7):
            winding = pv.ParametricTorus(ringradius=0.0395, crosssectionradius=0.0015)
            winding.rotate_x(90.0, inplace=True)
            winding.translate((-0.066, y_position, 0.0), inplace=True)
            plotter.add_mesh(winding, color="#d94832", smooth_shading=True, specular=0.25)
        plotter.add_text("fixed impressed-current loop", position="lower_left", color="#7f1d1d", font_size=13)
    plotter.add_text(title, position="upper_edge", color="#172126", font_size=20)
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.28)

    plotter.subplot(0, 1)
    plotter.add_mesh(
        solution_slice,
        scalars=scalar,
        cmap="turbo",
        clim=clim,
        interpolate_before_map=True,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": scalar_title,
            "vertical": True,
            "position_x": 0.84,
            "position_y": 0.12,
            "height": 0.72,
            "width": 0.055,
            "title_font_size": 15,
            "label_font_size": 13,
            "n_labels": 5,
            "fmt": "%.2g",
        },
    )
    plotter.add_mesh(material_slice.extract_feature_edges(), color="#111111", line_width=2.0)
    plotter.add_text("Central solution plane", position="upper_edge", color="#172126", font_size=20)
    plotter.add_text("linear scale; upper limit = 99.5th percentile", position="lower_left", color="#38474f", font_size=12)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.18)

    output = root / f"{name}_preview.png"
    plotter.show(screenshot=output, auto_close=True)
    return output


def compose(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    gutter = 36
    canvas = Image.new("RGB", (2 * width + gutter, 2 * height + gutter), "white")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(images):
        x = (index % 2) * (width + gutter)
        y = (index // 2) * (height + gutter)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline="#d8dfe2", width=2)
    canvas.save(output, compress_level=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/parsa/smart_parsa/results/magnetic_problem_examples"),
    )
    args = parser.parse_args()
    previews = [render_one(args.input_dir, *problem) for problem in PROBLEMS]
    overview = args.input_dir / "magnetic_problem_examples_overview.png"
    compose(previews, overview)
    print(f"Wrote {overview}")


if __name__ == "__main__":
    main()
