#!/usr/bin/env python3
"""Render transparent orthographic geometry-variety GIFs for Pump and Heat Exchanger."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


PUMP_ROOT = Path("/mnt/ssdraid/parsa/shift_pump_surface_vtp_remesh_v2_original")
HEAT_EXCHANGER_ROOT = Path("/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp")
DEFAULT_OUTPUT_DIR = Path("/home/parsa/smart_parsa/results/final/geometry_variety_gifs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pump-root", type=Path, default=PUMP_ROOT)
    parser.add_argument("--heat-exchanger-root", type=Path, default=HEAT_EXCHANGER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=50, help="Unique meshes to render per dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def discover_cases(root: Path, dataset: str) -> list[tuple[str, Path]]:
    if dataset == "pump":
        paths = sorted(root.glob("sample_*/merged_surfaces.vtp"))
    elif dataset == "heat_exchanger":
        paths = sorted(root.glob("case_*/heat_exchange_case_*_surface.vtp"))
    else:  # pragma: no cover - caller-controlled labels
        raise ValueError(f"Unknown dataset: {dataset}")
    if not paths:
        raise FileNotFoundError(f"No source VTPs found for {dataset} under {root}")
    return [(path.parent.name, path) for path in paths]


def choose_cases(cases: list[tuple[str, Path]], count: int, seed: int) -> list[tuple[str, Path]]:
    if count <= 0:
        raise ValueError("--count must be positive.")
    if len(cases) < count:
        raise ValueError(f"Requested {count} cases, but only {len(cases)} source VTPs are available.")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(cases), size=count, replace=False))
    return [cases[int(index)] for index in indices]


def _add_studio_lights(vtk, renderer, center: np.ndarray, scale: float) -> None:
    # Three key/fill/rim lights provide depth without a visible background plane.
    specs = (
        ((1.4, 1.1, 2.0), 0.95),
        ((-1.8, 0.6, 1.2), 0.50),
        ((0.2, -1.7, 1.6), 0.35),
    )
    renderer.AutomaticLightCreationOff()
    for direction, intensity in specs:
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(*(center + scale * np.asarray(direction, dtype=np.float64)))
        light.SetFocalPoint(*center)
        light.SetIntensity(float(intensity))
        light.SetColor(1.0, 1.0, 1.0)
        renderer.AddLight(light)


def _configure_camera(camera, bounds: tuple[float, float, float, float, float, float], view_axis: str) -> None:
    center = np.array(
        [0.5 * (bounds[0] + bounds[1]), 0.5 * (bounds[2] + bounds[3]), 0.5 * (bounds[4] + bounds[5])],
        dtype=np.float64,
    )
    spans = np.maximum(
        np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]], dtype=np.float64),
        1e-8,
    )
    distance = float(np.linalg.norm(spans) * 3.0)
    camera.ParallelProjectionOn()
    if view_axis == "z_plus_to_z_minus":
        camera.SetPosition(*(center + np.array([0.0, 0.0, distance])))
        camera.SetViewUp(0.0, 1.0, 0.0)
        visible_span = max(spans[0], spans[1])
    elif view_axis == "y_plus_to_y_minus":
        camera.SetPosition(*(center + np.array([0.0, distance, 0.0])))
        camera.SetViewUp(0.0, 0.0, 1.0)
        visible_span = max(spans[0], spans[2])
    else:  # pragma: no cover - worker payload is internally constructed
        raise ValueError(f"Unknown view axis: {view_axis}")
    camera.SetFocalPoint(*center)
    camera.SetParallelScale(float(visible_span * 0.60))


def render_frame(payload: tuple[str, str, str, int, int, tuple[float, float, float], str]) -> str:
    """Render one VTP in a fresh spawned process to avoid shared OpenGL state."""
    source_path_text, output_path_text, view_axis, width, height, color, label = payload
    os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    source_path = Path(source_path_text)
    output_path = Path(output_path_text)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(source_path))
    reader.Update()
    source = reader.GetOutput()
    if source is None or source.GetNumberOfPoints() == 0 or source.GetNumberOfPolys() == 0:
        raise RuntimeError(f"Invalid surface mesh for {label}: {source_path}")

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(source)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOn()
    normals.SetFeatureAngle(55.0)
    normals.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetInterpolationToPhong()
    prop.SetColor(*color)
    prop.SetAmbient(0.20)
    prop.SetDiffuse(0.72)
    prop.SetSpecular(0.80)
    prop.SetSpecularPower(42.0)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.SetBackgroundAlpha(0.0)
    renderer.AddActor(actor)
    if view_axis == "y_plus_to_y_minus":
        # Keep the front-facing perforations readable without adding visually
        # distracting feature lines to the Pump's naturally shaded surface.
        edges = vtk.vtkFeatureEdges()
        edges.SetInputConnection(normals.GetOutputPort())
        edges.BoundaryEdgesOn()
        edges.FeatureEdgesOn()
        edges.ManifoldEdgesOff()
        edges.NonManifoldEdgesOff()
        edges.SetFeatureAngle(55.0)
        edge_mapper = vtk.vtkPolyDataMapper()
        edge_mapper.SetInputConnection(edges.GetOutputPort())
        edge_actor = vtk.vtkActor()
        edge_actor.SetMapper(edge_mapper)
        edge_prop = edge_actor.GetProperty()
        edge_prop.SetColor(0.09, 0.12, 0.16)
        edge_prop.SetLineWidth(1.25)
        renderer.AddActor(edge_actor)
    bounds = source.GetBounds()
    center = np.array(
        [0.5 * (bounds[0] + bounds[1]), 0.5 * (bounds[2] + bounds[3]), 0.5 * (bounds[4] + bounds[5])],
        dtype=np.float64,
    )
    _add_studio_lights(vtk, renderer, center, max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]))

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetAlphaBitPlanes(1)
    window.SetMultiSamples(8)
    window.SetSize(int(width), int(height))
    window.AddRenderer(renderer)
    if hasattr(renderer, "UseFXAAOn"):
        renderer.UseFXAAOn()
    _configure_camera(renderer.GetActiveCamera(), bounds, view_axis)
    renderer.ResetCameraClippingRange()
    window.Render()

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGBA()
    capture.ReadFrontBufferOff()
    capture.Update()
    image = capture.GetOutput()
    array = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(height, width, 4)
    array = np.flipud(array)
    if int(array[..., 3].max()) == 0:
        raise RuntimeError(f"Transparent render contains no visible geometry for {label}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGBA").save(output_path)
    window.Finalize()
    return str(output_path)


def rgba_to_gif_frame(image: Image.Image) -> Image.Image:
    """Reserve palette index 255 as GIF's transparent background color."""
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    rgb = rgba.convert("RGB")
    quantized = rgb.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    indices = np.asarray(quantized).copy()
    indices[alpha < 128] = 255
    frame = Image.fromarray(indices.astype(np.uint8), mode="P")
    palette = quantized.getpalette()[: 255 * 3] + [0, 0, 0]
    frame.putpalette(palette)
    frame.info["transparency"] = 255
    return frame


def assemble_gif(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    duration_ms = int(round(1000.0 / float(fps)))
    frames = [rgba_to_gif_frame(Image.open(path)) for path in frame_paths]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        transparency=255,
        optimize=False,
    )


def render_dataset(
    dataset: str,
    cases: list[tuple[str, Path]],
    output_dir: Path,
    view_axis: str,
    color: tuple[float, float, float],
    args: argparse.Namespace,
) -> Path:
    frame_dir = output_dir / f".{dataset}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    payloads = [
        (
            str(path),
            str(frame_dir / f"frame_{index:03d}_{case_id}.png"),
            view_axis,
            int(args.width),
            int(args.height),
            color,
            case_id,
        )
        for index, (case_id, path) in enumerate(cases)
    ]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(int(args.workers), len(payloads)), mp_context=context) as pool:
        futures = [pool.submit(render_frame, payload) for payload in payloads]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Rendering {dataset}", dynamic_ncols=True):
            future.result()

    frame_paths = [Path(payload[1]) for payload in payloads]
    gif_path = output_dir / f"{dataset}_geometry_variety_{view_axis}_50cases_5fps_transparent.gif"
    assemble_gif(frame_paths, gif_path, args.fps)
    if not args.keep_frames:
        shutil.rmtree(frame_dir)
    return gif_path


def main() -> None:
    args = parse_args()
    if args.width < 64 or args.height < 64 or args.fps <= 0.0:
        raise ValueError("Image dimensions must be >= 64 and --fps must be positive.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pump_cases = choose_cases(discover_cases(args.pump_root.expanduser(), "pump"), args.count, args.seed)
    heat_cases = choose_cases(discover_cases(args.heat_exchanger_root.expanduser(), "heat_exchanger"), args.count, args.seed + 1)

    pump_gif = render_dataset(
        "pump", pump_cases, output_dir, "z_plus_to_z_minus", (0.42, 0.62, 0.82), args
    )
    heat_gif = render_dataset(
        "heat_exchanger", heat_cases, output_dir, "y_plus_to_y_minus", (0.89, 0.47, 0.18), args
    )
    manifest = {
        "frames_per_second": float(args.fps),
        "frame_size": [int(args.width), int(args.height)],
        "transparent_background": True,
        "pump": {
            "camera": "+z to -z orthographic",
            "gif": str(pump_gif),
            "cases": [case_id for case_id, _ in pump_cases],
        },
        "heat_exchanger": {
            "camera": "+y to -y orthographic",
            "gif": str(heat_gif),
            "cases": [case_id for case_id, _ in heat_cases],
        },
    }
    (output_dir / "geometry_variety_gifs_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {pump_gif}")
    print(f"Wrote {heat_gif}")


if __name__ == "__main__":
    main()
