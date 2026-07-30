#!/usr/bin/env python3
"""Download DrivAerML surface VTP files and extract one geometry-only VTP per run.

The source VTP files are surface PolyData files. This script preserves their
coordinates and surface topology, triangulates polygons when needed, writes a
compressed binary VTP, validates the result, optionally renders a preview, and
removes the temporary source VTP after successful processing.

The default output is deliberately outside the training cache:
    /mnt/ssdraid/parsa/drivaerml_surface_vtp/run_<id>/drivaer_<id>.vtp
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm


REPO_ID = "neashton/drivaerml"
RUN_RE = re.compile(r"^run_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/mnt/ssdraid/parsa/drivaerml_surface_vtp")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional local source root containing run_<id>/boundary_<id>.vtp; skips Hugging Face downloads.",
    )
    parser.add_argument("--run-ids", default=None, help="Optional comma-separated run ids, e.g. 1,11,34.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many discovered runs.")
    parser.add_argument("--start-at", type=int, default=0, help="Zero-based ordinal in sorted run order.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of geometries processed in parallel (default: 4).",
    )
    parser.add_argument(
        "--compression",
        choices=("lz4", "zlib", "none"),
        default="none",
        help="VTP payload compression; uncompressed XML-safe binary is the default for ParaView readability.",
    )
    parser.add_argument(
        "--merge-duplicate-vertices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge coincident vertices with VTK's threaded static cleaner (default: enabled).",
    )
    parser.add_argument(
        "--validate-output",
        action="store_true",
        help="Fully reread each written VTP with VTK for validation; slower and disabled by default.",
    )
    parser.add_argument(
        "--overwrite-geometry",
        action="store_true",
        help="Reconvert geometries even when an output VTP already exists.",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=10,
        help="Render the first geometry and then every Nth geometry in sorted run order. Use 0 to disable previews.",
    )
    parser.add_argument(
        "--preview-backend",
        choices=("auto", "open3d", "pyvista", "matplotlib"),
        default="auto",
        help="Preview renderer. auto tries Open3D, then PyVista, then Matplotlib.",
    )
    parser.add_argument(
        "--preview-max-triangles",
        type=int,
        default=100_000,
        help="Maximum connected triangles used for PNG previews; the saved VTP is never decimated.",
    )
    parser.add_argument(
        "--overwrite-previews",
        action="store_true",
        help="Regenerate scheduled PNG previews even when they already exist; existing VTPs are reused.",
    )
    parser.add_argument(
        "--keep-vtp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep temporary VTP files after successful conversion (default: delete them).",
    )
    parser.add_argument(
        "--keep-failed-vtp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep a VTP when conversion fails for debugging (default: delete it).",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed run.")
    return parser.parse_args()


def require_dependencies():
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("VTK Python bindings are required. Install vtk in the active environment.") from exc
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("huggingface_hub is required for remote VTP downloads.") from exc
    return vtk, vtk_to_numpy, HfApi, hf_hub_download


def discover_remote_runs(api, repo_id: str, revision: str) -> list[int]:
    run_ids = []
    for entry in api.list_repo_tree(repo_id, repo_type="dataset", revision=revision, recursive=False, expand=False):
        match = RUN_RE.fullmatch(str(entry.path))
        if match:
            run_ids.append(int(match.group(1)))
    if not run_ids:
        raise RuntimeError(f"No run_<id> directories found in {repo_id!r} at revision {revision!r}.")
    return sorted(set(run_ids))


def discover_local_runs(source_dir: Path) -> list[int]:
    run_ids = []
    for path in source_dir.glob("run_*"):
        match = RUN_RE.fullmatch(path.name)
        if match and (path / f"boundary_{match.group(1)}.vtp").is_file():
            run_ids.append(int(match.group(1)))
    if not run_ids:
        raise RuntimeError(f"No local boundary VTP files found under {source_dir}.")
    return sorted(set(run_ids))


def parse_run_ids(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            value = int(item)
            if value <= 0:
                raise ValueError(f"Run ids must be positive, got {value}.")
            values.append(value)
    if not values:
        raise ValueError("--run-ids did not contain any run ids.")
    return sorted(set(values))


def read_polydata(vtk, path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetPoints() is None:
        raise RuntimeError(f"VTP has no point coordinates: {path}")
    if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfCells() == 0:
        raise RuntimeError(
            f"VTP is empty: {path} points={polydata.GetNumberOfPoints()} cells={polydata.GetNumberOfCells()}"
        )
    return polydata


def triangulate_geometry(vtk, polydata, merge_duplicate_vertices: bool):
    """Create a triangulated, geometry-only PolyData object."""
    # Training geometry should not carry CFD fields or field-data metadata.
    polydata.GetPointData().Initialize()
    polydata.GetCellData().Initialize()
    polydata.GetFieldData().Initialize()

    polygons = polydata.GetPolys()
    is_triangle_only = (
        polygons is not None
        and polygons.GetNumberOfCells() == polydata.GetNumberOfCells()
        and polygons.GetNumberOfCells() > 0
        and polygons.IsHomogeneous() == 3
    )
    if is_triangle_only:
        # Avoid a full cell-by-cell triangle filter for the common VTP layout.
        triangulated = polydata
    else:
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputData(polydata)
        if hasattr(triangle_filter, "PreservePolysOff"):
            triangle_filter.PreservePolysOff()
        triangle_filter.Update()
        triangulated = triangle_filter.GetOutput()
    if triangulated is None or triangulated.GetNumberOfPolys() == 0:
        raise RuntimeError("Surface contains no polygonal triangles.")

    if not merge_duplicate_vertices:
        return triangulated

    clean_filter_class = getattr(vtk, "vtkStaticCleanPolyData", vtk.vtkCleanPolyData)
    clean_filter = clean_filter_class()
    clean_filter.SetInputData(triangulated)
    clean_filter.SetTolerance(0.0)
    if hasattr(clean_filter, "PointMergingOn"):
        clean_filter.PointMergingOn()
    if hasattr(clean_filter, "RemoveUnusedPointsOn"):
        clean_filter.RemoveUnusedPointsOn()
    clean_filter.Update()
    cleaned = clean_filter.GetOutput()
    if cleaned is None or cleaned.GetNumberOfPoints() == 0 or cleaned.GetNumberOfPolys() == 0:
        raise RuntimeError("Cleaning produced an empty triangulated surface.")
    return cleaned


def write_geometry_vtp(
    vtk,
    polydata,
    output_path: Path,
    compression: str,
    merge_duplicate_vertices: bool,
    validate_output: bool,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if partial_path.exists():
        partial_path.unlink()

    triangulated = triangulate_geometry(vtk, polydata, merge_duplicate_vertices)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(partial_path))
    writer.SetInputData(triangulated)
    # Inline base64 binary arrays are XML-safe across older ParaView builds.
    # This avoids the appended-data parser path that rejected the previous files.
    writer.SetDataModeToBinary()
    # Each DrivAerML array is below 4 GiB; UInt32 headers maximize compatibility
    # with older ParaView releases while remaining sufficient for this mesh.
    if hasattr(writer, "SetHeaderTypeToUInt32"):
        writer.SetHeaderTypeToUInt32()
    if compression == "lz4" and hasattr(vtk, "vtkLZ4DataCompressor"):
        writer.SetCompressor(vtk.vtkLZ4DataCompressor())
    elif compression == "zlib" and hasattr(vtk, "vtkZLibDataCompressor"):
        writer.SetCompressor(vtk.vtkZLibDataCompressor())
    else:
        # VTK's XML writer may default to Zlib; explicitly clear it for uncompressed VTP.
        writer.SetCompressor(None)
    if writer.Write() != 1:
        raise RuntimeError(f"VTK failed to write VTP: {output_path}")
    partial_path.replace(output_path)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"VTK wrote an empty VTP: {output_path}")
    point_count = triangulated.GetNumberOfPoints()
    triangle_count = triangulated.GetNumberOfPolys()
    if validate_output:
        validated = read_vtp(vtk, output_path)
        point_count = validated.GetNumberOfPoints()
        triangle_count = validated.GetNumberOfPolys()
    return point_count, triangle_count


def read_vtp(vtk, path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfPolys() == 0:
        raise RuntimeError(f"Written VTP failed validation: {path}")
    return polydata


def polydata_mesh_arrays(vtk_to_numpy, polydata) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=np.float32)
    raw_cells = np.asarray(vtk_to_numpy(polydata.GetPolys().GetData()))
    if raw_cells.size == 0 or raw_cells.size % 4 != 0:
        raise RuntimeError("VTP does not contain a triangle-only polygon array.")
    cells = raw_cells.reshape(-1, 4)
    if not np.all(cells[:, 0] == 3):
        raise RuntimeError("VTP contains non-triangle polygons.")
    return points, cells[:, 1:].astype(np.int32, copy=False)


def read_vtp_preview_mesh(vtk, vtk_to_numpy, vtp_path: Path, max_triangles: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a connected, bounded triangle mesh for a readable preview."""
    max_triangles = max(1, int(max_triangles))
    polydata = read_vtp(vtk, vtp_path)
    triangle_count = polydata.GetNumberOfPolys()
    preview_polydata = polydata
    if triangle_count > max_triangles and hasattr(vtk, "vtkQuadricClustering"):
        # Spatial clustering is much faster than global quadric decimation for
        # these 10M+ triangle surfaces while still producing a connected mesh.
        divisions = max(8, min(256, int(np.ceil((2.0 * max_triangles) ** (1.0 / 3.0)))))
        cluster = vtk.vtkQuadricClustering()
        cluster.SetInputData(polydata)
        cluster.SetNumberOfDivisions(divisions, divisions, divisions)
        cluster.UseFeatureEdgesOff()
        cluster.UseFeaturePointsOff()
        cluster.Update()
        candidate = cluster.GetOutput()
        if candidate is not None and candidate.GetNumberOfPolys() > 0:
            preview_polydata = candidate
    try:
        return polydata_mesh_arrays(vtk_to_numpy, preview_polydata)
    except RuntimeError:
        # Keep preview generation resilient if a decimator returns an unusual cell layout.
        points, faces = polydata_mesh_arrays(vtk_to_numpy, polydata)
        sample_count = min(faces.shape[0], max_triangles)
        selected = np.linspace(0, faces.shape[0] - 1, sample_count, dtype=np.int64)
        sampled_faces = faces[selected]
        vertices = points[sampled_faces].reshape(-1, 3).copy()
        sampled_faces = np.arange(vertices.shape[0], dtype=np.int32).reshape(-1, 3)
        return vertices, sampled_faces


def render_preview(
    vtp_path: Path,
    preview_path: Path,
    vtk,
    vtk_to_numpy,
    backend: str = "auto",
    max_triangles: int = 100_000,
) -> str:
    """Render with Open3D first, then PyVista, then a sampled Matplotlib view."""
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    errors = []
    if backend in {"auto", "open3d"}:
        try:
            import open3d as o3d

            vertices, faces = read_vtp_preview_mesh(vtk, vtk_to_numpy, vtp_path, max_triangles=max_triangles)
            surface_mesh = o3d.geometry.TriangleMesh()
            surface_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            surface_mesh.triangles = o3d.utility.Vector3iVector(faces)
            surface_mesh.compute_vertex_normals()
            renderer = o3d.visualization.rendering.OffscreenRenderer(1600, 1000)
            renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.base_color = [0.62, 0.72, 0.79, 1.0]
            renderer.scene.add_geometry("surface", surface_mesh, material)
            bounds = surface_mesh.get_axis_aligned_bounding_box()
            center = np.asarray(bounds.get_center(), dtype=np.float64)
            extent = np.asarray(bounds.get_extent(), dtype=np.float64)
            radius = max(float(extent.max()), 1.0)
            camera_position = center + np.asarray([1.5, -1.5, 1.0]) * radius
            renderer.setup_camera(50.0, center, camera_position, [0.0, 0.0, 1.0])
            image = renderer.render_to_image()
            if not o3d.io.write_image(str(preview_path), image):
                raise RuntimeError("Open3D failed to write the screenshot.")
            del renderer
            if preview_path.is_file() and preview_path.stat().st_size > 0:
                return f"open3d_sampled_{len(vertices) // 3}_triangles"
            raise RuntimeError("Open3D did not create a screenshot.")
        except Exception as exc:
            errors.append(f"open3d: {type(exc).__name__}: {exc}")
            if backend == "open3d":
                raise RuntimeError(errors[-1]) from exc

    if backend in {"auto", "pyvista"}:
        try:
            import pyvista as pv

            mesh = pv.read(str(vtp_path))
            plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))
            plotter.set_background("white")
            plotter.add_mesh(mesh, color="#9fb7c9", smooth_shading=False, show_edges=False)
            plotter.add_axes(line_width=2, labels_off=False)
            plotter.camera_position = "iso"
            plotter.camera.zoom(1.15)
            plotter.show(screenshot=str(preview_path), auto_close=True)
            if preview_path.is_file() and preview_path.stat().st_size > 0:
                return "pyvista"
            raise RuntimeError("PyVista did not create a screenshot.")
        except Exception as exc:
            errors.append(f"pyvista: {type(exc).__name__}: {exc}")
            if backend == "pyvista":
                raise RuntimeError(errors[-1]) from exc

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import vtk

    polydata = read_vtp(vtk, vtp_path)
    points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=np.float32)
    max_points = 150_000
    if points.shape[0] > max_points:
        selector = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[selector]
    figure = plt.figure(figsize=(16, 10), dpi=150)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.08, c="#6f8fa5", linewidths=0)
    axis.set_title(f"{vtp_path.stem} surface geometry", fontsize=20)
    axis.set_xlabel("x", fontsize=14)
    axis.set_ylabel("y", fontsize=14)
    axis.set_zlabel("z", fontsize=14)
    axis.set_box_aspect(np.ptp(points, axis=0).clip(min=1.0))
    figure.tight_layout()
    figure.savefig(preview_path, bbox_inches="tight")
    plt.close(figure)
    return f"matplotlib_fallback ({'; '.join(errors)})"


def download_vtp(
    hf_hub_download,
    repo_id: str,
    revision: str,
    run_id: int,
    temp_root: Path,
    hf_token: str | None,
) -> tuple[Path, Path]:
    run_temp_dir = temp_root / f"run_{run_id}"
    run_temp_dir.mkdir(parents=True, exist_ok=True)
    filename = f"run_{run_id}/boundary_{run_id}.vtp"
    downloaded_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=str(run_temp_dir),
            token=hf_token,
        )
    )
    return downloaded_path, run_temp_dir


def append_manifest(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def process_run(
    *,
    vtk,
    vtk_to_numpy,
    hf_hub_download,
    run_id: int,
    ordinal: int,
    output_dir: Path,
    repo_id: str,
    revision: str,
    hf_token: str | None,
    source_dir: Path | None,
    temp_root: Path,
    preview_every: int,
    preview_backend: str,
    preview_max_triangles: int,
    compression: str,
    merge_duplicate_vertices: bool,
    validate_output: bool,
    overwrite_geometry: bool,
    overwrite_previews: bool,
    keep_vtp: bool,
    keep_failed_vtp: bool,
) -> dict:
    run_output_dir = output_dir / f"run_{run_id}"
    geometry_path = run_output_dir / f"drivaer_{run_id}.vtp"
    preview_path = run_output_dir / f"drivaer_{run_id}_preview.png"
    started = time.time()
    source_vtp_path: Path | None = None
    run_temp_dir: Path | None = None
    source_kind = "local"
    try:
        if not overwrite_geometry and geometry_path.is_file() and geometry_path.stat().st_size > 0:
            validated = read_vtp(vtk, geometry_path)
            point_count = validated.GetNumberOfPoints()
            triangle_count = validated.GetNumberOfPolys()
        else:
            if source_dir is not None:
                source_vtp_path = source_dir / f"run_{run_id}" / f"boundary_{run_id}.vtp"
                if not source_vtp_path.is_file():
                    raise FileNotFoundError(f"Local VTP not found: {source_vtp_path}")
            else:
                source_kind = "huggingface"
                source_vtp_path, run_temp_dir = download_vtp(
                    hf_hub_download,
                    repo_id,
                    revision,
                    run_id,
                    temp_root,
                    hf_token,
                )
            polydata = read_polydata(vtk, source_vtp_path)
            point_count, triangle_count = write_geometry_vtp(
                vtk,
                polydata,
                geometry_path,
                compression=compression,
                merge_duplicate_vertices=merge_duplicate_vertices,
                validate_output=validate_output,
            )

        preview_backend_used = None
        if preview_every > 0 and ordinal % preview_every == 0 and (overwrite_previews or not preview_path.is_file()):
            preview_backend_used = render_preview(
                geometry_path,
                preview_path,
                vtk,
                vtk_to_numpy,
                backend=preview_backend,
                max_triangles=preview_max_triangles,
            )

        if source_vtp_path is not None and not keep_vtp and source_vtp_path.is_file():
            source_vtp_path.unlink()
        return {
            "run_id": run_id,
            "ordinal": ordinal,
            "status": "ok",
            "source": source_kind,
            "vtp": str(geometry_path),
            "preview": str(preview_path) if preview_path.is_file() else None,
            "preview_backend": preview_backend_used,
            "points": int(point_count),
            "triangles": int(triangle_count),
            "seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            "run_id": run_id,
            "ordinal": ordinal,
            "status": "error",
            "source": source_kind,
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.time() - started, 3),
        }
    finally:
        if source_vtp_path is not None and source_vtp_path.is_file() and (keep_vtp or keep_failed_vtp):
            pass
        elif source_vtp_path is not None and source_vtp_path.is_file():
            source_vtp_path.unlink()
        if run_temp_dir is not None and run_temp_dir.exists() and not keep_vtp and not keep_failed_vtp:
            shutil.rmtree(run_temp_dir, ignore_errors=True)


def process_run_worker(task: dict) -> dict:
    """Load VTK inside each spawned worker and process one geometry."""
    vtk, vtk_to_numpy, _hf_api, hf_hub_download = require_dependencies()
    return process_run(
        vtk=vtk,
        vtk_to_numpy=vtk_to_numpy,
        hf_hub_download=hf_hub_download,
        **task,
    )


def main() -> int:
    args = parse_args()
    if args.preview_every < 0:
        raise ValueError("--preview-every must be non-negative.")
    if args.start_at < 0:
        raise ValueError("--start-at must be non-negative.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    vtk, vtk_to_numpy, HfApi, hf_hub_download = require_dependencies()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "extraction_manifest.jsonl"
    temp_root = output_dir / ".surface_vtp_downloads"
    temp_root.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_dir).expanduser().resolve() if args.source_dir else None
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if args.run_ids:
        run_ids = parse_run_ids(args.run_ids)
    elif source_dir is not None:
        run_ids = discover_local_runs(source_dir)
    else:
        run_ids = discover_remote_runs(HfApi(token=hf_token), args.repo_id, args.revision)
    run_ids = run_ids[args.start_at :]
    if args.limit is not None:
        run_ids = run_ids[: max(0, int(args.limit))]
    if not run_ids:
        print("No runs selected.")
        return 0

    print(f"Selected {len(run_ids)} runs; output={output_dir}", flush=True)
    print(
        f"Source={'local ' + str(source_dir) if source_dir is not None else f'Hugging Face {args.repo_id}@{args.revision}'}; "
        f"workers={args.workers}; preview_every={args.preview_every}; "
        f"compression={args.compression}; merge_duplicates={args.merge_duplicate_vertices}; "
        f"full_validation={args.validate_output}; overwrite_geometry={args.overwrite_geometry}; "
        f"temporary source VTPs deleted={not args.keep_vtp}",
        flush=True,
    )

    success_count = 0
    error_count = 0
    tasks = [
        {
            "run_id": run_id,
            "ordinal": ordinal,
            "output_dir": output_dir,
            "repo_id": args.repo_id,
            "revision": args.revision,
            "hf_token": hf_token,
            "source_dir": source_dir,
            "temp_root": temp_root,
            "preview_every": args.preview_every,
            "preview_backend": args.preview_backend,
            "preview_max_triangles": args.preview_max_triangles,
            "compression": args.compression,
            "merge_duplicate_vertices": args.merge_duplicate_vertices,
            "validate_output": args.validate_output,
            "overwrite_geometry": args.overwrite_geometry,
            "overwrite_previews": args.overwrite_previews,
            "keep_vtp": args.keep_vtp,
            "keep_failed_vtp": args.keep_failed_vtp,
        }
        for ordinal, run_id in enumerate(run_ids)
    ]

    context = mp.get_context("spawn")
    pool = context.Pool(processes=args.workers)
    progress = tqdm(total=len(tasks), desc="Extracting geometries", unit="run", dynamic_ncols=True)
    try:
        for record in pool.imap_unordered(process_run_worker, tasks, chunksize=1):
            progress.update(1)
            append_manifest(manifest_path, record)
            if record["status"] == "ok":
                success_count += 1
                progress.set_postfix_str(f"run_{record['run_id']} ok", refresh=False)
                preview_text = f" preview={record['preview']}" if record.get("preview") else ""
                print(
                    f"run_{record['run_id']}: ok points={record['points']} triangles={record['triangles']} "
                    f"seconds={record['seconds']}{preview_text}",
                    flush=True,
                )
            else:
                error_count += 1
                progress.set_postfix_str(f"run_{record['run_id']} ERROR", refresh=False)
                print(f"run_{record['run_id']}: ERROR {record['error']}", flush=True)
                if args.fail_fast:
                    raise RuntimeError(record["error"])
    except KeyboardInterrupt:
        print("Interrupted: terminating all extraction workers...", flush=True)
        pool.terminate()
        pool.join()
        progress.close()
        print("All extraction workers terminated; completed outputs remain resumable.", flush=True)
        return 130
    except BaseException:
        pool.terminate()
        pool.join()
        progress.close()
        raise
    else:
        pool.close()
        pool.join()
        progress.close()

    if temp_root.exists() and not any(temp_root.iterdir()):
        temp_root.rmdir()
    print(
        f"Finished: success={success_count} errors={error_count} manifest={manifest_path}",
        flush=True,
    )
    return 1 if error_count else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed runs remain resumable.", file=sys.stderr)
        raise SystemExit(130)
