#!/usr/bin/env python3
"""Generate a deterministic mesh-FEM benchmark for sampling invariance.

Each case is a tapered, wavy cooling fin with three rounded through-holes.  Gmsh
creates an adaptive tetrahedral mesh, scikit-fem solves nonlinear steady heat
conduction, and all targets are interpolated from that numerical solution.  Native encoder
points are sampled uniformly over mesh triangles (therefore denser where the
adaptive mesh is finer); reference queries are area/volume uniform instead.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import cg, spsolve
from tqdm.auto import tqdm

from utils.geometry_density import estimate_log_sampling_density


def save_array(path: Path, array: np.ndarray, dtype=np.float32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=dtype), allow_pickle=False)
    partial.replace(path)


def fin_parameters(seed: int) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(seed)
    height = float(rng.uniform(1.25, 1.65))
    width = float(rng.uniform(0.78, 1.08))
    thickness = float(rng.uniform(0.18, 0.27))
    taper = float(rng.uniform(0.22, 0.42))
    waviness = float(rng.uniform(0.035, 0.095))
    # Fixed vertical bands prevent overlap while preserving a meaningful range
    # of thermal paths through the perforated solid.
    hole_z = rng.uniform(
        [0.23 * height, 0.46 * height, 0.67 * height],
        [0.34 * height, 0.58 * height, 0.79 * height],
    )
    hole_r = rng.uniform(0.060, 0.095, size=3)
    hole_x = rng.uniform(-0.16 * width, 0.16 * width, size=3)
    return {
        "height": height, "width": width, "thickness": thickness,
        "taper": taper, "waviness": waviness,
        "hole_z": hole_z.tolist(), "hole_r": hole_r.tolist(), "hole_x": hole_x.tolist(),
    }


def make_tetra_mesh(params: dict, h_min: float, h_max: float, gmsh_threads: int):
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        # Parallelize across independent geometries instead of nesting Gmsh
        # threads inside every process.  This gives predictable high host use.
        gmsh.option.setNumber("General.NumThreads", max(1, int(gmsh_threads)))
        # Zero triggers degeneracies in Gmsh's Delaunay recovery for nearly
        # coplanar profile points.  This tiny fixed numerical perturbation is
        # deterministic and does not alter the CAD geometry or case seed.
        gmsh.option.setNumber("Mesh.RandomFactor", 1.0e-8)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.add("perforated_fin")
        occ = gmsh.model.occ
        height, width, thickness = float(params["height"]), float(params["width"]), float(params["thickness"])
        taper, waviness = float(params["taper"]), float(params["waviness"])
        z = np.linspace(0.0, height, 25)
        half_width = 0.5 * width * (1.0 - taper * z / height) * (1.0 + waviness * np.sin(2.0 * np.pi * z / height))
        profile = [(-half_width[0], 0.0), (half_width[0], 0.0)]
        profile.extend((float(half_width[index]), float(z[index])) for index in range(1, len(z)))
        profile.extend((-float(half_width[index]), float(z[index])) for index in range(len(z) - 1, 0, -1))
        points = [occ.addPoint(x, -0.5 * thickness, zz) for x, zz in profile]
        lines = [occ.addLine(points[index], points[(index + 1) % len(points)]) for index in range(len(points))]
        face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        extruded = occ.extrude([(2, face)], 0.0, thickness, 0.0)
        volume_tag = next(tag for dim, tag in extruded if dim == 3)
        holes = []
        for x, zz, radius in zip(params["hole_x"], params["hole_z"], params["hole_r"]):
            holes.append((3, occ.addCylinder(float(x), -0.5 * thickness, float(zz), 0.0, thickness, 0.0, float(radius))))
        cut, _ = occ.cut([(3, volume_tag)], holes, removeObject=True, removeTool=True)
        if len(cut) != 1:
            raise RuntimeError("Boolean hole subtraction did not produce exactly one cooling-fin volume.")
        occ.synchronize()

        # Use Gmsh-native distance fields rather than a Python size callback:
        # this keeps refinement focused around the physical flux hot spots
        # without serializing Gmsh's inner meshing loop through Python.
        surface_entities = gmsh.model.getEntities(2)
        base_faces: list[int] = []
        hole_faces: list[int] = []
        hole_x = np.asarray(params["hole_x"], dtype=np.float64)
        hole_z = np.asarray(params["hole_z"], dtype=np.float64)
        for _dim, tag in surface_entities:
            center = np.asarray(occ.getCenterOfMass(2, tag), dtype=np.float64)
            if abs(center[2]) < 1.0e-9:
                base_faces.append(tag)
            # A cylindrical hole wall has its center at the cylinder axis and
            # midway through the extrusion thickness; large planar faces do not.
            nearby_hole = np.min(np.hypot(center[0] - hole_x, center[2] - hole_z))
            if abs(center[1]) < 0.15 * thickness and nearby_hole < 1.0e-7:
                hole_faces.append(tag)
        if not base_faces or len(hole_faces) != len(hole_x):
            raise RuntimeError(
                f"Could not identify all refinement surfaces: base={base_faces}, holes={hole_faces}."
            )

        def threshold_from_faces(faces: list[int], distance_max: float) -> int:
            distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance, "FacesList", faces)
            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMin", h_min)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMax", h_max)
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(threshold, "DistMax", distance_max)
            return threshold

        hole_threshold = threshold_from_faces(hole_faces, 0.105)
        base_threshold = threshold_from_faces(base_faces, 0.085)
        combined = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(combined, "FieldsList", [hole_threshold, base_threshold])
        gmsh.model.mesh.field.setAsBackgroundMesh(combined)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(node_tags)
        sorted_tags = np.asarray(node_tags, dtype=np.int64)[order]
        points = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)[order]
        _, tetra_tags = gmsh.model.mesh.getElementsByType(4)
        tetra = np.searchsorted(sorted_tags, np.asarray(tetra_tags, dtype=np.int64)).reshape(-1, 4)
        if tetra.size == 0:
            raise RuntimeError("Gmsh produced no tetrahedra.")
        return points.astype(np.float64), tetra.astype(np.int64)
    finally:
        gmsh.finalize()


def tetra_volumes(points: np.ndarray, tetra: np.ndarray) -> np.ndarray:
    coords = points[tetra]
    return np.abs(np.einsum("ij,ij->i", coords[:, 1] - coords[:, 0], np.cross(coords[:, 2] - coords[:, 0], coords[:, 3] - coords[:, 0]))) / 6.0


def geometry_heat_source(points: np.ndarray, params: dict, source_strength: float) -> np.ndarray:
    """Deterministic, geometry-linked internal heating distributed through the fin."""
    x, y, z = points.T
    thickness = float(params["thickness"])
    source = np.full(points.shape[0], 0.22, dtype=np.float64)
    for index, (hole_x, hole_z, hole_r) in enumerate(zip(params["hole_x"], params["hole_z"], params["hole_r"])):
        # Every heat-generation lobe is centered on a visible geometric hole;
        # no latent random forcing is introduced into the learning problem.
        radial = ((x - float(hole_x)) / (2.35 * float(hole_r))) ** 2
        vertical = ((z - float(hole_z)) / (2.10 * float(hole_r))) ** 2
        through_thickness = (y / max(0.40 * thickness, 1.0e-8)) ** 2
        source += (0.72 + 0.10 * index) * np.exp(-(radial + vertical + through_thickness))
    return float(source_strength) * source


def solve_heat(
    points: np.ndarray,
    tetra: np.ndarray,
    params: dict,
    conductivity: float,
    convection: float,
    radiation: float,
    nonlinear_conductivity: float,
    source_strength: float,
):
    """Solve nonlinear conduction with geometry-linked heating and radiative loss.

    The Picard scheme solves the physically monotone equation
    ``-div(k(T) grad T)=q_geo`` with ``k(T)=k0(1+aT^2)`` and boundary loss
    ``q_n=hT+rT^3``.  Both nonlinearities remain positive, which makes every
    linearized system symmetric positive definite and avoids synthetic targets.
    """
    from skfem import Basis, BilinearForm, FacetBasis, LinearForm, MeshTet, asm
    from skfem.element import ElementTetP1
    from skfem.helpers import dot, grad
    import pyamg

    mesh = MeshTet(points.T, tetra.T)
    basis = Basis(mesh, ElementTetP1())
    @BilinearForm
    def diffusion(u, v, w):
        return w["kappa"] * dot(grad(u), grad(v))
    @BilinearForm
    def robin(u, v, w):
        return w["loss"] * u * v
    @LinearForm
    def source(v, w):
        return w["source"] * v
    boundary = mesh.boundary_facets()
    if boundary.size == 0:
        raise RuntimeError("Unable to identify exposed fin boundary facets.")
    boundary_basis = FacetBasis(mesh, ElementTetP1(), facets=boundary)
    source_values = geometry_heat_source(points, params, source_strength)
    rhs = asm(source, basis, source=basis.interpolate(source_values))
    temperature = np.full(points.shape[0], 0.15, dtype=np.float64)
    residual = np.inf
    nonlinear_change = np.inf
    for iteration in range(40):
        kappa = conductivity * (1.0 + nonlinear_conductivity * np.square(temperature))
        # Picard linearization of r*T^3.  A small positive convection term
        # anchors the temperature even on the first iteration.
        boundary_temperature = boundary_basis.interpolate(temperature)
        loss = convection + radiation * np.square(boundary_temperature)
        stiffness = asm(diffusion, basis, kappa=basis.interpolate(kappa))
        stiffness = stiffness + asm(robin, boundary_basis, loss=loss)
        system = stiffness.tocsr()
        preconditioner = pyamg.smoothed_aggregation_solver(system).aspreconditioner()
        updated, info = cg(system, rhs, rtol=1.0e-10, atol=0.0, maxiter=20_000, M=preconditioner)
        if info != 0:
            updated = spsolve(system, rhs)
        residual = np.linalg.norm(system @ updated - rhs) / max(np.linalg.norm(rhs), 1.0e-12)
        if not np.isfinite(residual) or residual > 2.0e-8:
            raise RuntimeError(f"Nonlinear heat linear solve residual is too high: {residual:.3e}")
        # Under-relaxation gives a monotone, robust fixed-point solve for the
        # deliberately strong nonlinear coefficients used in this benchmark.
        next_temperature = 0.65 * updated + 0.35 * temperature
        nonlinear_change = np.linalg.norm(next_temperature - temperature) / max(np.linalg.norm(next_temperature), 1.0e-12)
        temperature = next_temperature
        if nonlinear_change < 2.0e-7:
            break
    else:
        raise RuntimeError(f"Nonlinear heat solve did not converge; relative iterate change={nonlinear_change:.3e}.")
    if not np.isfinite(temperature).all() or temperature.min() < -1.0e-7:
        raise RuntimeError("Nonlinear heat solution is non-finite or violates positivity.")
    return temperature, float(residual), float(nonlinear_change), int(iteration + 1)


def boundary_triangles(points: np.ndarray, tetra: np.ndarray):
    templates = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int64)
    faces = tetra[:, templates].reshape(-1, 3)
    owners = np.repeat(np.arange(tetra.shape[0], dtype=np.int64), 4)
    ordered = np.sort(faces, axis=1)
    _, first, counts = np.unique(ordered, axis=0, return_index=True, return_counts=True)
    keep = first[counts == 1]
    faces, owners = faces[keep], owners[keep]
    tri = points[faces]
    owner_centers = points[tetra[owners]].mean(axis=1)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    inward = np.einsum("ij,ij->i", normal, owner_centers - tri.mean(axis=1)) > 0.0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    tri = points[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(2.0 * area[:, None], 1.0e-14)
    if np.any(area <= 1.0e-14):
        raise RuntimeError("Boundary extraction found degenerate triangles.")
    return faces, owners, area, normals


def tetra_gradients(points: np.ndarray, tetra: np.ndarray, values: np.ndarray) -> np.ndarray:
    xyz = points[tetra]
    matrices = np.concatenate([np.ones((xyz.shape[0], 4, 1)), xyz], axis=2)
    coeff = np.linalg.solve(matrices, values[tetra][..., None])[..., 0]
    return coeff[:, 1:]


def barycentric_samples(rng: np.random.Generator, vertices: np.ndarray) -> np.ndarray:
    weights = -np.log(np.maximum(rng.random((vertices.shape[0], vertices.shape[1])), 1.0e-12))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ni,nij->nj", weights, vertices)


def interpolate_tet_field(points: np.ndarray, tetra: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Evaluate a P1 tetrahedral field at in-domain query points."""
    from skfem import MeshTet

    mesh = MeshTet(points.T, tetra.T)
    elements = mesh.element_finder()(query[:, 0], query[:, 1], query[:, 2])
    if np.any(elements < 0):
        raise RuntimeError("Mesh-convergence query lies outside the coarse mesh.")
    xyz = points[tetra[elements]]
    matrices = np.concatenate([np.ones((query.shape[0], 4, 1)), xyz], axis=2)
    coeff = np.linalg.solve(matrices, values[tetra[elements]][..., None])[..., 0]
    return coeff[:, 0] + np.einsum("ni,ni->n", coeff[:, 1:], query)


def verify_mesh_convergence(args: dict, case_id: int = 0) -> dict:
    """Compare coarse and production meshes for the same deterministic geometry."""
    seed = int(np.random.SeedSequence([args["seed"], case_id, 9127]).generate_state(1)[0])
    params = fin_parameters(seed)
    fine_points, fine_tetra = make_tetra_mesh(params, args["mesh_h_min"], args["mesh_h_max"], args["gmsh_threads"])
    coarse_points, coarse_tetra = make_tetra_mesh(params, args["mesh_h_min"] * 1.45, args["mesh_h_max"] * 1.45, args["gmsh_threads"])
    fine_temperature, fine_residual, fine_change, fine_iterations = solve_heat(
        fine_points, fine_tetra, params, args["conductivity"], args["convection"], args["radiation"],
        args["nonlinear_conductivity"], args["source_strength"],
    )
    coarse_temperature, coarse_residual, coarse_change, coarse_iterations = solve_heat(
        coarse_points, coarse_tetra, params, args["conductivity"], args["convection"], args["radiation"],
        args["nonlinear_conductivity"], args["source_strength"],
    )
    rng = np.random.default_rng(np.random.SeedSequence([args["seed"], case_id, 441]))
    volumes = tetra_volumes(fine_points, fine_tetra)
    cells = rng.choice(fine_tetra.shape[0], size=min(4096, fine_tetra.shape[0]), replace=True, p=volumes / volumes.sum())
    bary = -np.log(np.maximum(rng.random((cells.shape[0], 4)), 1.0e-12)); bary /= bary.sum(axis=1, keepdims=True)
    query = np.einsum("ni,nij->nj", bary, fine_points[fine_tetra[cells]])
    fine_values = np.einsum("ni,ni->n", bary, fine_temperature[fine_tetra[cells]])
    coarse_values = interpolate_tet_field(coarse_points, coarse_tetra, coarse_temperature, query)
    relative = float(np.linalg.norm(coarse_values - fine_values) / max(np.linalg.norm(fine_values), 1.0e-12))
    return {
        "case_id": case_id,
        "coarse_nodes": int(coarse_points.shape[0]), "fine_nodes": int(fine_points.shape[0]),
        "coarse_tetrahedra": int(coarse_tetra.shape[0]), "fine_tetrahedra": int(fine_tetra.shape[0]),
        "coarse_to_fine_temperature_rel_l2": relative,
        "coarse_solver_relative_residual": coarse_residual, "fine_solver_relative_residual": fine_residual,
        "coarse_nonlinear_relative_change": coarse_change, "fine_nonlinear_relative_change": fine_change,
        "coarse_nonlinear_iterations": coarse_iterations, "fine_nonlinear_iterations": fine_iterations,
    }


def generate_case(case_id: int, split: str, args: dict) -> dict:
    started = time.perf_counter()
    root = Path(args["output_dir"]); case_dir = root / f"case_{case_id:05d}"
    marker = case_dir / "_COMPLETE.json"
    if marker.is_file() and not args["overwrite"]:
        return {"case_id": case_id, "skipped": True}
    seed = int(np.random.SeedSequence([args["seed"], case_id, 9127]).generate_state(1)[0])
    params = fin_parameters(seed)
    points, tetra = make_tetra_mesh(params, args["mesh_h_min"], args["mesh_h_max"], args["gmsh_threads"])
    volumes = tetra_volumes(points, tetra)
    if np.any(volumes <= 1.0e-14):
        raise RuntimeError("Invalid tetrahedral mesh with non-positive volume.")
    temperature, residual, nonlinear_change, nonlinear_iterations = solve_heat(
        points, tetra, params, args["conductivity"], args["convection"], args["radiation"],
        args["nonlinear_conductivity"], args["source_strength"],
    )
    faces, owners, areas, normals = boundary_triangles(points, tetra)
    area_p05, area_p50, area_p95 = np.percentile(areas, [5.0, 50.0, 95.0])
    nonuniformity = float(area_p95 / max(area_p05, 1.0e-20))
    if nonuniformity < float(args["min_surface_area_ratio"]):
        raise RuntimeError(
            "Adaptive surface mesh is insufficiently non-uniform: "
            f"p95/p05 triangle-area ratio={nonuniformity:.2f}, "
            f"required>={args['min_surface_area_ratio']:.2f}."
        )
    gradients = tetra_gradients(points, tetra, temperature)
    element_temperature = temperature[tetra].mean(axis=1)
    element_conductivity = args["conductivity"] * (1.0 + args["nonlinear_conductivity"] * np.square(element_temperature))
    face_flux = -element_conductivity[owners] * np.einsum("ij,ij->i", gradients[owners], normals)
    rng = np.random.default_rng(np.random.SeedSequence([args["seed"], case_id, 221]))
    # Native cloud: every triangle has equal selection probability.  Adaptive
    # refinement therefore creates genuine higher local point density.
    native_faces = rng.integers(0, faces.shape[0], size=args["geometry_points"])
    geometry = barycentric_samples(rng, points[faces[native_faces]])
    # Reference surface samples are area-uniform and independent of the input.
    surface_faces = rng.choice(faces.shape[0], size=args["surface_points"], replace=True, p=areas / areas.sum())
    surface = barycentric_samples(rng, points[faces[surface_faces]])
    surface_data = face_flux[surface_faces, None].astype(np.float32)
    volume_cells = rng.choice(tetra.shape[0], size=args["volume_points"], replace=True, p=volumes / volumes.sum())
    bary = -np.log(np.maximum(rng.random((args["volume_points"], 4)), 1.0e-12)); bary /= bary.sum(axis=1, keepdims=True)
    volume = np.einsum("ni,nij->nj", bary, points[tetra[volume_cells]])
    volume_data = np.einsum("ni,ni->n", bary, temperature[tetra[volume_cells]])[:, None].astype(np.float32)
    case_dir.mkdir(parents=True, exist_ok=True)
    save_array(case_dir / "geometry_coords.npy", geometry)
    save_array(case_dir / "surface_coords.npy", surface)
    save_array(case_dir / "surface_data.npy", surface_data)
    save_array(case_dir / "volume_coords.npy", volume)
    save_array(case_dir / "volume_data.npy", volume_data)
    save_array(case_dir / "surface_mesh_points.npy", points)
    save_array(case_dir / "surface_mesh_faces.npy", faces, dtype=np.int64)
    save_array(case_dir / "surface_fem_face_flux.npy", face_flux)
    save_array(case_dir / "volume_mesh_tetra.npy", tetra, dtype=np.int64)
    save_array(case_dir / "fem_nodal_temperature.npy", temperature)
    metadata = {
        "case_id": case_id, "split": split, "parameters": params,
        "surface_sum": float(surface_data.sum()), "surface_sq_sum": float(np.square(surface_data).sum()), "surface_count": int(surface_data.shape[0]),
        "volume_sum": float(volume_data.sum()), "volume_sq_sum": float(np.square(volume_data).sum()), "volume_count": int(volume_data.shape[0]),
        "position_min": np.minimum(geometry.min(axis=0), np.minimum(surface.min(axis=0), volume.min(axis=0))).tolist(),
        "position_max": np.maximum(geometry.max(axis=0), np.maximum(surface.max(axis=0), volume.max(axis=0))).tolist(),
        "mesh": {
            "nodes": int(points.shape[0]), "tetrahedra": int(tetra.shape[0]),
            "surface_triangles": int(faces.shape[0]), "min_tetra_volume": float(volumes.min()),
            "linear_solver_relative_residual": residual,
            "nonlinear_relative_change": nonlinear_change,
            "nonlinear_iterations": nonlinear_iterations,
            "surface_triangle_area_p05": float(area_p05),
            "surface_triangle_area_p50": float(area_p50),
            "surface_triangle_area_p95": float(area_p95),
            "surface_triangle_area_p95_over_p05": nonuniformity,
        },
        "physics": {
            "equation": "-div(k(T) grad(T)) = q_geo(x; holes)",
            "conductivity": args["conductivity"], "nonlinear_conductivity": args["nonlinear_conductivity"],
            "convection": args["convection"], "radiation": args["radiation"],
            "source_strength": args["source_strength"], "ambient_temperature": 0.0,
        },
        "generator": "toy_perforated_fin_nonlinear_fem_v2",
    }
    (case_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    marker.write_text(json.dumps({"case_id": case_id, "split": split}) + "\n", encoding="utf-8")
    return {
        "case_id": case_id,
        "skipped": False,
        "elapsed_seconds": time.perf_counter() - started,
        **metadata["mesh"],
    }


def cache_density(case_id: int, root: str, bounds: np.ndarray, knn_k: int) -> None:
    case_dir = Path(root) / f"case_{case_id:05d}"
    coords = np.asarray(np.load(case_dir / "geometry_coords.npy", mmap_mode="r"), dtype=np.float32)
    normalized = np.clip((coords - bounds[0]) / np.maximum(bounds[1] - bounds[0], 1.0e-12), 0.0, 1.0 - 1.0e-6)
    density = estimate_log_sampling_density(torch.from_numpy(normalized).unsqueeze(0), knn_k=knn_k, estimator="kde").squeeze(0).cpu().numpy()
    save_array(case_dir / f"geometry_log_density_k{knn_k}_kde.npy", density, dtype=np.float16)


def physical_cpu_count() -> int:
    """Return physical-core count on Linux, falling back conservatively."""
    topology = Path("/sys/devices/system/cpu")
    cores: set[tuple[str, str]] = set()
    for cpu_dir in topology.glob("cpu[0-9]*"):
        try:
            package = (cpu_dir / "topology" / "physical_package_id").read_text().strip()
            core = (cpu_dir / "topology" / "core_id").read_text().strip()
            cores.add((package, core))
        except OSError:
            continue
    return len(cores) if cores else max(1, (os.cpu_count() or 1) // 2)


def resolve_workers(requested: int, fallback: int) -> int:
    if requested > 0:
        return requested
    return fallback


def worker_initializer() -> None:
    # Each process owns one Gmsh/FEM or kNN task.  Avoid nested BLAS/OpenMP
    # pools, which otherwise make a 40-process job slower rather than faster.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("SMART_KNN_N_JOBS", "1")


def parse_case_ids(raw: str) -> list[int]:
    try:
        case_ids = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--export-cases must be a comma-separated list of non-negative integers.") from exc
    if not case_ids or any(case_id < 0 for case_id in case_ids):
        raise ValueError("--export-cases must contain at least one non-negative case id.")
    return list(dict.fromkeys(case_ids))


def export_case_vtps(root: Path, case_id: int, results_dir: Path) -> None:
    """Export the actual FEM mesh and solution for visual ground-truth audit."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    case_dir = root / f"case_{case_id:05d}"
    points = np.asarray(np.load(case_dir / "surface_mesh_points.npy", mmap_mode="r"), dtype=np.float32)
    faces = np.asarray(np.load(case_dir / "surface_mesh_faces.npy", mmap_mode="r"), dtype=np.int64)
    tetra = np.asarray(np.load(case_dir / "volume_mesh_tetra.npy", mmap_mode="r"), dtype=np.int64)
    temperature = np.asarray(np.load(case_dir / "fem_nodal_temperature.npy", mmap_mode="r"), dtype=np.float32)
    face_flux = np.asarray(np.load(case_dir / "surface_fem_face_flux.npy", mmap_mode="r"), dtype=np.float32)
    results_dir.mkdir(parents=True, exist_ok=True)
    vtk_points = vtk.vtkPoints(); vtk_points.SetData(numpy_to_vtk(points, deep=True))
    surface = vtk.vtkPolyData(); surface.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 3 * faces.shape[0] + 1, 3, dtype=np.int64), deep=True), numpy_to_vtkIdTypeArray(faces.reshape(-1), deep=True))
    surface.SetPolys(cells)
    nodal_flux = np.zeros(points.shape[0], dtype=np.float64); counts = np.zeros(points.shape[0], dtype=np.float64)
    np.add.at(nodal_flux, faces.reshape(-1), np.repeat(face_flux, 3)); np.add.at(counts, faces.reshape(-1), 1.0)
    for name, values in {"temperature": temperature, "outward_heat_flux": (nodal_flux / np.maximum(counts, 1.0)).astype(np.float32)}.items():
        array = numpy_to_vtk(np.ascontiguousarray(values), deep=True); array.SetName(name); surface.GetPointData().AddArray(array)
    surface_writer = vtk.vtkXMLPolyDataWriter(); surface_writer.SetFileName(str(results_dir / f"perforated_fin_case_{case_id:05d}_surface_ground_truth.vtp")); surface_writer.SetInputData(surface); surface_writer.SetDataModeToBinary(); surface_writer.SetCompressor(None)
    if surface_writer.Write() != 1: raise RuntimeError("Failed to write surface ground-truth VTP.")
    volume = vtk.vtkUnstructuredGrid(); volume.SetPoints(vtk_points)
    tetra_cells = vtk.vtkCellArray()
    tetra_cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 4 * tetra.shape[0] + 1, 4, dtype=np.int64), deep=True), numpy_to_vtkIdTypeArray(tetra.reshape(-1), deep=True))
    cell_types = np.full(tetra.shape[0], vtk.VTK_TETRA, dtype=np.uint8)
    volume.SetCells(numpy_to_vtk(cell_types, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR), tetra_cells)
    temp_array = numpy_to_vtk(np.ascontiguousarray(temperature), deep=True); temp_array.SetName("temperature"); volume.GetPointData().AddArray(temp_array)
    volume_writer = vtk.vtkXMLUnstructuredGridWriter(); volume_writer.SetFileName(str(results_dir / f"perforated_fin_case_{case_id:05d}_volume_ground_truth.vtu")); volume_writer.SetInputData(volume); volume_writer.SetDataModeToBinary(); volume_writer.SetCompressor(None)
    if volume_writer.Write() != 1: raise RuntimeError("Failed to write volume ground-truth VTU.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/mnt/ssdraid/parsa/toy_perforated_fin_nonlinear_fem_v2")
    parser.add_argument("--results-dir", default="/home/parsa/smart_parsa/results/toy_perforated_fin_nonlinear_fem")
    parser.add_argument("--train-cases", type=int, default=128)
    parser.add_argument("--validation-cases", type=int, default=32)
    # Keep twice the fair 131K encoder budget so SATLOSS can form two shifted
    # 131K views without replacement, as it does from DrivAerML's full source.
    parser.add_argument("--geometry-points", type=int, default=262144)
    parser.add_argument("--surface-points", type=int, default=65536)
    parser.add_argument("--volume-points", type=int, default=65536)
    parser.add_argument("--mesh-h-min", type=float, default=0.0035)
    parser.add_argument("--mesh-h-max", type=float, default=0.035)
    parser.add_argument("--conductivity", type=float, default=1.0)
    parser.add_argument("--convection", type=float, default=1.25)
    parser.add_argument("--radiation", type=float, default=3.5, help="Dimensionless nonlinear radiative boundary coefficient.")
    parser.add_argument("--nonlinear-conductivity", type=float, default=2.5, help="Coefficient a in k(T)=k0*(1+a*T^2).")
    parser.add_argument("--source-strength", type=float, default=18.0, help="Scale of deterministic, hole-linked volumetric heating.")
    parser.add_argument("--density-knn-k", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0, help="Legacy fallback for both stages; 0 uses all physical CPU cores.")
    parser.add_argument("--mesh-workers", type=int, default=0, help="Concurrent Gmsh/FEM cases; 0 inherits --workers or uses all physical cores.")
    parser.add_argument("--density-workers", type=int, default=0, help="Concurrent KDE-cache cases; 0 inherits --workers or uses all physical cores.")
    parser.add_argument("--gmsh-threads", type=int, default=1, help="Threads within each Gmsh process. Keep 1 when using many mesh workers.")
    parser.add_argument("--min-surface-area-ratio", type=float, default=30.0, help="Required p95/p05 surface-triangle area ratio; audits adaptive mesh non-uniformity.")
    parser.add_argument("--export-cases", default="0,1,2", help="Comma-separated case ids to export as surface VTP and volume VTU ground truth.")
    parser.add_argument("--verify-mesh-convergence", action="store_true", help="Compare production mesh temperature to a 1.45x coarser mesh for case 0 before generating the dataset.")
    parser.add_argument("--mesh-convergence-tolerance", type=float, default=0.010)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.train_cases, args.validation_cases, args.geometry_points, args.surface_points, args.volume_points) <= 0:
        raise ValueError("Case and point budgets must be positive.")
    if args.mesh_h_min <= 0.0 or args.mesh_h_max <= args.mesh_h_min:
        raise ValueError("Require 0 < --mesh-h-min < --mesh-h-max.")
    if args.min_surface_area_ratio <= 1.0:
        raise ValueError("--min-surface-area-ratio must exceed one.")
    root = Path(args.output_dir).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    physical_cores = physical_cpu_count()
    worker_fallback = resolve_workers(args.workers, physical_cores)
    mesh_workers = resolve_workers(args.mesh_workers, worker_fallback)
    density_workers = resolve_workers(args.density_workers, worker_fallback)
    if min(mesh_workers, density_workers, args.gmsh_threads) <= 0:
        raise ValueError("Worker and Gmsh thread counts must be positive.")
    print(
        "Generation parallelism: "
        f"physical_cores={physical_cores}, mesh_workers={mesh_workers}, "
        f"density_workers={density_workers}, gmsh_threads={args.gmsh_threads}."
    )
    records = [(index, "train") for index in range(args.train_cases)] + [(args.train_cases + index, "validation") for index in range(args.validation_cases)]
    args_dict = vars(args).copy()
    if args.verify_mesh_convergence:
        convergence = verify_mesh_convergence(args_dict)
        if convergence["coarse_to_fine_temperature_rel_l2"] > float(args.mesh_convergence_tolerance):
            raise RuntimeError(f"Mesh-convergence check failed: {convergence}")
        (root / "mesh_convergence_check.json").write_text(json.dumps(convergence, indent=2) + "\n", encoding="utf-8")
        print(f"Mesh convergence: {convergence}")
    generated_seconds: list[float] = []
    # The optional convergence audit initializes Gmsh and PyAMG in the parent.
    # Forking after those native runtimes can inherit locked thread state and
    # leave every child asleep.  Spawn gives each FEM worker a clean process.
    worker_context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=mesh_workers,
        mp_context=worker_context,
        initializer=worker_initializer,
    ) as pool:
        futures = [pool.submit(generate_case, case_id, split, args_dict) for case_id, split in records]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Meshing and solving fins"):
            result = future.result()
            if not result.get("skipped", False):
                generated_seconds.append(float(result["elapsed_seconds"]))
    if generated_seconds:
        print(
            f"FEM generation complete: mean_case_seconds={np.mean(generated_seconds):.2f}, "
            f"aggregate_case_seconds={np.sum(generated_seconds):.1f}."
        )
    manifest = {"version": "toy_perforated_fin_nonlinear_fem_v2", "seed": args.seed, "train_ids": list(range(args.train_cases)), "validation_ids": list(range(args.train_cases, args.train_cases + args.validation_cases)), "geometry_points": args.geometry_points, "surface_points": args.surface_points, "volume_points": args.volume_points}
    (root / "preprocessed_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    mins, maxs = np.full(3, np.inf), np.full(3, -np.inf); sums = [0.0, 0.0, 0, 0]
    for case_id in manifest["train_ids"]:
        meta = json.loads((root / f"case_{case_id:05d}" / "case_metadata.json").read_text(encoding="utf-8"))
        mins = np.minimum(mins, meta["position_min"]); maxs = np.maximum(maxs, meta["position_max"])
        sums[0] += meta["surface_sum"]; sums[1] += meta["surface_sq_sum"]; sums[2] += meta["surface_count"]
        sums[3] += 0  # Kept for explicit train-only statistic construction below.
    def field_stats(prefix: str):
        total = total_sq = 0.0; count = 0
        for case_id in manifest["train_ids"]:
            meta = json.loads((root / f"case_{case_id:05d}" / "case_metadata.json").read_text(encoding="utf-8"))
            total += float(meta[f"{prefix}_sum"]); total_sq += float(meta[f"{prefix}_sq_sum"]); count += int(meta[f"{prefix}_count"])
        mean = total / count; std = np.sqrt(max((total_sq - total * total / count) / max(count - 1, 1), 1.0e-12))
        return np.asarray([mean, std], dtype=np.float32)
    np.save(root / "surface_stats_toy_perforated_fin_nonlinear_fem_train_stats_v2.npy", field_stats("surface"), allow_pickle=False)
    np.save(root / "volume_stats_toy_perforated_fin_nonlinear_fem_train_stats_v2.npy", field_stats("volume"), allow_pickle=False)
    bounds = np.stack([mins, maxs]).astype(np.float32)
    np.save(root / "position_stats_toy_perforated_fin_nonlinear_fem_train_stats_v2.npy", bounds, allow_pickle=False)
    with ProcessPoolExecutor(
        max_workers=density_workers,
        mp_context=worker_context,
        initializer=worker_initializer,
    ) as pool:
        futures = [pool.submit(cache_density, case_id, str(root), bounds, args.density_knn_k) for case_id, _ in records]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Caching KDE-{args.density_knn_k}"):
            future.result()
    results_dir = Path(args.results_dir).expanduser().resolve()
    export_ids = parse_case_ids(args.export_cases)
    available_ids = set(manifest["train_ids"] + manifest["validation_ids"])
    for case_id in export_ids:
        if case_id not in available_ids:
            raise ValueError(f"Requested export case {case_id} is outside the generated case ids.")
        export_case_vtps(root, case_id, results_dir)
    print(f"Completed mesh-FEM perforated-fin benchmark: {root}")


if __name__ == "__main__":
    main()
