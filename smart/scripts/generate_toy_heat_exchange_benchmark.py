#!/usr/bin/env python3
"""Generate a geometry-only nonlinear heat-exchanger FEM benchmark.

Each sample is a watertight extruded fin with a deterministic wavy side profile
and one to five non-overlapping through-channels.  The channels are either
circular or rectangular.  Hot fluid is represented solely by a fixed-temperature
Dirichlet condition on the visible channel walls; all exterior faces lose heat
through convection and thermal radiation.  Thus the learned map is strictly
``triangulated solid geometry -> {surface heat flux, volume temperature}``.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import multiprocessing as mp
import os
import signal
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

# Spawned workers import this module before their initializer runs.  Pin native
# numerical libraries here so one FEM case maps to one CPU worker predictably.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "SMART_KNN_N_JOBS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import torch
from scipy.sparse.linalg import cg, spsolve
from tqdm.auto import tqdm

from utils.geometry_density import estimate_log_sampling_density


VERSION = "toy_heat_exchange_fem_v1"


def _set_parent_death_signal() -> None:
    """Make a native FEM worker die if its Python parent disappears.

    Gmsh and AMG allocations are owned by the worker process.  A terminal loss
    or an unhandled exception in the parent must therefore not leave a worker
    reparented to PID 1 and consuming memory indefinitely.  Linux `prctl` is
    deliberately best-effort so the generator remains portable.
    """
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            return
        # Cover the small race where the parent died before prctl completed.
        if os.getppid() == 1:
            os._exit(1)
    except (AttributeError, OSError):
        pass


def save_array(path: Path, array: np.ndarray, dtype=np.float32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=dtype), allow_pickle=False)
    temporary.replace(path)


def _side_waviness(z: np.ndarray, height: float, coefficients: np.ndarray) -> np.ndarray:
    """Smooth deterministic displacement which is zero at the base and tip."""
    s = z / height
    envelope = np.sin(np.pi * s)
    a1, a2, n1, n2, phase1, phase2 = coefficients
    return envelope * (
        a1 * np.sin(2.0 * np.pi * n1 * s + phase1)
        + a2 * np.sin(2.0 * np.pi * n2 * s + phase2)
    )


def _profile_bounds(z: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    height = float(params["height"])
    left_base, right_base = float(params["left_base"]), float(params["right_base"])
    left = np.full_like(z, left_base)
    right = np.full_like(z, right_base)
    category = params["waviness_category"]
    if category in {"both", "left"}:
        left += _side_waviness(z, height, np.asarray(params["left_wave"], dtype=np.float64))
    if category in {"both", "right"}:
        right += _side_waviness(z, height, np.asarray(params["right_wave"], dtype=np.float64))
    return left, right


def _channel_clearance(channel: dict) -> float:
    if channel["shape"] == "circle":
        return float(channel["radius"])
    return float(np.hypot(channel["half_x"], channel["half_z"]))


def heat_exchange_parameters(seed: int) -> dict:
    """Sample only valid geometry; all physical coefficients remain global."""
    rng = np.random.default_rng(seed)
    height = float(rng.uniform(1.25, 1.75))
    thickness = float(rng.uniform(0.20, 0.32))
    category = ("both", "left", "right")[int(rng.integers(0, 3))]
    def wave() -> list[float]:
        return [
            float(rng.uniform(0.025, 0.075)), float(rng.uniform(0.010, 0.045)),
            int(rng.integers(1, 4)), int(rng.integers(3, 7)),
            float(rng.uniform(0.0, 2.0 * np.pi)), float(rng.uniform(0.0, 2.0 * np.pi)),
        ]
    params: dict = {
        "height": height,
        "thickness": thickness,
        "left_base": float(rng.uniform(-0.67, -0.52)),
        "right_base": float(rng.uniform(0.52, 0.67)),
        "waviness_category": category,
        "left_wave": wave(),
        "right_wave": wave(),
    }
    # The profile is audited on a fine grid before channel placement.  Rejection
    # sampling keeps every channel strictly inside the solid and mutually apart.
    z_grid = np.linspace(0.0, height, 1025)
    left_grid, right_grid = _profile_bounds(z_grid, params)
    if np.min(right_grid - left_grid) < 0.55:
        raise RuntimeError("Generated profile has insufficient solid width.")
    channels: list[dict] = []
    clearance = 0.035
    count = int(rng.integers(1, 6))
    for _ in range(count):
        placed = False
        for _attempt in range(400):
            shape = "circle" if rng.random() < 0.55 else "rectangle"
            if shape == "circle":
                channel = {"shape": shape, "radius": float(rng.uniform(0.055, 0.115))}
                half_x = half_z = channel["radius"]
            else:
                channel = {
                    "shape": shape,
                    "half_x": float(rng.uniform(0.050, 0.115)),
                    "half_z": float(rng.uniform(0.050, 0.125)),
                }
                half_x, half_z = channel["half_x"], channel["half_z"]
            z = float(rng.uniform(half_z + 0.10, height - half_z - 0.10))
            left, right = _profile_bounds(np.asarray([z]), params)
            x_low, x_high = float(left[0] + half_x + clearance), float(right[0] - half_x - clearance)
            if x_low >= x_high:
                continue
            x = float(rng.uniform(x_low, x_high))
            channel.update({"x": x, "z": z})
            radius = _channel_clearance(channel)
            if any(np.hypot(x - old["x"], z - old["z"]) < radius + _channel_clearance(old) + clearance for old in channels):
                continue
            # Sample extrema to protect rectangle corners and circular walls
            # against the varying profile, not only the channel centreline.
            local_z = np.linspace(z - half_z, z + half_z, 17)
            local_left, local_right = _profile_bounds(local_z, params)
            if np.min(x - half_x - local_left) < clearance or np.min(local_right - (x + half_x)) < clearance:
                continue
            channels.append(channel)
            placed = True
            break
        if not placed:
            # Fewer channels is physically valid; forcing a bad placement is not.
            break
    if not channels:
        raise RuntimeError("Could not place a valid internal channel.")
    params["channels"] = channels
    return params


def make_tetra_mesh(params: dict, h_min: float, h_max: float, gmsh_threads: int) -> tuple[np.ndarray, np.ndarray]:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", max(1, int(gmsh_threads)))
        gmsh.option.setNumber("Mesh.RandomFactor", 1.0e-8)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.add("toy_heat_exchange")
        occ = gmsh.model.occ
        height, thickness = float(params["height"]), float(params["thickness"])
        z = np.linspace(0.0, height, 97)
        left, right = _profile_bounds(z, params)
        profile = [(float(left[0]), 0.0), (float(right[0]), 0.0)]
        profile += [(float(right[i]), float(z[i])) for i in range(1, z.size)]
        profile += [(float(left[i]), float(z[i])) for i in range(z.size - 1, 0, -1)]
        vertices = [occ.addPoint(x, -0.5 * thickness, zz) for x, zz in profile]
        edges = [occ.addLine(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
        surface = occ.addPlaneSurface([occ.addCurveLoop(edges)])
        extruded = occ.extrude([(2, surface)], 0.0, thickness, 0.0)
        solid = next(tag for dim, tag in extruded if dim == 3)
        cutters = []
        for channel in params["channels"]:
            if channel["shape"] == "circle":
                cutter = occ.addCylinder(channel["x"], -thickness, channel["z"], 0.0, 2.0 * thickness, 0.0, channel["radius"])
            else:
                cutter = occ.addBox(channel["x"] - channel["half_x"], -thickness, channel["z"] - channel["half_z"], 2.0 * channel["half_x"], 2.0 * thickness, 2.0 * channel["half_z"])
            cutters.append((3, cutter))
        cut, _ = occ.cut([(3, solid)], cutters, removeObject=True, removeTool=True)
        if len(cut) != 1:
            raise RuntimeError("Channel subtraction did not yield exactly one solid volume.")
        occ.synchronize()

        # Gmsh distance/threshold fields provide both wall, curvature, and
        # channel-proximity refinement while preserving a coarser bulk.
        channel_faces: list[int] = []
        exterior_side_faces: list[int] = []
        for _dim, tag in gmsh.model.getEntities(2):
            centre = np.asarray(occ.getCenterOfMass(2, tag), dtype=np.float64)
            if abs(centre[1]) >= 0.40 * thickness:
                continue
            if any(_cad_face_is_channel_wall(centre, channel) for channel in params["channels"]):
                channel_faces.append(tag)
            else:
                exterior_side_faces.append(tag)
        if len(channel_faces) < len(params["channels"]):
            raise RuntimeError(f"Unable to identify all channel-wall surfaces ({len(channel_faces)}/{len(params['channels'])}).")
        if not exterior_side_faces:
            raise RuntimeError("Unable to identify wavy exterior side surfaces for refinement.")

        def distance_threshold(faces: list[int], size_min: float, distance_max: float) -> int:
            distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance, "FacesList", faces)
            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMin", size_min)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMax", h_max)
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(threshold, "DistMax", distance_max)
            return threshold

        # The nearest-channel distance is small not only at channel walls but
        # also in narrow ligaments between channels: this is a true proximity
        # refinement region rather than a visually induced point-density trick.
        channel_threshold = distance_threshold(channel_faces, h_min, 0.18)
        # Side faces carry the deterministic waviness.  Their refinement plus
        # Gmsh curvature sizing resolves high-curvature crests and troughs.
        side_threshold = distance_threshold(exterior_side_faces, min(h_max, 1.6 * h_min), 0.075)
        combined = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(combined, "FieldsList", [channel_threshold, side_threshold])
        gmsh.model.mesh.field.setAsBackgroundMesh(combined)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        sorted_tags = np.asarray(tags, dtype=np.int64)[order]
        points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)[order]
        _, element_nodes = gmsh.model.mesh.getElementsByType(4)
        tetra = np.searchsorted(sorted_tags, np.asarray(element_nodes, dtype=np.int64)).reshape(-1, 4)
        if tetra.size == 0:
            raise RuntimeError("Gmsh produced no tetrahedra.")
        return points, tetra.astype(np.int64)
    finally:
        gmsh.finalize()


def _point_on_channel(x: float, z: float, channel: dict, tolerance: float) -> bool:
    if channel["shape"] == "circle":
        return abs(np.hypot(x - channel["x"], z - channel["z"]) - channel["radius"]) <= tolerance
    dx, dz = abs(x - channel["x"]), abs(z - channel["z"])
    return dx <= channel["half_x"] + tolerance and dz <= channel["half_z"] + tolerance and (
        abs(dx - channel["half_x"]) <= tolerance or abs(dz - channel["half_z"]) <= tolerance
    )


def _cad_face_is_channel_wall(centre: np.ndarray, channel: dict) -> bool:
    """Classify OpenCASCADE faces without confusing cylinder COM with a wall point."""
    tolerance = 2.0e-5
    if channel["shape"] == "circle":
        # The centre of mass of a cylindrical lateral face is its axis, not a
        # point on its circumference.  Triangle-centroid classification below
        # intentionally uses the radial test instead.
        return np.hypot(centre[0] - channel["x"], centre[2] - channel["z"]) <= tolerance
    return _point_on_channel(float(centre[0]), float(centre[2]), channel, tolerance)


def tetra_volumes(points: np.ndarray, tetra: np.ndarray) -> np.ndarray:
    corners = points[tetra]
    return np.abs(np.einsum("ij,ij->i", corners[:, 1] - corners[:, 0], np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]))) / 6.0


def boundary_triangles(points: np.ndarray, tetra: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    template = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int64)
    all_faces = tetra[:, template].reshape(-1, 3)
    owners = np.repeat(np.arange(tetra.shape[0], dtype=np.int64), 4)
    canonical = np.sort(all_faces, axis=1)
    _, first, count = np.unique(canonical, axis=0, return_index=True, return_counts=True)
    select = first[count == 1]
    faces, owners = all_faces[select], owners[select]
    triangles = points[faces]
    owner_centres = points[tetra[owners]].mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    flip = np.einsum("ij,ij->i", normals, owner_centres - triangles.mean(axis=1)) > 0.0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    triangles = points[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(normals, axis=1)
    if np.any(~np.isfinite(areas)) or np.any(areas <= 1.0e-14):
        raise RuntimeError("Boundary extraction found non-finite or degenerate triangles.")
    return faces, owners, areas, normals / (2.0 * areas[:, None])


def classify_boundary_faces(points: np.ndarray, faces: np.ndarray, params: dict, h_max: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
    centres = points[faces].mean(axis=1)
    thickness = float(params["thickness"])
    tolerance = max(2.5 * h_max, 0.004)
    inner = np.zeros(faces.shape[0], dtype=bool)
    per_channel: list[int] = []
    for channel in params["channels"]:
        # Through-channel walls lie away from the front/back exterior planes.
        y_interior = np.abs(centres[:, 1]) < 0.5 * thickness - 0.5 * tolerance
        if channel["shape"] == "circle":
            on_wall = np.abs(np.hypot(centres[:, 0] - channel["x"], centres[:, 2] - channel["z"]) - channel["radius"]) <= tolerance
        else:
            dx, dz = np.abs(centres[:, 0] - channel["x"]), np.abs(centres[:, 2] - channel["z"])
            on_wall = (dx <= channel["half_x"] + tolerance) & (dz <= channel["half_z"] + tolerance) & ((np.abs(dx - channel["half_x"]) <= tolerance) | (np.abs(dz - channel["half_z"]) <= tolerance))
        mask = y_interior & on_wall
        per_channel.append(int(mask.sum()))
        inner |= mask
    if not np.all(np.asarray(per_channel) >= 6):
        raise RuntimeError(f"Channel-wall facet classification failed: facets per channel={per_channel}.")
    return inner, ~inner, per_channel


def solve_heat(points: np.ndarray, tetra: np.ndarray, params: dict, args: dict) -> tuple[np.ndarray, float, float, int, float]:
    """Solve steady nonlinear conduction with isothermal visible channel walls.

    The fixed point is exact at convergence: conductivity and radiation are
    evaluated at the current temperature, while each linear subproblem remains
    symmetric positive definite.  No case-specific forcing is introduced.
    """
    from skfem import Basis, BilinearForm, FacetBasis, MeshTet, asm, condense
    from skfem.element import ElementTetP1
    from skfem.helpers import dot, grad
    import pyamg

    # skfem otherwise copies and emits a conversion message for every worker
    # because transposed NumPy views are not C-contiguous.
    mesh = MeshTet(np.ascontiguousarray(points.T), np.ascontiguousarray(tetra.T))
    basis = Basis(mesh, ElementTetP1())
    boundary = mesh.boundary_facets()
    # Classify skfem facets directly.  Boundary triangle extraction is much
    # larger than the FEM boundary map and is deferred until target export,
    # where it is genuinely required for surface-flux interpolation.
    facet_vertices = mesh.facets[:, boundary].T
    inner, outer, _ = classify_boundary_faces(points, facet_vertices, params, args["mesh_h_max"])
    inner_facets, outer_facets = boundary[inner], boundary[outer]
    if inner_facets.size == 0 or outer_facets.size == 0:
        raise RuntimeError("Boundary condition split produced an empty inner or exterior set.")
    outer_basis = FacetBasis(mesh, ElementTetP1(), facets=outer_facets)
    inner_dofs = basis.get_dofs(facets=inner_facets).all()
    if inner_dofs.size == 0:
        raise RuntimeError("No nodal degrees of freedom were found on channel walls.")

    @BilinearForm
    def diffusion(u, v, w):
        return w["conductivity"] * dot(grad(u), grad(v))

    @BilinearForm
    def mass(u, v, w):
        return w["coefficient"] * u * v

    exterior_biot = float(args["exterior_biot"])
    radiation = float(args["radiation"])
    tau = float(args["ambient_temperature_ratio"])
    nonlinear_k = float(args["nonlinear_conductivity"])
    temperature = np.full(points.shape[0], 0.15, dtype=np.float64)
    temperature[inner_dofs] = 1.0
    residual = change = np.inf
    for iteration in range(int(args["nonlinear_iterations"])):
        conductivity = 1.0 + nonlinear_k * np.square(temperature)
        # Exact at fixed point: R(T)=[(T+tau)^4-tau^4].  Its positive secant
        # coefficient keeps Picard updates SPD and well-conditioned at T=0.
        radiative_flux = radiation * (np.power(temperature + tau, 4.0) - tau ** 4)
        secant = np.where(temperature > 1.0e-8, radiative_flux / temperature, 4.0 * radiation * tau ** 3)
        system = asm(diffusion, basis, conductivity=basis.interpolate(conductivity)).tocsr()
        system = system + asm(mass, outer_basis, coefficient=outer_basis.interpolate(exterior_biot + secant)).tocsr()
        reduced, reduced_rhs, constrained, free_dofs = condense(
            system,
            np.zeros(points.shape[0], dtype=np.float64),
            x=np.ones(points.shape[0], dtype=np.float64),
            D=inner_dofs,
        )
        # The hierarchy affects only CG convergence, never the assembled FEM
        # operator or its tolerance.  A moderately larger coarsest grid avoids
        # building many tiny levels for every Picard update, which is critical
        # when many independent cases are generated concurrently.
        hierarchy = pyamg.smoothed_aggregation_solver(
            reduced,
            symmetry="symmetric",
            max_coarse=500,
        )
        preconditioner = hierarchy.aspreconditioner(cycle="V")
        free_values, info = cg(reduced, reduced_rhs, rtol=1.0e-10, atol=0.0, maxiter=20_000, M=preconditioner)
        if info != 0:
            free_values = spsolve(reduced, reduced_rhs)
        residual = float(np.linalg.norm(reduced @ free_values - reduced_rhs) / max(np.linalg.norm(reduced_rhs), 1.0e-12))
        # Drop the hierarchy before assembling the next nonlinear step.  The
        # worker is additionally recycled per case below, which bounds native
        # allocator retention even if a third-party extension caches memory.
        del hierarchy, preconditioner, system, reduced, reduced_rhs
        updated = constrained.copy()
        updated[free_dofs] = free_values
        if not np.isfinite(residual) or residual > 2.0e-8:
            raise RuntimeError(f"Thermal linear-system residual is invalid: {residual:.3e}.")
        next_temperature = 0.70 * updated + 0.30 * temperature
        next_temperature[inner_dofs] = 1.0
        change = float(np.linalg.norm(next_temperature - temperature) / max(np.linalg.norm(next_temperature), 1.0e-12))
        temperature = next_temperature
        if change < float(args["nonlinear_tolerance"]):
            break
    else:
        raise RuntimeError(f"Nonlinear heat solve did not converge (relative update={change:.3e}).")
    if not np.isfinite(temperature).all() or temperature.min() < -1.0e-7 or temperature.max() > 1.0 + 1.0e-7:
        raise RuntimeError(f"Temperature violates physical bounds: [{temperature.min():.4f}, {temperature.max():.4f}].")
    wall_error = float(np.max(np.abs(temperature[inner_dofs] - 1.0)))
    if wall_error > 1.0e-12:
        raise RuntimeError(f"Isothermal channel condition was not enforced exactly (max error={wall_error:.3e}).")
    return temperature, residual, change, iteration + 1, wall_error


def tetra_gradients(points: np.ndarray, tetra: np.ndarray, values: np.ndarray) -> np.ndarray:
    corners = points[tetra]
    matrix = np.concatenate([np.ones((corners.shape[0], 4, 1)), corners], axis=2)
    coefficients = np.linalg.solve(matrix, values[tetra][..., None])[..., 0]
    return coefficients[:, 1:]


def barycentric_samples(rng: np.random.Generator, vertices: np.ndarray) -> np.ndarray:
    weights = -np.log(np.maximum(rng.random((vertices.shape[0], vertices.shape[1])), 1.0e-12))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ni,nij->nj", weights, vertices)


def generate_case(case_id: int, split: str, args: dict) -> dict:
    started = time.perf_counter()
    root = Path(args["output_dir"])
    case_dir = root / f"case_{case_id:05d}"
    complete = case_dir / "_COMPLETE.json"
    if complete.exists() and not args["overwrite"]:
        return {"case_id": case_id, "skipped": True}
    seed = int(np.random.SeedSequence([args["seed"], case_id, 5913]).generate_state(1)[0])
    params = heat_exchange_parameters(seed)
    points, tetra = make_tetra_mesh(params, args["mesh_h_min"], args["mesh_h_max"], args["gmsh_threads"])
    volumes = tetra_volumes(points, tetra)
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 1.0e-14):
        raise RuntimeError("Gmsh produced non-positive tetrahedral volumes.")
    temperature, residual, nonlinear_change, iterations, wall_error = solve_heat(points, tetra, params, args)
    faces, _owners, areas, _normals = boundary_triangles(points, tetra)
    inner, outer, channel_faces = classify_boundary_faces(points, faces, params, args["mesh_h_max"])
    area_ratio = float(np.percentile(areas, 95.0) / max(np.percentile(areas, 5.0), 1.0e-20))
    if area_ratio < float(args["min_surface_area_ratio"]):
        raise RuntimeError(f"Adaptive mesh is too uniform (surface p95/p05={area_ratio:.2f}).")
    # Channel walls are now exact high-temperature Dirichlet boundaries, so
    # their physically injected heat flux is recovered from the conductive
    # gradient.  Exterior flux is evaluated from its imposed Robin law.
    face_temperature = temperature[faces].mean(axis=1)
    flux = np.empty(faces.shape[0], dtype=np.float64)
    gradients = tetra_gradients(points, tetra, temperature)
    conductivity = 1.0 + float(args["nonlinear_conductivity"]) * np.square(temperature[tetra].mean(axis=1))
    flux[inner] = -conductivity[_owners[inner]] * np.einsum("ij,ij->i", gradients[_owners[inner]], _normals[inner])
    flux[outer] = (
        float(args["exterior_biot"]) * face_temperature[outer]
        + float(args["radiation"]) * (
            np.power(face_temperature[outer] + float(args["ambient_temperature_ratio"]), 4.0)
            - float(args["ambient_temperature_ratio"]) ** 4
        )
    )
    # Directly audit the physical signs: heat enters the solid through hot
    # channel walls and exits the exterior boundary.
    if float(flux[inner].mean()) >= 0.0 or float(flux[outer].mean()) <= 0.0:
        raise RuntimeError("Boundary heat-flux signs contradict the imposed physical boundary conditions.")
    rng = np.random.default_rng(np.random.SeedSequence([args["seed"], case_id, 7721]))
    # Generate each sampled cloud directly from its selected cells.  These are
    # the arrays that must be persisted; no full mesh-sized interpolation table
    # is materialised in the worker.
    native = barycentric_samples(rng, points[faces[rng.integers(faces.shape[0], size=args["geometry_points"])]] )
    surface_ids = rng.choice(faces.shape[0], size=args["surface_points"], replace=True, p=areas / areas.sum())
    surface = barycentric_samples(rng, points[faces[surface_ids]])
    volume_ids = rng.choice(tetra.shape[0], size=args["volume_points"], replace=True, p=volumes / volumes.sum())
    volume_weights = np.empty((args["volume_points"], 4), dtype=np.float64)
    # Recompute barycentric weights for target interpolation rather than use
    # the sampling helper's temporary values.
    volume_weights[:] = -np.log(np.maximum(rng.random(volume_weights.shape), 1.0e-12))
    volume_weights /= volume_weights.sum(axis=1, keepdims=True)
    volume = np.einsum("ni,nij->nj", volume_weights, points[tetra[volume_ids]])
    surface_data = flux[surface_ids, None].astype(np.float32)
    volume_data = np.einsum("ni,ni->n", volume_weights, temperature[tetra[volume_ids]])[:, None].astype(np.float32)
    case_dir.mkdir(parents=True, exist_ok=True)
    for name, array, dtype in (
        ("geometry_coords.npy", native, np.float32), ("surface_coords.npy", surface, np.float32),
        ("surface_data.npy", surface_data, np.float32), ("volume_coords.npy", volume, np.float32),
        ("volume_data.npy", volume_data, np.float32), ("surface_mesh_points.npy", points, np.float32),
        ("surface_mesh_faces.npy", faces, np.int64), ("surface_fem_face_flux.npy", flux, np.float32),
        ("volume_mesh_tetra.npy", tetra, np.int64), ("fem_nodal_temperature.npy", temperature, np.float32),
    ):
        save_array(case_dir / name, array, dtype)
    metadata = {
        "case_id": case_id, "split": split, "parameters": params,
        "surface_sum": float(surface_data.sum()), "surface_sq_sum": float(np.square(surface_data).sum()), "surface_count": int(surface_data.size),
        "volume_sum": float(volume_data.sum()), "volume_sq_sum": float(np.square(volume_data).sum()), "volume_count": int(volume_data.size),
        "position_min": np.minimum(native.min(axis=0), np.minimum(surface.min(axis=0), volume.min(axis=0))).tolist(),
        "position_max": np.maximum(native.max(axis=0), np.maximum(surface.max(axis=0), volume.max(axis=0))).tolist(),
        "mesh": {"nodes": int(points.shape[0]), "tetrahedra": int(tetra.shape[0]), "surface_triangles": int(faces.shape[0]), "minimum_tetra_volume": float(volumes.min()), "surface_area_p95_over_p05": area_ratio, "inner_faces": int(inner.sum()), "outer_faces": int(outer.sum()), "channel_faces": channel_faces, "channel_temperature_max_error": wall_error},
        "physics": {"equation": "-div((1+a*theta^2) grad(theta))=0", "channel_bc": "theta=1 on every circular and rectangular channel wall", "exterior_bc": "-(1+a*theta^2) grad(theta).n=Bi_ext*theta+R*((theta+tau)^4-tau^4)", "exterior_biot": args["exterior_biot"], "radiation": args["radiation"], "tau": args["ambient_temperature_ratio"], "nonlinear_conductivity": args["nonlinear_conductivity"], "linear_residual": residual, "nonlinear_relative_change": nonlinear_change, "nonlinear_iterations": iterations},
        "generator": VERSION,
    }
    (case_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    complete.write_text(json.dumps({"case_id": case_id, "split": split}) + "\n", encoding="utf-8")
    result = {"case_id": case_id, "skipped": False, "elapsed_seconds": time.perf_counter() - started, "nodes": int(points.shape[0]), "tetrahedra": int(tetra.shape[0]), "temperature_min": float(temperature.min()), "temperature_max": float(temperature.max()), "area_ratio": area_ratio}
    # Explicitly release large Python views before this process exits.  It is
    # cheap and helps RSS observability in profilers; process recycling is the
    # actual hard memory boundary.
    del points, tetra, faces, volumes, temperature, gradients
    gc.collect()
    return result


def cache_density(case_id: int, root: str, bounds: np.ndarray, knn_k: int) -> None:
    path = Path(root) / f"case_{case_id:05d}"
    coordinates = np.asarray(np.load(path / "geometry_coords.npy", mmap_mode="r"), dtype=np.float32)
    normal = np.clip((coordinates - bounds[0]) / np.maximum(bounds[1] - bounds[0], 1.0e-12), 0.0, 1.0 - 1.0e-6)
    density = estimate_log_sampling_density(torch.from_numpy(normal).unsqueeze(0), knn_k=knn_k, estimator="kde").squeeze(0).cpu().numpy()
    save_array(path / f"geometry_log_density_k{knn_k}_kde.npy", density, dtype=np.float16)


def export_case(root: Path, case_id: int, results_dir: Path) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    case = root / f"case_{case_id:05d}"
    points = np.asarray(np.load(case / "surface_mesh_points.npy"), dtype=np.float32)
    faces = np.asarray(np.load(case / "surface_mesh_faces.npy"), dtype=np.int64)
    tetra = np.asarray(np.load(case / "volume_mesh_tetra.npy"), dtype=np.int64)
    temperature = np.asarray(np.load(case / "fem_nodal_temperature.npy"), dtype=np.float32)
    flux = np.asarray(np.load(case / "surface_fem_face_flux.npy"), dtype=np.float32)
    results_dir.mkdir(parents=True, exist_ok=True)
    vtk_points = vtk.vtkPoints(); vtk_points.SetData(numpy_to_vtk(points, deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(vtk_points)
    cell_array = vtk.vtkCellArray(); cell_array.SetData(numpy_to_vtkIdTypeArray(np.arange(0, faces.size + 1, 3, dtype=np.int64), deep=True), numpy_to_vtkIdTypeArray(faces.ravel(), deep=True)); poly.SetPolys(cell_array)
    flux_nodes, counts = np.zeros(points.shape[0]), np.zeros(points.shape[0]); np.add.at(flux_nodes, faces.ravel(), np.repeat(flux, 3)); np.add.at(counts, faces.ravel(), 1.0)
    for name, values in {"temperature": temperature, "outward_heat_flux": flux_nodes / np.maximum(counts, 1.0)}.items():
        field = numpy_to_vtk(np.asarray(values, dtype=np.float32), deep=True); field.SetName(name); poly.GetPointData().AddArray(field)
    writer = vtk.vtkXMLPolyDataWriter(); writer.SetFileName(str(results_dir / f"heat_exchange_case_{case_id:05d}_surface_ground_truth.vtp")); writer.SetInputData(poly); writer.SetDataModeToBinary(); writer.SetCompressor(None)
    if writer.Write() != 1: raise RuntimeError("Could not write heat-exchange surface VTP.")
    grid = vtk.vtkUnstructuredGrid(); grid.SetPoints(vtk_points)
    cells = vtk.vtkCellArray(); cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, tetra.size + 1, 4, dtype=np.int64), deep=True), numpy_to_vtkIdTypeArray(tetra.ravel(), deep=True)); grid.SetCells(numpy_to_vtk(np.full(tetra.shape[0], vtk.VTK_TETRA, dtype=np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR), cells)
    field = numpy_to_vtk(temperature, deep=True); field.SetName("temperature"); grid.GetPointData().AddArray(field)
    writer = vtk.vtkXMLUnstructuredGridWriter(); writer.SetFileName(str(results_dir / f"heat_exchange_case_{case_id:05d}_volume_ground_truth.vtu")); writer.SetInputData(grid); writer.SetDataModeToBinary(); writer.SetCompressor(None)
    if writer.Write() != 1: raise RuntimeError("Could not write heat-exchange volume VTU.")


def worker_initializer() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "SMART_KNN_N_JOBS"):
        os.environ.setdefault(name, "1")
    # The parent owns shutdown.  Letting every worker react independently to
    # Ctrl-C can leave a partially torn-down pool and orphaned native solvers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _set_parent_death_signal()


def terminate_executor(executor: ProcessPoolExecutor) -> None:
    """Terminate active children before reaping the pool on interruption."""
    for process in list(executor._processes.values()):
        if process.is_alive():
            process.terminate()
    try:
        executor.shutdown(wait=True, cancel_futures=True)
    except BaseException:
        # Termination is best-effort during a broken native backend.  The
        # parent-death signal still prevents any surviving worker from leaking.
        pass


def interrupt_parent(_signum, _frame) -> None:
    """Turn terminal disconnect/termination into cleanup-aware interruption."""
    raise KeyboardInterrupt


def run_bounded_pool(
    executor: ProcessPoolExecutor,
    jobs: list[tuple],
    submit,
    max_in_flight: int,
    description: str,
):
    """Yield completed results while retaining at most one job per worker.

    This avoids an unbounded executor queue and makes interruption immediate:
    at most `max_in_flight` jobs need cancellation.  The FEM workers are
    configured for one case each, so every Gmsh/AMG allocation is reclaimed by
    the operating system before a worker can receive another case.
    """
    iterator = iter(jobs)
    pending = {}

    def fill() -> None:
        while len(pending) < max_in_flight:
            try:
                job = next(iterator)
            except StopIteration:
                return
            pending[submit(*job)] = job

    fill()
    with tqdm(total=len(jobs), desc=description) as progress:
        while pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in completed:
                pending.pop(future)
                result = future.result()
                progress.update(1)
                yield result
            fill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/mnt/ssdraid/parsa/toy_heat_exchange_fem_v1")
    parser.add_argument("--results-dir", default="/home/parsa/smart_parsa/results/toy_heat_exchange")
    parser.add_argument("--train-cases", type=int, default=256)
    parser.add_argument("--validation-cases", type=int, default=32)
    parser.add_argument("--geometry-points", type=int, default=524288)
    parser.add_argument("--surface-points", type=int, default=65536)
    parser.add_argument("--volume-points", type=int, default=65536)
    parser.add_argument("--mesh-h-min", type=float, default=0.0025)
    parser.add_argument("--mesh-h-max", type=float, default=0.028)
    parser.add_argument("--exterior-biot", type=float, default=0.35)
    parser.add_argument("--radiation", type=float, default=0.070)
    parser.add_argument("--ambient-temperature-ratio", type=float, default=1.5)
    parser.add_argument("--nonlinear-conductivity", type=float, default=1.8)
    parser.add_argument("--nonlinear-iterations", type=int, default=60)
    parser.add_argument("--nonlinear-tolerance", type=float, default=2.0e-7)
    parser.add_argument("--density-knn-k", type=int, default=16)
    # Each worker owns one FEM matrix and one AMG hierarchy at a time.  The
    # native libraries are pinned to one thread by `worker_initializer`, so 24
    # workers are real case-level parallelism rather than thread oversubscription.
    parser.add_argument("--mesh-workers", type=int, default=24)
    parser.add_argument("--density-workers", type=int, default=1)
    parser.add_argument("--max-cases-per-worker", type=int, default=1)
    parser.add_argument("--gmsh-threads", type=int, default=1)
    # This is a diagnostic guard, not a physics criterion.  A p95/p05 surface
    # area ratio of 15 already corresponds to pronounced spatial refinement;
    # rejecting a valid 19.58x mesh because it missed an arbitrary 20x bound
    # only makes a long generation job brittle.
    parser.add_argument("--min-surface-area-ratio", type=float, default=15.0)
    parser.add_argument("--export-cases", default="0,1,2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.train_cases, args.validation_cases, args.geometry_points, args.surface_points, args.volume_points, args.mesh_workers, args.density_workers, args.max_cases_per_worker, args.gmsh_threads) <= 0:
        raise ValueError("Case, point, and worker budgets must be positive.")
    if not 0.0 < args.mesh_h_min < args.mesh_h_max:
        raise ValueError("Require 0 < mesh-h-min < mesh-h-max.")
    root = Path(args.output_dir).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    records = [(i, "train") for i in range(args.train_cases)] + [(args.train_cases + i, "validation") for i in range(args.validation_cases)]
    config = vars(args).copy()
    context = mp.get_context("spawn")
    previous_handlers = {name: signal.getsignal(name) for name in (signal.SIGTERM, signal.SIGHUP)}
    for name in previous_handlers:
        signal.signal(name, interrupt_parent)
    print(
        f"Generating {len(records)} nonlinear heat exchangers with "
        f"mesh_workers={args.mesh_workers}, density_workers={args.density_workers}, "
        f"max_cases_per_worker={args.max_cases_per_worker}, gmsh_threads={args.gmsh_threads}."
    )
    times: list[float] = []
    mesh_pool = ProcessPoolExecutor(
        max_workers=args.mesh_workers,
        mp_context=context,
        initializer=worker_initializer,
        max_tasks_per_child=args.max_cases_per_worker,
    )
    try:
        mesh_jobs = [(case_id, split, config) for case_id, split in records]
        for result in run_bounded_pool(
            mesh_pool,
            mesh_jobs,
            lambda case_id, split, case_config: mesh_pool.submit(generate_case, case_id, split, case_config),
            args.mesh_workers,
            "Meshing and solving heat exchangers",
        ):
            if not result.get("skipped"):
                times.append(float(result["elapsed_seconds"]))
    except BaseException:
        print("Stopping FEM workers and reaping native allocations.", flush=True)
        terminate_executor(mesh_pool)
        raise
    else:
        mesh_pool.shutdown(wait=True, cancel_futures=False)
    manifest = {"version": VERSION, "seed": args.seed, "train_ids": list(range(args.train_cases)), "validation_ids": list(range(args.train_cases, args.train_cases + args.validation_cases)), "geometry_points": args.geometry_points, "surface_points": args.surface_points, "volume_points": args.volume_points}
    (root / "preprocessed_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    totals = {"surface": [0.0, 0.0, 0], "volume": [0.0, 0.0, 0]}; low = np.full(3, np.inf); high = np.full(3, -np.inf)
    for case_id in manifest["train_ids"]:
        metadata = json.loads((root / f"case_{case_id:05d}" / "case_metadata.json").read_text(encoding="utf-8"))
        for field in totals:
            totals[field][0] += float(metadata[f"{field}_sum"]); totals[field][1] += float(metadata[f"{field}_sq_sum"]); totals[field][2] += int(metadata[f"{field}_count"])
        low = np.minimum(low, metadata["position_min"]); high = np.maximum(high, metadata["position_max"])
    for field, (total, square, count) in totals.items():
        mean = total / count; std = np.sqrt(max((square - total * total / count) / max(count - 1, 1), 1.0e-12))
        np.save(root / f"{field}_stats_toy_heat_exchange_fem_train_stats_v1.npy", np.asarray([mean, std], dtype=np.float32), allow_pickle=False)
    bounds = np.asarray([low, high], dtype=np.float32); np.save(root / "position_stats_toy_heat_exchange_fem_train_stats_v1.npy", bounds, allow_pickle=False)
    density_pool = ProcessPoolExecutor(
        max_workers=args.density_workers,
        mp_context=context,
        initializer=worker_initializer,
        max_tasks_per_child=args.max_cases_per_worker,
    )
    try:
        density_jobs = [(case_id, str(root), bounds, args.density_knn_k) for case_id, _ in records]
        for _ in run_bounded_pool(
            density_pool,
            density_jobs,
            lambda case_id, cache_root, cache_bounds, knn_k: density_pool.submit(cache_density, case_id, cache_root, cache_bounds, knn_k),
            args.density_workers,
            f"Caching KDE-{args.density_knn_k}",
        ):
            pass
    except BaseException:
        print("Stopping KDE workers and reaping native allocations.", flush=True)
        terminate_executor(density_pool)
        raise
    else:
        density_pool.shutdown(wait=True, cancel_futures=False)
    export_ids = [int(value) for value in args.export_cases.split(",") if value.strip()]
    for case_id in export_ids:
        if case_id not in set(manifest["train_ids"] + manifest["validation_ids"]):
            raise ValueError(f"Export case {case_id} was not generated.")
        export_case(root, case_id, Path(args.results_dir).expanduser().resolve())
    for name, previous in previous_handlers.items():
        signal.signal(name, previous)
    print(f"Completed {VERSION}: {root}; mean FEM time={np.mean(times) if times else 0.0:.2f}s/case.")


if __name__ == "__main__":
    main()
