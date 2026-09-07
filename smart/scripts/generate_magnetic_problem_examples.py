#!/usr/bin/env python3
"""Generate four compact two-region magnetic benchmark examples with fields."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


MU0 = 4.0e-7 * np.pi
REGION_IDS = {"material": 1, "air": 2}


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    kind: str
    field_direction: tuple[float, float, float]
    outer_min: tuple[float, float, float]
    outer_max: tuple[float, float, float]
    mesh_min: float
    mesh_max: float


SPECS = {
    "flux_concentrator": ProblemSpec(
        "flux_concentrator",
        "nonlinear_magnetostatic",
        (1.0, 0.0, 0.0),
        (-0.145, -0.105, -0.080),
        (0.145, 0.105, 0.080),
        0.0030,
        0.018,
    ),
    "c_core": ProblemSpec(
        "c_core",
        "current_driven_magnetostatic",
        (0.0, 1.0, 0.0),
        (-0.135, -0.115, -0.075),
        (0.135, 0.115, 0.075),
        0.0032,
        0.018,
    ),
    "magnetic_shield": ProblemSpec(
        "magnetic_shield",
        "nonlinear_magnetostatic",
        (1.0, 0.0, 0.0),
        (-0.130, -0.110, -0.095),
        (0.130, 0.110, 0.095),
        0.0030,
        0.017,
    ),
    "eddy_current_plate": ProblemSpec(
        "eddy_current_plate",
        "time_harmonic_eddy_current",
        (0.0, 0.0, 1.0),
        (-0.120, -0.120, -0.055),
        (0.120, 0.120, 0.055),
        0.0025,
        0.017,
    ),
}


def add_box(gmsh, lower: tuple[float, float, float], upper: tuple[float, float, float]) -> int:
    return gmsh.model.occ.addBox(
        lower[0],
        lower[1],
        lower[2],
        upper[0] - lower[0],
        upper[1] - lower[1],
        upper[2] - lower[2],
    )


def build_flux_concentrator(gmsh) -> list[int]:
    occ = gmsh.model.occ
    gap = 0.012
    left = occ.addCone(-0.110, 0.0, 0.0, 0.104, 0.0, 0.0, 0.038, 0.009)
    right = occ.addCone(0.110, 0.0, 0.0, -0.104, 0.0, 0.0, 0.038, 0.009)
    if 2.0 * 0.006 != gap:
        raise RuntimeError("Flux-concentrator gap construction is inconsistent.")
    return [left, right]


def build_c_core(gmsh) -> list[int]:
    occ = gmsh.model.occ
    outer = occ.addBox(-0.080, -0.075, -0.024, 0.145, 0.150, 0.048)
    window = occ.addBox(-0.050, -0.045, -0.026, 0.088, 0.090, 0.052)
    opening = occ.addBox(0.032, -0.006, -0.027, 0.040, 0.012, 0.054)
    cut, _ = occ.cut([(3, outer)], [(3, window), (3, opening)], removeObject=True, removeTool=True)
    volumes = [tag for dim, tag in cut if dim == 3]
    if len(volumes) != 1:
        raise RuntimeError("C-core construction did not produce one connected volume.")
    return volumes


def build_magnetic_shield(gmsh) -> list[int]:
    occ = gmsh.model.occ
    outer = occ.addSphere(0.0, 0.0, 0.0, 0.075)
    inner = occ.addSphere(0.0, 0.0, 0.0, 0.061)
    occ.dilate([(3, outer)], 0.0, 0.0, 0.0, 1.0, 0.78, 0.68)
    occ.dilate([(3, inner)], 0.0, 0.0, 0.0, 1.0, 0.72, 0.60)
    cut, _ = occ.cut([(3, outer)], [(3, inner)], removeObject=True, removeTool=True)
    volumes = [tag for dim, tag in cut if dim == 3]
    if len(volumes) != 1:
        raise RuntimeError("Shield construction did not produce one shell volume.")
    return volumes


def build_eddy_current_plate(gmsh) -> list[int]:
    occ = gmsh.model.occ
    plate = occ.addCylinder(0.0, 0.0, -0.0045, 0.0, 0.0, 0.009, 0.076)
    holes = []
    for x, y, radius in ((-0.028, 0.022, 0.011), (0.026, 0.025, 0.009), (0.010, -0.032, 0.012)):
        holes.append((3, occ.addCylinder(x, y, -0.006, 0.0, 0.0, 0.012, radius)))
    cut, _ = occ.cut([(3, plate)], holes, removeObject=True, removeTool=True)
    volumes = [tag for dim, tag in cut if dim == 3]
    if len(volumes) != 1:
        raise RuntimeError("Eddy-current plate construction did not produce one volume.")
    return volumes


BUILDERS = {
    "flux_concentrator": build_flux_concentrator,
    "c_core": build_c_core,
    "magnetic_shield": build_magnetic_shield,
    "eddy_current_plate": build_eddy_current_plate,
}


def build_two_region_geometry(gmsh, spec: ProblemSpec) -> dict[str, list[int]]:
    occ = gmsh.model.occ
    material = BUILDERS[spec.name](gmsh)
    enclosure = add_box(gmsh, spec.outer_min, spec.outer_max)
    air_cut, _ = occ.cut(
        [(3, enclosure)],
        [(3, tag) for tag in material],
        removeObject=True,
        removeTool=False,
    )
    air = [tag for dim, tag in air_cut if dim == 3]
    if not air:
        raise RuntimeError(f"{spec.name}: air-region construction failed.")
    occ.synchronize()
    return {"material": material, "air": air}


def boundary_faces(gmsh, volumes: list[int]) -> list[int]:
    faces: set[int] = set()
    for volume in volumes:
        faces.update(
            tag
            for dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False, recursive=False)
            if dim == 2
        )
    return sorted(faces)


def configure_mesh(gmsh, spec: ProblemSpec, material_volumes: list[int]) -> None:
    material_faces = boundary_faces(gmsh, material_volumes)
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "FacesList", material_faces)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", spec.mesh_min)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", spec.mesh_max)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", 0.032)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 28)
    gmsh.option.setNumber("Mesh.MeshSizeMin", spec.mesh_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", spec.mesh_max)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)


def node_table(gmsh) -> tuple[np.ndarray, np.ndarray]:
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags, dtype=np.int64)
    points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    order = np.argsort(tags)
    return tags[order], points[order]


def compact_connectivity(
    node_tags: np.ndarray,
    node_points: np.ndarray,
    connectivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    used = np.unique(connectivity)
    source = np.searchsorted(node_tags, used)
    if np.any(source >= len(node_tags)) or np.any(node_tags[source] != used):
        raise RuntimeError("Gmsh connectivity references unknown nodes.")
    return node_points[source], np.searchsorted(used, connectivity), used


def extract_volume(gmsh, volumes_by_region: dict[str, list[int]]) -> pv.UnstructuredGrid:
    node_tags, node_points = node_table(gmsh)
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for region, volumes in volumes_by_region.items():
        for volume in volumes:
            types, _, node_blocks = gmsh.model.mesh.getElements(3, volume)
            for element_type, nodes in zip(types, node_blocks):
                if element_type != 4:
                    continue
                block = np.asarray(nodes, dtype=np.int64).reshape(-1, 4)
                blocks.append(block)
                labels.append(np.full(len(block), REGION_IDS[region], dtype=np.uint8))
    tetrahedra = np.vstack(blocks)
    points, compact, used = compact_connectivity(node_tags, node_points, tetrahedra)
    cells = np.column_stack((np.full(len(compact), 4), compact)).ravel()
    mesh = pv.UnstructuredGrid(cells, np.full(len(compact), pv.CellType.TETRA, np.uint8), points)
    mesh.cell_data["region_id"] = np.concatenate(labels)
    mesh.point_data["gmsh_node_tag"] = used
    xyz = points[compact]
    six_volume = np.einsum(
        "ij,ij->i",
        xyz[:, 1] - xyz[:, 0],
        np.cross(xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0]),
    )
    cell_volume = np.abs(six_volume) / 6.0
    if not np.isfinite(cell_volume).all() or np.any(cell_volume <= 1.0e-18):
        raise RuntimeError("Invalid tetrahedra in exported volume mesh.")
    mesh.cell_data["tetra_volume_m3"] = cell_volume.astype(np.float32)
    return mesh


def extract_material_surface(gmsh, material_volumes: list[int]) -> pv.PolyData:
    node_tags, node_points = node_table(gmsh)
    blocks: list[np.ndarray] = []
    for face in boundary_faces(gmsh, material_volumes):
        types, _, node_blocks = gmsh.model.mesh.getElements(2, face)
        for element_type, nodes in zip(types, node_blocks):
            if element_type == 2:
                blocks.append(np.asarray(nodes, dtype=np.int64).reshape(-1, 3))
    triangles = np.vstack(blocks)
    points, compact, used = compact_connectivity(node_tags, node_points, triangles)
    cells = np.column_stack((np.full(len(compact), 3), compact)).ravel()
    mesh = pv.PolyData(points, cells)
    xyz = points[compact]
    areas = 0.5 * np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    if not np.isfinite(areas).all() or np.any(areas <= 1.0e-14):
        raise RuntimeError("Invalid triangles in material-surface mesh.")
    mesh.cell_data["region_id"] = np.full(mesh.n_cells, REGION_IDS["material"], dtype=np.uint8)
    mesh.cell_data["triangle_area_m2"] = areas.astype(np.float32)
    mesh.point_data["gmsh_node_tag"] = used
    return mesh


def tetra_geometry(points: np.ndarray, tetrahedra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = points[tetrahedra]
    affine = np.concatenate((np.ones((*xyz.shape[:2], 1)), xyz), axis=2)
    inverse = np.linalg.inv(affine)
    gradients = np.transpose(inverse[:, 1:, :], (0, 2, 1))
    six_volume = np.einsum(
        "ij,ij->i",
        xyz[:, 1] - xyz[:, 0],
        np.cross(xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0]),
    )
    return gradients, np.abs(six_volume) / 6.0


def outer_boundary_nodes(points: np.ndarray, spec: ProblemSpec) -> np.ndarray:
    lower = np.asarray(spec.outer_min)
    upper = np.asarray(spec.outer_max)
    tolerance = 2.0e-7
    return np.any((np.abs(points - lower) < tolerance) | (np.abs(points - upper) < tolerance), axis=1)


def average_cell_vectors_to_points(
    tetrahedra: np.ndarray,
    volumes: np.ndarray,
    vectors: np.ndarray,
    point_count: int,
) -> np.ndarray:
    result = np.zeros((point_count, vectors.shape[1]), dtype=vectors.dtype)
    weights = np.zeros(point_count, dtype=np.float64)
    weighted = vectors * volumes[:, None]
    for local_node in range(4):
        np.add.at(result, tetrahedra[:, local_node], weighted)
        np.add.at(weights, tetrahedra[:, local_node], volumes)
    result /= weights[:, None]
    return result


def map_point_fields(volume: pv.UnstructuredGrid, surface: pv.PolyData, names: list[str]) -> None:
    volume_tags = np.asarray(volume.point_data["gmsh_node_tag"], dtype=np.int64)
    surface_tags = np.asarray(surface.point_data["gmsh_node_tag"], dtype=np.int64)
    indices = np.searchsorted(volume_tags, surface_tags)
    if np.any(indices >= len(volume_tags)) or np.any(volume_tags[indices] != surface_tags):
        raise RuntimeError("Could not map volume solution onto material surface.")
    for name in names:
        surface.point_data[name] = np.asarray(volume.point_data[name])[indices]


def solve_nonlinear_magnetostatic(
    volume: pv.UnstructuredGrid,
    surface: pv.PolyData,
    spec: ProblemSpec,
) -> dict[str, float | int]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    points = np.asarray(volume.points, dtype=np.float64)
    tetrahedra = np.asarray(volume.cells_dict[pv.CellType.TETRA], dtype=np.int64)
    regions = np.asarray(volume.cell_data["region_id"], dtype=np.uint8)
    gradients, volumes = tetra_geometry(points, tetrahedra)
    gradient_products = np.einsum("nik,njk->nij", gradients, gradients)
    rows = np.repeat(tetrahedra, 4, axis=1).ravel()
    columns = np.tile(tetrahedra, (1, 4)).ravel()
    fixed = outer_boundary_nodes(points, spec)
    free = ~fixed

    direction = np.asarray(spec.field_direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    applied_b = 0.075
    applied_h = applied_b / MU0
    boundary_phi = -applied_h * (points @ direction)
    phi = boundary_phi.copy()
    material = regions == REGION_IDS["material"]
    relative_mu = np.where(material, 900.0, 1.0)
    change = np.inf
    iteration = 0
    for iteration in range(1, 81):
        local = relative_mu[:, None, None] * volumes[:, None, None] * gradient_products
        matrix = coo_matrix((local.ravel(), (rows, columns)), shape=(len(points), len(points))).tocsr()
        rhs = -(matrix[free][:, fixed] @ boundary_phi[fixed])
        candidate = boundary_phi.copy()
        candidate[free] = spsolve(matrix[free][:, free], rhs)
        cell_h = -np.einsum("ni,nij->nj", candidate[tetrahedra], gradients)
        previous_b_magnitude = MU0 * relative_mu * np.linalg.norm(cell_h, axis=1)
        target_mu = np.ones(len(tetrahedra), dtype=np.float64)
        target_mu[material] = 1.0 + 999.0 / (1.0 + (previous_b_magnitude[material] / 1.45) ** 4)
        updated_mu = 0.22 * target_mu + 0.78 * relative_mu
        change = float(np.max(np.abs(updated_mu - relative_mu) / np.maximum(relative_mu, 1.0)))
        relative_mu = updated_mu
        phi = candidate
        if change < 5.0e-4:
            break
    else:
        raise RuntimeError(
            f"{spec.name}: nonlinear permeability iteration did not converge; final change={change:.3e}."
        )

    local = relative_mu[:, None, None] * volumes[:, None, None] * gradient_products
    matrix = coo_matrix((local.ravel(), (rows, columns)), shape=(len(points), len(points))).tocsr()
    rhs = -(matrix[free][:, fixed] @ boundary_phi[fixed])
    phi = boundary_phi.copy()
    phi[free] = spsolve(matrix[free][:, free], rhs)
    residual = matrix @ phi
    residual_relative = float(np.linalg.norm(residual[free]) / max(np.linalg.norm(rhs), 1.0e-14))
    cell_h = -np.einsum("ni,nij->nj", phi[tetrahedra], gradients)
    cell_b = MU0 * relative_mu[:, None] * cell_h
    point_h = average_cell_vectors_to_points(tetrahedra, volumes, cell_h, len(points))
    point_b = average_cell_vectors_to_points(tetrahedra, volumes, cell_b, len(points))

    volume.point_data["magnetic_scalar_potential_A"] = phi.astype(np.float32)
    volume.point_data["H_A_m"] = point_h.astype(np.float32)
    volume.point_data["H_magnitude_A_m"] = np.linalg.norm(point_h, axis=1).astype(np.float32)
    volume.point_data["B_T"] = point_b.astype(np.float32)
    volume.point_data["B_magnitude_T"] = np.linalg.norm(point_b, axis=1).astype(np.float32)
    volume.cell_data["relative_permeability"] = relative_mu.astype(np.float32)
    volume.cell_data["H_A_m"] = cell_h.astype(np.float32)
    volume.cell_data["B_T"] = cell_b.astype(np.float32)
    volume.cell_data["B_magnitude_T"] = np.linalg.norm(cell_b, axis=1).astype(np.float32)
    volume.cell_data["magnetic_energy_J_m3"] = (0.5 * np.einsum("ij,ij->i", cell_b, cell_h)).astype(np.float32)
    map_point_fields(
        volume,
        surface,
        ["magnetic_scalar_potential_A", "H_A_m", "H_magnitude_A_m", "B_T", "B_magnitude_T"],
    )
    return {
        "nonlinear_iterations": iteration,
        "nonlinear_mu_change": change,
        "residual_relative": residual_relative,
        "applied_flux_density_T": applied_b,
        "B_max_T": float(np.linalg.norm(point_b, axis=1).max()),
        "material_mu_min": float(relative_mu[material].min()),
        "material_mu_max": float(relative_mu[material].max()),
    }


def solve_eddy_current(
    volume: pv.UnstructuredGrid,
    surface: pv.PolyData,
    spec: ProblemSpec,
) -> dict[str, float | int]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    points = np.asarray(volume.points, dtype=np.float64)
    tetrahedra = np.asarray(volume.cells_dict[pv.CellType.TETRA], dtype=np.int64)
    regions = np.asarray(volume.cell_data["region_id"], dtype=np.uint8)
    gradients, volumes = tetra_geometry(points, tetrahedra)
    material = regions == REGION_IDS["material"]
    sigma = np.where(material, 5.8e7, 0.0)
    frequency = 50.0
    omega = 2.0 * np.pi * frequency

    eye = np.eye(3)
    curls = np.cross(gradients[:, :, None, :], eye[None, None, :, :]).reshape(-1, 12, 3)
    divergences = gradients[:, :, :, None] * eye[None, None, :, :]
    divergences = np.einsum("niab->nia", divergences).reshape(-1, 12)
    curl_curl = np.einsum("nik,njk->nij", curls, curls)
    div_div = divergences[:, :, None] * divergences[:, None, :]
    stiffness = volumes[:, None, None] * (curl_curl + div_div)

    mass4 = np.ones((4, 4), dtype=np.float64)
    np.fill_diagonal(mass4, 2.0)
    mass12 = np.kron(mass4, eye)[None, :, :] * (volumes[:, None, None] / 20.0)
    local = stiffness.astype(np.complex128)
    local += (1j * omega * MU0 * sigma)[:, None, None] * mass12

    dofs = (tetrahedra[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(-1, 12)
    rows = np.repeat(dofs, 12, axis=1).ravel()
    columns = np.tile(dofs, (1, 12)).ravel()
    matrix = coo_matrix((local.ravel(), (rows, columns)), shape=(3 * len(points), 3 * len(points))).tocsr()

    fixed_nodes = outer_boundary_nodes(points, spec)
    fixed = np.repeat(fixed_nodes, 3)
    free = ~fixed
    applied_b = np.array([0.0, 0.0, 0.075])
    boundary_a = 0.5 * np.cross(np.broadcast_to(applied_b, points.shape), points)
    vector_potential = boundary_a.astype(np.complex128).ravel()
    rhs = -(matrix[free][:, fixed] @ vector_potential[fixed])
    vector_potential[free] = spsolve(matrix[free][:, free], rhs)
    residual = matrix @ vector_potential
    residual_relative = float(np.linalg.norm(residual[free]) / max(np.linalg.norm(rhs), 1.0e-14))

    nodal_a = vector_potential.reshape(-1, 3)
    cell_b = np.cross(gradients, nodal_a[tetrahedra]).sum(axis=1)
    cell_e = -1j * omega * nodal_a[tetrahedra].mean(axis=1)
    cell_j = sigma[:, None] * cell_e
    cell_loss = 0.5 * sigma * np.einsum("ij,ij->i", cell_e.conj(), cell_e).real
    point_b = average_cell_vectors_to_points(tetrahedra, volumes, cell_b, len(points))
    point_j = average_cell_vectors_to_points(tetrahedra, volumes, cell_j, len(points))

    volume.point_data["A_real_T_m"] = nodal_a.real.astype(np.float32)
    volume.point_data["A_imag_T_m"] = nodal_a.imag.astype(np.float32)
    volume.point_data["B_real_T"] = point_b.real.astype(np.float32)
    volume.point_data["B_imag_T"] = point_b.imag.astype(np.float32)
    volume.point_data["B_magnitude_T"] = np.linalg.norm(point_b, axis=1).astype(np.float32)
    volume.point_data["J_real_A_m2"] = point_j.real.astype(np.float32)
    volume.point_data["J_imag_A_m2"] = point_j.imag.astype(np.float32)
    volume.point_data["J_magnitude_A_m2"] = np.linalg.norm(point_j, axis=1).astype(np.float32)
    volume.cell_data["conductivity_S_m"] = sigma.astype(np.float32)
    volume.cell_data["B_magnitude_T"] = np.linalg.norm(cell_b, axis=1).astype(np.float32)
    volume.cell_data["J_magnitude_A_m2"] = np.linalg.norm(cell_j, axis=1).astype(np.float32)
    volume.cell_data["joule_loss_W_m3"] = cell_loss.astype(np.float32)
    map_point_fields(
        volume,
        surface,
        [
            "A_real_T_m",
            "A_imag_T_m",
            "B_real_T",
            "B_imag_T",
            "B_magnitude_T",
            "J_real_A_m2",
            "J_imag_A_m2",
            "J_magnitude_A_m2",
        ],
    )
    return {
        "frequency_Hz": frequency,
        "applied_flux_density_T": float(np.linalg.norm(applied_b)),
        "residual_relative": residual_relative,
        "B_max_T": float(np.linalg.norm(point_b, axis=1).max()),
        "J_max_A_m2": float(np.linalg.norm(point_j, axis=1).max()),
        "joule_loss_W": float(np.dot(cell_loss, volumes)),
    }


def solve_current_driven_magnetostatic(
    volume: pv.UnstructuredGrid,
    surface: pv.PolyData,
    spec: ProblemSpec,
) -> dict[str, float | int]:
    """Solve a gauged vector-potential problem driven by a fixed current loop."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    points = np.asarray(volume.points, dtype=np.float64)
    tetrahedra = np.asarray(volume.cells_dict[pv.CellType.TETRA], dtype=np.int64)
    regions = np.asarray(volume.cell_data["region_id"], dtype=np.uint8)
    gradients, volumes = tetra_geometry(points, tetrahedra)
    material = regions == REGION_IDS["material"]
    relative_mu = np.where(material, 700.0, 1.0)

    eye = np.eye(3)
    curls = np.cross(gradients[:, :, None, :], eye[None, None, :, :]).reshape(-1, 12, 3)
    divergences = gradients[:, :, :, None] * eye[None, None, :, :]
    divergences = np.einsum("niab->nia", divergences).reshape(-1, 12)
    curl_curl = np.einsum("nik,njk->nij", curls, curls)
    div_div = divergences[:, :, None] * divergences[:, None, :]
    local = (volumes / relative_mu)[:, None, None] * (curl_curl + div_div)

    centroids = points[tetrahedra].mean(axis=1)
    x_relative = centroids[:, 0] + 0.066
    z_relative = centroids[:, 2]
    loop_radius = np.sqrt(x_relative * x_relative + z_relative * z_relative)
    current_support = (
        (~material)
        & (np.abs(centroids[:, 1]) < 0.052)
        & (loop_radius > 0.033)
        & (loop_radius < 0.046)
    )
    current_density = np.zeros((len(tetrahedra), 3), dtype=np.float64)
    safe_radius = np.maximum(loop_radius[current_support], 1.0e-12)
    current_density[current_support, 0] = 8.0e5 * z_relative[current_support] / safe_radius
    current_density[current_support, 2] = -8.0e5 * x_relative[current_support] / safe_radius
    if np.count_nonzero(current_support) < 20:
        raise RuntimeError("C-core impressed-current support is under-resolved.")

    dofs = (tetrahedra[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(-1, 12)
    rows = np.repeat(dofs, 12, axis=1).ravel()
    columns = np.tile(dofs, (1, 12)).ravel()
    matrix = coo_matrix((local.ravel(), (rows, columns)), shape=(3 * len(points), 3 * len(points))).tocsr()
    rhs = np.zeros(3 * len(points), dtype=np.float64)
    cell_load = MU0 * current_density * (volumes / 4.0)[:, None]
    for local_node in range(4):
        for component in range(3):
            np.add.at(rhs, 3 * tetrahedra[:, local_node] + component, cell_load[:, component])

    fixed_nodes = outer_boundary_nodes(points, spec)
    fixed = np.repeat(fixed_nodes, 3)
    free = ~fixed
    vector_potential = np.zeros(3 * len(points), dtype=np.float64)
    vector_potential[free] = spsolve(matrix[free][:, free], rhs[free])
    residual = matrix @ vector_potential - rhs
    residual_relative = float(np.linalg.norm(residual[free]) / max(np.linalg.norm(rhs[free]), 1.0e-14))

    nodal_a = vector_potential.reshape(-1, 3)
    cell_b = np.cross(gradients, nodal_a[tetrahedra]).sum(axis=1)
    cell_h = cell_b / (MU0 * relative_mu[:, None])
    point_b = average_cell_vectors_to_points(tetrahedra, volumes, cell_b, len(points))
    point_h = average_cell_vectors_to_points(tetrahedra, volumes, cell_h, len(points))

    volume.point_data["magnetic_vector_potential_T_m"] = nodal_a.astype(np.float32)
    volume.point_data["B_T"] = point_b.astype(np.float32)
    volume.point_data["B_magnitude_T"] = np.linalg.norm(point_b, axis=1).astype(np.float32)
    volume.point_data["H_A_m"] = point_h.astype(np.float32)
    volume.point_data["H_magnitude_A_m"] = np.linalg.norm(point_h, axis=1).astype(np.float32)
    volume.cell_data["relative_permeability"] = relative_mu.astype(np.float32)
    volume.cell_data["impressed_current_density_A_m2"] = current_density.astype(np.float32)
    volume.cell_data["B_T"] = cell_b.astype(np.float32)
    volume.cell_data["B_magnitude_T"] = np.linalg.norm(cell_b, axis=1).astype(np.float32)
    volume.cell_data["magnetic_energy_J_m3"] = (0.5 * np.einsum("ij,ij->i", cell_b, cell_h)).astype(np.float32)
    map_point_fields(
        volume,
        surface,
        ["magnetic_vector_potential_T_m", "B_T", "B_magnitude_T", "H_A_m", "H_magnitude_A_m"],
    )
    return {
        "impressed_current_density_A_m2": 8.0e5,
        "current_support_volume_m3": float(volumes[current_support].sum()),
        "residual_relative": residual_relative,
        "B_max_T": float(np.linalg.norm(point_b, axis=1).max()),
        "material_relative_permeability": 700.0,
    }


def generate_problem(
    spec: ProblemSpec,
    output_dir: Path,
    gmsh_threads: int,
    max_cells: int,
) -> dict[str, object]:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", gmsh_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", gmsh_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", gmsh_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", gmsh_threads)
        gmsh.model.add(spec.name)
        regions = build_two_region_geometry(gmsh, spec)
        configure_mesh(gmsh, spec, regions["material"])
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        volume = extract_volume(gmsh, regions)
        surface = extract_material_surface(gmsh, regions["material"])
    finally:
        gmsh.finalize()

    if volume.n_cells > max_cells or surface.n_cells > max_cells:
        raise RuntimeError(
            f"{spec.name}: mesh cap exceeded: volume={volume.n_cells:,}, surface={surface.n_cells:,}"
        )
    if set(np.unique(volume.cell_data["region_id"]).tolist()) != {1, 2}:
        raise RuntimeError(f"{spec.name}: expected exactly two volume region IDs.")

    if spec.kind == "time_harmonic_eddy_current":
        solution = solve_eddy_current(volume, surface, spec)
    elif spec.kind == "current_driven_magnetostatic":
        solution = solve_current_driven_magnetostatic(volume, surface, spec)
    else:
        solution = solve_nonlinear_magnetostatic(volume, surface, spec)
    if solution["residual_relative"] > 2.0e-6:
        raise RuntimeError(f"{spec.name}: field residual is too large: {solution['residual_relative']:.3e}")

    output_dir.mkdir(parents=True, exist_ok=True)
    surface_path = output_dir / f"{spec.name}_material_surface.vtp"
    volume_path = output_dir / f"{spec.name}_full_volume.vtu"
    surface.save(surface_path, binary=True)
    volume.save(volume_path, binary=True)
    return {
        "name": spec.name,
        "kind": spec.kind,
        "surface_vtp": str(surface_path),
        "volume_vtu": str(volume_path),
        "surface_points": int(surface.n_points),
        "surface_triangles": int(surface.n_cells),
        "volume_points": int(volume.n_points),
        "volume_tetrahedra": int(volume.n_cells),
        "solution": solution,
        "specification": asdict(spec),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/parsa/smart_parsa/results/magnetic_problem_examples"),
    )
    parser.add_argument("--problems", default=",".join(SPECS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gmsh-threads", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=1_000_000)
    args = parser.parse_args()

    names = [name.strip() for name in args.problems.split(",") if name.strip()]
    unknown = sorted(set(names) - set(SPECS))
    if unknown:
        raise ValueError(f"Unknown problems: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(names)), mp_context=context) as executor:
        futures = {
            executor.submit(generate_problem, SPECS[name], args.output_dir, args.gmsh_threads, args.max_cells): name
            for name in names
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"{record['name']}: {record['surface_triangles']:,} surface triangles, "
                f"{record['volume_tetrahedra']:,} tetrahedra, "
                f"Bmax={record['solution']['B_max_T']:.3g} T",
                flush=True,
            )
    records.sort(key=lambda record: names.index(str(record["name"])))
    summary = {
        "description": "Compact two-region magnetic geometry-to-field candidates.",
        "region_ids": REGION_IDS,
        "formulations": {
            "nonlinear_magnetostatic": (
                "Current-free scalar magnetic potential with a fixed uniform far-field flux density "
                "and a smooth saturating permeability law."
            ),
            "current_driven_magnetostatic": (
                "Gauged vector magnetic potential with a fixed analytic impressed-current loop "
                "surrounding the C-core back limb."
            ),
            "time_harmonic_eddy_current": (
                "Complex gauged vector magnetic potential with a fixed 50 Hz applied field and "
                "conductive-material eddy currents."
            ),
        },
        "fixed_across_examples": {
            "magnetostatic_applied_flux_density_T": 0.075,
            "eddy_current_frequency_Hz": 50.0,
            "eddy_current_conductivity_S_m": 5.8e7,
        },
        "problems": records,
    }
    summary_path = args.output_dir / "magnetic_problem_examples_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
