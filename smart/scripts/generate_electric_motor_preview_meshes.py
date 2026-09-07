#!/usr/bin/env python3
"""Generate two-region, curvature-adaptive 3D electric-motor preview meshes."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


REGION_IDS = {"solid": 1, "coolant": 2}


@dataclass(frozen=True)
class MotorParameters:
    case_id: int
    stack_length: float
    shaft_radius: float
    rotor_base_radius: float
    pole_height: float
    pole_arc_fraction: float
    air_gap: float
    stator_outer_radius: float
    tooth_tip_depth: float
    slot_width: float
    slot_depth: float

    @property
    def rotor_tip_radius(self) -> float:
        return self.rotor_base_radius + self.pole_height

    @property
    def stator_bore_radius(self) -> float:
        return self.rotor_tip_radius + self.air_gap


def sample_parameters(case_id: int) -> MotorParameters:
    """Sample a deterministic valid 12-slot/8-pole motor geometry."""
    rng = np.random.default_rng(20260905 + case_id)
    rotor_base_radius = float(rng.uniform(0.068, 0.082))
    pole_height = float(rng.uniform(0.007, 0.012))
    air_gap = float(rng.uniform(0.0014, 0.0024))
    stator_bore = rotor_base_radius + pole_height + air_gap
    radial_stator_thickness = float(rng.uniform(0.044, 0.058))
    tooth_tip_depth = float(rng.uniform(0.0045, 0.0080))
    slot_depth = float(rng.uniform(0.024, 0.034))
    parameters = MotorParameters(
        case_id=case_id,
        stack_length=float(rng.uniform(0.085, 0.130)),
        shaft_radius=float(rng.uniform(0.020, 0.027)),
        rotor_base_radius=rotor_base_radius,
        pole_height=pole_height,
        pole_arc_fraction=float(rng.uniform(0.50, 0.70)),
        air_gap=air_gap,
        stator_outer_radius=stator_bore + radial_stator_thickness,
        tooth_tip_depth=tooth_tip_depth,
        slot_width=float(rng.uniform(0.009, 0.014)),
        slot_depth=slot_depth,
    )
    if parameters.slot_depth + parameters.tooth_tip_depth >= radial_stator_thickness - 0.006:
        raise RuntimeError("Invalid stator geometry: insufficient back-iron thickness.")
    if parameters.shaft_radius >= parameters.rotor_base_radius - 0.018:
        raise RuntimeError("Invalid rotor geometry: insufficient rotor web thickness.")
    return parameters


def annulus(gmsh, outer_radius: float, inner_radius: float, length: float) -> int:
    occ = gmsh.model.occ
    outer = occ.addCylinder(0.0, 0.0, -0.5 * length, 0.0, 0.0, length, outer_radius)
    inner = occ.addCylinder(0.0, 0.0, -0.5 * length, 0.0, 0.0, length, inner_radius)
    result, _ = occ.cut([(3, outer)], [(3, inner)], removeObject=True, removeTool=True)
    volumes = [tag for dim, tag in result if dim == 3]
    if len(volumes) != 1:
        raise RuntimeError("Annulus construction did not produce one volume.")
    return volumes[0]


def radial_prism(
    gmsh,
    inner_radius: float,
    outer_radius: float,
    half_angle: float,
    length: float,
    angle: float,
) -> int:
    """Create a true curved-sector prism."""
    occ = gmsh.model.occ
    z0 = -0.5 * length
    corners = []
    for radius, offset in (
        (inner_radius, -half_angle),
        (outer_radius, -half_angle),
        (outer_radius, half_angle),
        (inner_radius, half_angle),
    ):
        theta = angle + offset
        corners.append(occ.addPoint(radius * np.cos(theta), radius * np.sin(theta), z0))
    center = occ.addPoint(0.0, 0.0, z0)
    edges = [
        occ.addLine(corners[0], corners[1]),
        occ.addCircleArc(corners[1], center, corners[2]),
        occ.addLine(corners[2], corners[3]),
        occ.addCircleArc(corners[3], center, corners[0]),
    ]
    face = occ.addPlaneSurface([occ.addCurveLoop(edges)])
    volumes = [tag for dim, tag in occ.extrude([(2, face)], 0.0, 0.0, length) if dim == 3]
    if len(volumes) != 1:
        raise RuntimeError("Pole-sector construction did not produce one volume.")
    return volumes[0]


def slot_box(gmsh, inner_radius: float, depth: float, width: float, length: float, angle: float) -> int:
    occ = gmsh.model.occ
    tag = occ.addBox(inner_radius, -0.5 * width, -0.501 * length, depth, width, 1.002 * length)
    occ.rotate([(3, tag)], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, angle)
    return tag


def add_motor_geometry(gmsh, p: MotorParameters) -> dict[str, list[int]]:
    """Build active solid and surrounding coolant as two conforming regions."""
    occ = gmsh.model.occ
    stator = annulus(gmsh, p.stator_outer_radius, p.stator_bore_radius, p.stack_length)
    slot_inner = p.stator_bore_radius + p.tooth_tip_depth
    slot_tools = [
        (
            3,
            slot_box(
                gmsh,
                slot_inner,
                p.slot_depth,
                p.slot_width,
                p.stack_length,
                index * 2.0 * np.pi / 12,
            ),
        )
        for index in range(12)
    ]
    cut_stator, _ = occ.cut([(3, stator)], slot_tools, removeObject=True, removeTool=True)
    stator_volumes = [tag for dim, tag in cut_stator if dim == 3]
    if len(stator_volumes) != 1:
        raise RuntimeError("Stator slots split the stator unexpectedly.")

    rotor_base = annulus(gmsh, p.rotor_base_radius, p.shaft_radius, p.stack_length)
    half_angle = np.pi / 8.0 * p.pole_arc_fraction
    pole_volumes = [
        (
            3,
            radial_prism(
                gmsh,
                p.rotor_base_radius - 0.003,
                p.rotor_tip_radius,
                half_angle,
                p.stack_length,
                index * 2.0 * np.pi / 8,
            ),
        )
        for index in range(8)
    ]
    fused_rotor, _ = occ.fuse([(3, rotor_base)], pole_volumes, removeObject=True, removeTool=True)
    rotor_volumes = [tag for dim, tag in fused_rotor if dim == 3]
    if len(rotor_volumes) != 1:
        raise RuntimeError("Rotor pole fusion did not produce one connected volume.")

    enclosure_radius = p.stator_outer_radius + 0.016
    enclosure_length = p.stack_length + 0.030
    enclosure = occ.addCylinder(
        0.0,
        0.0,
        -0.5 * enclosure_length,
        0.0,
        0.0,
        enclosure_length,
        enclosure_radius,
    )
    solid_tools = [(3, stator_volumes[0]), (3, rotor_volumes[0])]
    cut_coolant, _ = occ.cut(
        [(3, enclosure)],
        solid_tools,
        removeObject=True,
        removeTool=False,
    )
    coolant_volumes = [tag for dim, tag in cut_coolant if dim == 3]
    if len(coolant_volumes) != 1:
        raise RuntimeError("Coolant construction did not produce one connected volume.")
    occ.synchronize()
    return {"solid": stator_volumes + rotor_volumes, "coolant": coolant_volumes}


def entities_by_region(gmsh, volumes_by_region: dict[str, list[int]], dimension: int) -> dict[int, int]:
    result: dict[int, int] = {}
    for region, volumes in volumes_by_region.items():
        if dimension == 3:
            entities = [(3, volume) for volume in volumes]
        else:
            entities = []
            for volume in volumes:
                entities.extend(gmsh.model.getBoundary([(3, volume)], oriented=False, recursive=False))
        for dim, tag in entities:
            if dim == dimension:
                result.setdefault(tag, REGION_IDS[region])
    return result


def configure_mesh(gmsh, p: MotorParameters, volumes_by_region: dict[str, list[int]], scale: float) -> None:
    """Refine curved edges and the nonuniform air-gap band, then grade inward."""
    all_curves: set[int] = set()
    for volume in volumes_by_region["solid"]:
        surfaces = gmsh.model.getBoundary([(3, volume)], oriented=False, recursive=False)
        for surface in surfaces:
            curves = gmsh.model.getBoundary([surface], oriented=False, recursive=False)
            all_curves.update(tag for dim, tag in curves if dim == 1)

    h_min = scale * max(0.00105, 0.68 * p.air_gap)
    h_max = scale * 0.0105

    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "CurvesList", sorted(all_curves))
    edge_threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(edge_threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(edge_threshold, "SizeMin", h_min)
    gmsh.model.mesh.field.setNumber(edge_threshold, "SizeMax", h_max)
    gmsh.model.mesh.field.setNumber(edge_threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(edge_threshold, "DistMax", 0.010 * scale)

    radial = gmsh.model.mesh.field.add("MathEval")
    gap_radius = 0.5 * (p.rotor_tip_radius + p.stator_bore_radius)
    radial_distance = f"Sqrt((Sqrt(x*x+y*y)-{gap_radius:.12g})^2)"
    radial_formula = (
        f"{h_min:.12g}+({h_max:.12g}-{h_min:.12g})*"
        f"Min(1,{radial_distance}/{(0.008 * scale):.12g})"
    )
    gmsh.model.mesh.field.setString(radial, "F", radial_formula)

    combined = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(combined, "FieldsList", [edge_threshold, radial])
    gmsh.model.mesh.field.setAsBackgroundMesh(combined)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.RandomFactor", 1.0e-9)


def node_table(gmsh) -> tuple[np.ndarray, np.ndarray]:
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags, dtype=np.int64)
    points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    order = np.argsort(tags)
    return tags[order], points[order]


def compact_points(
    node_tags: np.ndarray,
    node_points: np.ndarray,
    connectivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    used_tags = np.unique(connectivity)
    source_indices = np.searchsorted(node_tags, used_tags)
    if np.any(source_indices >= len(node_tags)) or np.any(node_tags[source_indices] != used_tags):
        raise RuntimeError("Mesh connectivity refers to unknown Gmsh nodes.")
    return node_points[source_indices], np.searchsorted(used_tags, connectivity), used_tags


def extract_surface_mesh(gmsh, face_regions: dict[int, int]) -> pv.PolyData:
    node_tags, node_points = node_table(gmsh)
    triangles: list[np.ndarray] = []
    region_blocks: list[np.ndarray] = []
    for face, region_id in face_regions.items():
        types, _, blocks = gmsh.model.mesh.getElements(2, face)
        for element_type, nodes in zip(types, blocks):
            if element_type != 2:
                continue
            block = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
            triangles.append(block)
            region_blocks.append(np.full(len(block), region_id, dtype=np.uint8))
    if not triangles:
        raise RuntimeError("Gmsh produced no triangular surface elements.")
    triangles_array = np.vstack(triangles)
    points, compact, used_tags = compact_points(node_tags, node_points, triangles_array)
    cells = np.column_stack((np.full(len(compact), 3), compact)).ravel()
    mesh = pv.PolyData(points, cells)
    xyz = points[compact]
    areas = 0.5 * np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    if not np.isfinite(areas).all() or np.any(areas <= 1.0e-14):
        raise RuntimeError("Surface mesh contains non-finite or degenerate triangles.")
    mesh.cell_data["region_id"] = np.concatenate(region_blocks)
    mesh.cell_data["triangle_area_m2"] = areas.astype(np.float32)
    mesh.cell_data["local_length_scale_m"] = np.sqrt(areas).astype(np.float32)
    mesh.point_data["gmsh_node_tag"] = used_tags
    return mesh


def extract_volume_mesh(gmsh, volume_regions: dict[int, int]) -> pv.UnstructuredGrid:
    node_tags, node_points = node_table(gmsh)
    tetrahedra: list[np.ndarray] = []
    region_blocks: list[np.ndarray] = []
    for volume, region_id in volume_regions.items():
        types, _, blocks = gmsh.model.mesh.getElements(3, volume)
        for element_type, nodes in zip(types, blocks):
            if element_type != 4:
                continue
            block = np.asarray(nodes, dtype=np.int64).reshape(-1, 4)
            tetrahedra.append(block)
            region_blocks.append(np.full(len(block), region_id, dtype=np.uint8))
    if not tetrahedra:
        raise RuntimeError("Gmsh produced no linear tetrahedra.")
    tetra_array = np.vstack(tetrahedra)
    points, compact, used_tags = compact_points(node_tags, node_points, tetra_array)
    cells = np.column_stack((np.full(len(compact), 4), compact)).ravel()
    cell_types = np.full(len(compact), pv.CellType.TETRA, dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    xyz = points[compact]
    signed_six_volume = np.einsum(
        "ij,ij->i",
        xyz[:, 1] - xyz[:, 0],
        np.cross(xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0]),
    )
    volumes = np.abs(signed_six_volume) / 6.0
    if not np.isfinite(volumes).all() or np.any(volumes <= 1.0e-18):
        raise RuntimeError("Volume mesh contains non-finite or degenerate tetrahedra.")
    mesh.cell_data["region_id"] = np.concatenate(region_blocks)
    mesh.cell_data["tetra_volume_m3"] = volumes.astype(np.float32)
    mesh.point_data["gmsh_node_tag"] = used_tags
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
    volumes = np.abs(six_volume) / 6.0
    return gradients, volumes


def thermal_coefficients(
    temperature: np.ndarray,
    centroids: np.ndarray,
    region_ids: np.ndarray,
    p: MotorParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the fixed nonlinear material and loss laws at cell centers."""
    ambient = 293.15
    temperature_rise = np.clip(temperature - ambient, 0.0, 160.0)
    solid = region_ids == REGION_IDS["solid"]
    conductivity = np.where(
        solid,
        24.0 / (1.0 + 0.0022 * temperature_rise),
        1.4,
    )

    x, y = centroids[:, 0], centroids[:, 1]
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    stator = solid & (radius >= p.stator_bore_radius)
    rotor = solid & ~stator
    source = np.zeros(len(region_ids), dtype=np.float64)
    source[stator] = 3.0e5 * (0.72 + 0.28 * np.cos(6.0 * theta[stator]) ** 2)
    source[rotor] = 1.15e5 * (0.76 + 0.24 * np.cos(4.0 * theta[rotor]) ** 2)
    source *= 1.0 + 0.0037 * temperature_rise
    return conductivity, source


def assemble_thermal_system(
    tetrahedra: np.ndarray,
    gradients: np.ndarray,
    volumes: np.ndarray,
    conductivity: np.ndarray,
    source: np.ndarray,
    point_count: int,
):
    from scipy.sparse import coo_matrix

    gradient_products = np.einsum("nik,njk->nij", gradients, gradients)
    local_stiffness = conductivity[:, None, None] * volumes[:, None, None] * gradient_products
    rows = np.repeat(tetrahedra, 4, axis=1).ravel()
    columns = np.tile(tetrahedra, (1, 4)).ravel()
    matrix = coo_matrix((local_stiffness.ravel(), (rows, columns)), shape=(point_count, point_count)).tocsr()
    rhs = np.zeros(point_count, dtype=np.float64)
    cell_load = source * volumes / 4.0
    for local_node in range(4):
        np.add.at(rhs, tetrahedra[:, local_node], cell_load)
    return matrix, rhs


def solve_nonlinear_thermal(
    volume: pv.UnstructuredGrid,
    surface: pv.PolyData,
    p: MotorParameters,
) -> dict[str, float | int]:
    """Solve steady nonlinear conduction with a fixed-temperature coolant jacket."""
    from scipy.sparse.linalg import LinearOperator, cg

    ambient = 293.15
    tetrahedra = np.asarray(volume.cells_dict[pv.CellType.TETRA], dtype=np.int64)
    points = np.asarray(volume.points, dtype=np.float64)
    region_ids = np.asarray(volume.cell_data["region_id"], dtype=np.uint8)
    gradients, volumes = tetra_geometry(points, tetrahedra)
    centroids = points[tetrahedra].mean(axis=1)

    enclosure_radius = p.stator_outer_radius + 0.016
    enclosure_half_length = 0.5 * (p.stack_length + 0.030)
    radius = np.linalg.norm(points[:, :2], axis=1)
    tolerance = 2.0e-7
    fixed = (np.abs(radius - enclosure_radius) < tolerance) | (
        np.abs(np.abs(points[:, 2]) - enclosure_half_length) < tolerance
    )
    if np.count_nonzero(fixed) < 100:
        raise RuntimeError(f"Case {p.case_id} has too few fixed-temperature boundary nodes.")
    free = ~fixed

    temperature = np.full(len(points), ambient, dtype=np.float64)
    nonlinear_change = np.inf
    matrix = rhs = conductivity = source = None
    iteration = 0
    for iteration in range(1, 31):
        cell_temperature = temperature[tetrahedra].mean(axis=1)
        conductivity, source = thermal_coefficients(cell_temperature, centroids, region_ids, p)
        matrix, rhs = assemble_thermal_system(
            tetrahedra,
            gradients,
            volumes,
            conductivity,
            source,
            len(points),
        )
        free_matrix = matrix[free][:, free]
        free_rhs = rhs[free] - matrix[free][:, fixed] @ np.full(np.count_nonzero(fixed), ambient)
        diagonal = free_matrix.diagonal()
        if np.any(diagonal <= 0.0):
            raise RuntimeError(f"Case {p.case_id} produced a non-positive thermal matrix diagonal.")
        preconditioner = LinearOperator(free_matrix.shape, matvec=lambda value: value / diagonal)
        candidate_free, info = cg(
            free_matrix,
            free_rhs,
            x0=temperature[free],
            rtol=2.0e-9,
            atol=0.0,
            maxiter=1500,
            M=preconditioner,
        )
        if info != 0:
            raise RuntimeError(f"Case {p.case_id} thermal CG failed with info={info}.")
        candidate = np.full(len(points), ambient, dtype=np.float64)
        candidate[free] = candidate_free
        updated = 0.72 * candidate + 0.28 * temperature
        nonlinear_change = float(np.max(np.abs(updated - temperature)))
        temperature = updated
        temperature[fixed] = ambient
        if nonlinear_change < 2.0e-5:
            break
    else:
        raise RuntimeError(f"Case {p.case_id} nonlinear thermal solve did not converge.")

    cell_temperature = temperature[tetrahedra].mean(axis=1)
    conductivity, source = thermal_coefficients(cell_temperature, centroids, region_ids, p)
    matrix, rhs = assemble_thermal_system(
        tetrahedra,
        gradients,
        volumes,
        conductivity,
        source,
        len(points),
    )
    residual = matrix @ temperature - rhs
    residual_relative = float(np.linalg.norm(residual[free]) / max(np.linalg.norm(rhs[free]), 1.0e-14))
    generated_heat = float(np.dot(source, volumes))
    removed_heat = float(-residual[fixed].sum())
    energy_balance_relative = abs(removed_heat - generated_heat) / max(generated_heat, 1.0e-14)

    cell_gradient = np.einsum("ni,nij->nj", temperature[tetrahedra], gradients)
    cell_flux = -conductivity[:, None] * cell_gradient
    point_flux = np.zeros((len(points), 3), dtype=np.float64)
    point_weights = np.zeros(len(points), dtype=np.float64)
    weighted_flux = cell_flux * volumes[:, None]
    for local_node in range(4):
        np.add.at(point_flux, tetrahedra[:, local_node], weighted_flux)
        np.add.at(point_weights, tetrahedra[:, local_node], volumes)
    point_flux /= point_weights[:, None]

    volume.point_data["temperature_K"] = temperature.astype(np.float32)
    volume.point_data["heat_flux_W_m2"] = point_flux.astype(np.float32)
    volume.point_data["heat_flux_magnitude_W_m2"] = np.linalg.norm(point_flux, axis=1).astype(np.float32)
    volume.point_data["fixed_temperature_boundary"] = fixed.astype(np.uint8)
    volume.cell_data["temperature_K"] = cell_temperature.astype(np.float32)
    volume.cell_data["conductivity_W_mK"] = conductivity.astype(np.float32)
    volume.cell_data["heat_source_W_m3"] = source.astype(np.float32)
    volume.cell_data["heat_flux_W_m2"] = cell_flux.astype(np.float32)
    volume.cell_data["heat_flux_magnitude_W_m2"] = np.linalg.norm(cell_flux, axis=1).astype(np.float32)

    volume_tags = np.asarray(volume.point_data["gmsh_node_tag"], dtype=np.int64)
    surface_tags = np.asarray(surface.point_data["gmsh_node_tag"], dtype=np.int64)
    indices = np.searchsorted(volume_tags, surface_tags)
    if np.any(indices >= len(volume_tags)) or np.any(volume_tags[indices] != surface_tags):
        raise RuntimeError(f"Case {p.case_id} could not map the volume solution to the surface mesh.")
    surface.point_data["temperature_K"] = temperature[indices].astype(np.float32)
    surface.point_data["heat_flux_W_m2"] = point_flux[indices].astype(np.float32)
    surface.point_data["heat_flux_magnitude_W_m2"] = np.linalg.norm(point_flux[indices], axis=1).astype(np.float32)
    surface.point_data["fixed_temperature_boundary"] = fixed[indices].astype(np.uint8)

    if residual_relative > 2.0e-6 or energy_balance_relative > 2.0e-6:
        raise RuntimeError(
            f"Case {p.case_id} failed thermal validation: residual={residual_relative:.3e}, "
            f"energy_balance={energy_balance_relative:.3e}"
        )
    if float(temperature.max()) <= ambient + 0.1 or float(temperature.max()) > 430.0:
        raise RuntimeError(f"Case {p.case_id} produced an implausible temperature range.")
    return {
        "nonlinear_iterations": iteration,
        "nonlinear_final_change_K": nonlinear_change,
        "linear_residual_relative": residual_relative,
        "energy_balance_relative": energy_balance_relative,
        "generated_heat_W": generated_heat,
        "temperature_min_K": float(temperature.min()),
        "temperature_max_K": float(temperature.max()),
        "heat_flux_max_W_m2": float(np.linalg.norm(point_flux, axis=1).max()),
    }


def generate_case(
    parameters: MotorParameters,
    output_dir: Path,
    gmsh_threads: int,
    max_mesh_entities: int,
    mesh_scale: float,
) -> dict[str, object]:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", gmsh_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", gmsh_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", gmsh_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", gmsh_threads)
        gmsh.model.add(f"electric_motor_preview_{parameters.case_id:03d}")
        volumes_by_region = add_motor_geometry(gmsh, parameters)
        face_regions = entities_by_region(gmsh, volumes_by_region, 2)
        volume_regions = entities_by_region(gmsh, volumes_by_region, 3)
        configure_mesh(gmsh, parameters, volumes_by_region, mesh_scale)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        surface = extract_surface_mesh(gmsh, face_regions)
        volume = extract_volume_mesh(gmsh, volume_regions)
    finally:
        gmsh.finalize()

    counts = {
        "surface_points": int(surface.n_points),
        "surface_triangles": int(surface.n_cells),
        "volume_points": int(volume.n_points),
        "volume_tetrahedra": int(volume.n_cells),
    }
    over_budget = {name: count for name, count in counts.items() if count > max_mesh_entities}
    if over_budget:
        raise RuntimeError(f"Case {parameters.case_id} exceeds mesh cap {max_mesh_entities:,}: {over_budget}")

    surface_regions = set(np.unique(surface.cell_data["region_id"]).tolist())
    volume_regions_present = set(np.unique(volume.cell_data["region_id"]).tolist())
    if surface_regions != {1, 2} or volume_regions_present != {1, 2}:
        raise RuntimeError(
            f"Case {parameters.case_id} does not contain exactly two regions: "
            f"surface={surface_regions}, volume={volume_regions_present}"
        )

    surface_areas = np.asarray(surface.cell_data["triangle_area_m2"], dtype=np.float64)
    volume_sizes = np.asarray(volume.cell_data["tetra_volume_m3"], dtype=np.float64)
    surface_ratio = float(np.percentile(surface_areas, 95) / np.percentile(surface_areas, 5))
    volume_ratio = float(np.percentile(volume_sizes, 95) / np.percentile(volume_sizes, 5))
    if surface_ratio < 8.0 or volume_ratio < 20.0:
        raise RuntimeError(
            f"Case {parameters.case_id} is insufficiently graded: "
            f"surface p95/p05={surface_ratio:.2f}, volume p95/p05={volume_ratio:.2f}, counts={counts}"
        )

    solution = solve_nonlinear_thermal(volume, surface, parameters)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"electric_motor_case_{parameters.case_id:03d}"
    surface_path = output_dir / f"{stem}_surface.vtp"
    volume_path = output_dir / f"{stem}_volume.vtu"
    surface.save(surface_path, binary=True)
    volume.save(volume_path, binary=True)
    return {
        "case_id": parameters.case_id,
        "surface_vtp": str(surface_path),
        "volume_vtu": str(volume_path),
        **counts,
        "surface_area_p95_over_p05": surface_ratio,
        "tetra_volume_p95_over_p05": volume_ratio,
        "surface_triangles_by_region": {
            str(region): int(np.count_nonzero(surface.cell_data["region_id"] == region)) for region in (1, 2)
        },
        "volume_tetrahedra_by_region": {
            str(region): int(np.count_nonzero(volume.cell_data["region_id"] == region)) for region in (1, 2)
        },
        "solution": solution,
        "parameters_si": asdict(parameters),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/parsa/smart_parsa/results/electric_motor_two_region_preview_meshes"),
    )
    parser.add_argument("--case-ids", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gmsh-threads", type=int, default=8)
    parser.add_argument("--max-mesh-entities", type=int, default=1_000_000)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    args = parser.parse_args()

    case_ids = [int(value.strip()) for value in args.case_ids.split(",") if value.strip()]
    if not case_ids:
        raise ValueError("At least one case ID is required.")
    if args.workers < 1 or args.gmsh_threads < 1:
        raise ValueError("workers and gmsh-threads must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (sample_parameters(case_id), args.output_dir, args.gmsh_threads, args.max_mesh_entities, args.mesh_scale)
        for case_id in case_ids
    ]
    records: list[dict[str, object]] = []
    if len(jobs) == 1 or args.workers == 1:
        records = [generate_case(*job) for job in jobs]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)), mp_context=context) as executor:
            futures = {executor.submit(generate_case, *job): job[0].case_id for job in jobs}
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                print(
                    f"case {record['case_id']:03d}: "
                    f"surface={record['surface_triangles']:,} triangles, "
                    f"volume={record['volume_tetrahedra']:,} tetrahedra",
                    flush=True,
                )

    records.sort(key=lambda record: int(record["case_id"]))
    summary = {
        "description": "Two-region curvature-adaptive motor meshes with nonlinear steady thermal solutions.",
        "region_ids": REGION_IDS,
        "physics": {
            "pde": "-div(k(T, region) grad(T)) = q(T, geometry)",
            "outer_boundary_condition": "T = 293.15 K on the outer coolant jacket",
            "interface_condition": "continuous temperature and normal heat flux",
            "solid_conductivity_W_mK": "24 / (1 + 0.0022 * max(T - 293.15, 0))",
            "coolant_effective_conductivity_W_mK": 1.4,
            "stator_reference_heat_source_W_m3": 3.0e5,
            "rotor_reference_heat_source_W_m3": 1.15e5,
            "heat_source_temperature_coefficient_per_K": 0.0037,
        },
        "mesh_entity_cap_per_output": args.max_mesh_entities,
        "workers": min(args.workers, len(jobs)),
        "gmsh_threads_per_worker": args.gmsh_threads,
        "cases": records,
    }
    summary_path = args.output_dir / "electric_motor_two_region_preview_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
