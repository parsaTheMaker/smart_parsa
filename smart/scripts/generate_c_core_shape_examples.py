#!/usr/bin/env python3
"""Generate eight solved C-core variants with a fixed straight winding limb."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

try:
    from generate_magnetic_problem_examples import (
        average_cell_vectors_to_points,
        extract_material_surface,
        extract_volume,
        map_point_fields,
        tetra_geometry,
    )
except ImportError:  # Support importing as smart.scripts.* in tests.
    from scripts.generate_magnetic_problem_examples import (
        average_cell_vectors_to_points,
        extract_material_surface,
        extract_volume,
        map_point_fields,
        tetra_geometry,
    )


MATERIAL_REGION = 1
CORE_THICKNESS = 0.0308
CORE_Z_MIN = -0.5 * CORE_THICKNESS
MU0 = 4.0e-7 * np.pi
RETURN_BODY_LEFT_X = -0.015
MIN_MAGNETIC_WEB = 0.012
APERTURE_FILLET_RADIUS = 0.008
STRAIGHT_ARM_END_X = 0.0
MIN_TETRA_MEAN_RATIO = 0.02
MIN_TETRA_P01_MEAN_RATIO = 0.25
MIN_TRIANGLE_P01_QUALITY = 0.74
MESH_MIN_SIZE = 0.00023
MESH_EDGE_SIZE = 0.00029
MESH_SURFACE_MAX_SIZE = 0.00115
MESH_MAX_SIZE = 0.00150
AIR_MESH_MAX_SIZE = 0.012
COIL_MESH_SIZE = 0.00038
CURVATURE_POINTS_PER_2PI = 128
EDGE_DISTANCE_SAMPLING = 240
COIL_DISTANCE_SAMPLING = 120
EDGE_REFINEMENT_DIST_MIN = 0.0004
EDGE_REFINEMENT_DIST_MAX = 0.014
COIL_REFINEMENT_DIST_MAX = 0.014
AIR_MIN = np.array((-0.150, -0.135, -0.088), dtype=np.float64)
AIR_MAX = np.array((0.180, 0.135, 0.088), dtype=np.float64)
COIL_AXIS_X = -0.0665
COIL_INNER_RADIUS = 0.022
COIL_OUTER_RADIUS = 0.031
COIL_CENTERLINE_RADIUS = 0.0265
COIL_HALF_LENGTH = 0.040
COIL_TURNS = 10
COIL_CURRENT_A = 2.0


@dataclass(frozen=True)
class CoreVariant:
    name: str
    profile: str
    center_x: float
    width: float
    height: float
    window_ratio_x: float
    window_ratio_y: float
    window_topology: str
    pole_style: str
    inner_gap: float
    tip_gap: float
    transition_angle_deg: float
    section_profile: str


VARIANTS = (
    CoreVariant("rectangle_unified_parallel", "rectangle", 0.040, 0.148, 0.148, 0.50, 0.49, "unified", "parallel", 0.016, 0.016, 68.0, "rectangular"),
    CoreVariant("rectangle_dual_focused", "rectangle", 0.038, 0.136, 0.164, 0.57, 0.55, "dual", "focused", 0.028, 0.006, 0.0, "circular"),
    CoreVariant("circle_unified_parallel", "circle", 0.038, 0.160, 0.160, 0.50, 0.50, "unified", "parallel", 0.008, 0.008, 65.0, "circular"),
    CoreVariant("circle_dual_focused", "circle", 0.035, 0.146, 0.146, 0.50, 0.59, "dual", "focused", 0.026, 0.004, 0.0, "rectangular"),
    CoreVariant("ellipse_wide_unified_parallel", "ellipse", 0.043, 0.184, 0.130, 0.53, 0.46, "unified", "parallel", 0.020, 0.020, 60.0, "rectangular"),
    CoreVariant("ellipse_wide_dual_focused", "ellipse", 0.039, 0.166, 0.144, 0.50, 0.54, "dual", "focused", 0.028, 0.010, 0.0, "circular"),
    CoreVariant("ellipse_tall_unified_parallel", "ellipse", 0.034, 0.132, 0.184, 0.46, 0.55, "unified", "parallel", 0.006, 0.006, 72.0, "circular"),
    CoreVariant("ellipse_tall_dual_focused", "ellipse", 0.031, 0.144, 0.170, 0.46, 0.64, "dual", "focused", 0.032, 0.014, 0.0, "rectangular"),
)


def validate_variant(variant: CoreVariant) -> dict[str, float | None]:
    """Reject geometries that violate manufacturable magnetic-web constraints."""
    inner_width = variant.width * variant.window_ratio_x
    inner_height = variant.height * variant.window_ratio_y
    radial_web_x = 0.5 * (variant.width - inner_width)
    radial_web_y = 0.5 * (variant.height - inner_height)
    inter_window_web = variant.center_x - 0.5 * inner_width - RETURN_BODY_LEFT_X
    if min(radial_web_x, radial_web_y) < MIN_MAGNETIC_WEB:
        raise ValueError(f"{variant.name}: return-body web is below {MIN_MAGNETIC_WEB:.3f} m.")
    if variant.section_profile not in {"rectangular", "circular"}:
        raise ValueError(f"{variant.name}: unknown section profile {variant.section_profile!r}.")
    if (
        variant.section_profile == "rectangular"
        and variant.window_topology == "dual"
        and inter_window_web < MIN_MAGNETIC_WEB
    ):
        raise ValueError(f"{variant.name}: inter-window web is only {inter_window_web:.4f} m.")
    if min(variant.inner_gap, variant.tip_gap) <= 0.0:
        raise ValueError(f"{variant.name}: pole gap must be positive.")
    if max(variant.inner_gap, variant.tip_gap) > 0.60 * inner_height:
        raise ValueError(f"{variant.name}: pole gap is too large for its window.")
    if variant.pole_style == "parallel" and not np.isclose(variant.inner_gap, variant.tip_gap):
        raise ValueError(f"{variant.name}: parallel poles require equal gap widths.")
    if variant.pole_style == "focused" and variant.tip_gap >= variant.inner_gap:
        raise ValueError(f"{variant.name}: focused poles must narrow toward the tip.")
    if variant.window_topology == "unified" and not 55.0 <= variant.transition_angle_deg <= 75.0:
        raise ValueError(f"{variant.name}: unified aperture transition angle is outside [55, 75] degrees.")
    right_edge = variant.center_x + 0.5 * variant.width
    slot_start = (
        gap_slot_start(variant)
        if variant.section_profile == "rectangular"
        else circular_core_dimensions(variant)["slot_start_m"]
    )
    taper_half_angle = np.rad2deg(
        np.arctan2(0.5 * abs(variant.inner_gap - variant.tip_gap), right_edge - slot_start)
    )
    if taper_half_angle > 20.0:
        raise ValueError(f"{variant.name}: pole taper half-angle is {taper_half_angle:.1f} degrees.")
    return {
        "minimum_radial_web_m": float(min(radial_web_x, radial_web_y)),
        "inter_window_web_m": (
            float(inter_window_web)
            if variant.window_topology == "dual" and variant.section_profile == "rectangular"
            else (0.018 if variant.window_topology == "dual" else None)
        ),
        "pole_taper_half_angle_deg": float(taper_half_angle),
    }


def validate_mesh(
    solid: pv.UnstructuredGrid,
    surface: pv.PolyData,
    mesh_size_scale: float = 1.0,
) -> dict[str, object]:
    """Apply topology, resolution, and element-quality gates to an exported mesh."""
    if set(solid.cells_dict) != {pv.CellType.TETRA}:
        raise RuntimeError("Solid mesh must contain first-order tetrahedra only.")
    if not surface.is_all_triangles:
        raise RuntimeError("Material surface must contain triangles only.")

    triangles = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    triangle_xyz = np.asarray(surface.points, dtype=np.float64)[triangles]
    triangle_edges = np.stack(
        (
            np.linalg.norm(triangle_xyz[:, 1] - triangle_xyz[:, 0], axis=1),
            np.linalg.norm(triangle_xyz[:, 2] - triangle_xyz[:, 1], axis=1),
            np.linalg.norm(triangle_xyz[:, 0] - triangle_xyz[:, 2], axis=1),
        ),
        axis=1,
    )
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(triangle_xyz[:, 1] - triangle_xyz[:, 0], triangle_xyz[:, 2] - triangle_xyz[:, 0]),
        axis=1,
    )
    triangle_quality = 4.0 * np.sqrt(3.0) * triangle_area / np.sum(triangle_edges**2, axis=1)

    surface_edges = np.sort(
        np.concatenate(
            (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
        ),
        axis=1,
    )
    _, edge_counts = np.unique(surface_edges, axis=0, return_counts=True)
    boundary_edges = int(np.count_nonzero(edge_counts == 1))
    non_manifold_edges = int(np.count_nonzero(edge_counts > 2))
    if boundary_edges or non_manifold_edges:
        raise RuntimeError(
            f"Material surface is not watertight (boundary={boundary_edges}, non-manifold={non_manifold_edges})."
        )

    tetrahedra = np.asarray(solid.cells_dict[pv.CellType.TETRA], dtype=np.int64)
    tetra_xyz = np.asarray(solid.points, dtype=np.float64)[tetrahedra]
    tetra_edges = np.stack(
        tuple(
            np.linalg.norm(tetra_xyz[:, j] - tetra_xyz[:, i], axis=1)
            for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        ),
        axis=1,
    )
    six_volume = np.abs(
        np.einsum(
            "ij,ij->i",
            tetra_xyz[:, 1] - tetra_xyz[:, 0],
            np.cross(tetra_xyz[:, 2] - tetra_xyz[:, 0], tetra_xyz[:, 3] - tetra_xyz[:, 0]),
        )
    )
    tetra_volume = six_volume / 6.0
    tetra_mean_ratio = 12.0 * np.power(3.0 * tetra_volume, 2.0 / 3.0) / np.sum(tetra_edges**2, axis=1)
    tetra_p01 = float(np.percentile(tetra_mean_ratio, 1.0))
    triangle_p01 = float(np.percentile(triangle_quality, 1.0))
    triangle_quality_limit = max(
        0.68,
        MIN_TRIANGLE_P01_QUALITY - 0.08 * (mesh_size_scale - 1.0),
    )
    if float(tetra_mean_ratio.min()) < MIN_TETRA_MEAN_RATIO or tetra_p01 < MIN_TETRA_P01_MEAN_RATIO:
        raise RuntimeError(
            f"Tetrahedral quality failed (min={tetra_mean_ratio.min():.3f}, p01={tetra_p01:.3f})."
        )
    if triangle_p01 < triangle_quality_limit:
        raise RuntimeError(f"Surface triangle quality failed (p01={triangle_p01:.3f}).")

    edge_values = triangle_edges.ravel()
    edge_quantiles = np.percentile(edge_values, [1.0, 50.0, 99.0])
    adaptivity_limit = max(1.2, 2.0 / np.sqrt(mesh_size_scale))
    if edge_quantiles[2] / edge_quantiles[0] < adaptivity_limit:
        raise RuntimeError("Adaptive surface mesh is unexpectedly uniform.")
    z_extent = float(surface.bounds[5] - surface.bounds[4])
    if not np.isclose(z_extent, CORE_THICKNESS, rtol=0.0, atol=2.0e-7):
        raise RuntimeError(f"Solid thickness changed to {z_extent:.8f} m.")

    return {
        "watertight_surface": True,
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "triangle_quality": {
            "minimum": float(triangle_quality.min()),
            "p01": triangle_p01,
            "median": float(np.median(triangle_quality)),
            "acceptance_limit": float(triangle_quality_limit),
        },
        "tetra_mean_ratio": {
            "minimum": float(tetra_mean_ratio.min()),
            "p01": tetra_p01,
            "median": float(np.median(tetra_mean_ratio)),
        },
        "surface_edge_length_m": {
            "p01": float(edge_quantiles[0]),
            "median": float(edge_quantiles[1]),
            "p99": float(edge_quantiles[2]),
            "p99_to_p01": float(edge_quantiles[2] / edge_quantiles[0]),
            "acceptance_limit": float(adaptivity_limit),
        },
    }


def ellipse_face(gmsh, center_x: float, width: float, height: float) -> int:
    occ = gmsh.model.occ
    if width >= height:
        return occ.addDisk(center_x, 0.0, CORE_Z_MIN, width / 2.0, height / 2.0)
    tag = occ.addDisk(center_x, 0.0, CORE_Z_MIN, height / 2.0, width / 2.0)
    occ.rotate([(2, tag)], center_x, 0.0, CORE_Z_MIN, 0.0, 0.0, 1.0, 0.5 * np.pi)
    return tag


def gap_slot_start(variant: CoreVariant) -> float:
    """Place the slot inside its window far enough to avoid Boolean sliver faces."""
    inner_rx = 0.5 * variant.width * variant.window_ratio_x
    if variant.profile == "rectangle":
        return variant.center_x + inner_rx - 0.0015
    inner_ry = 0.5 * variant.height * variant.window_ratio_y
    normalized_y = min(0.92, 0.5 * variant.inner_gap / inner_ry)
    intersection_x = variant.center_x + inner_rx * np.sqrt(1.0 - normalized_y**2)
    return float(intersection_x - 0.0015)


def add_gap_face(
    gmsh,
    x_start: float,
    x_end: float,
    inner_gap: float,
    tip_gap: float,
    z: float = CORE_Z_MIN,
) -> int:
    occ = gmsh.model.occ
    points = [
        occ.addPoint(x_start, -inner_gap / 2.0, z),
        occ.addPoint(x_end, -tip_gap / 2.0, z),
        occ.addPoint(x_end, tip_gap / 2.0, z),
        occ.addPoint(x_start, inner_gap / 2.0, z),
    ]
    lines = [occ.addLine(points[index], points[(index + 1) % 4]) for index in range(4)]
    wire = occ.addWire(lines)
    return occ.addPlaneSurface([wire])


def circular_core_dimensions(variant: CoreVariant) -> dict[str, float]:
    """Return constrained dimensions for an analytic round-section C-core."""
    radius = 0.5 * CORE_THICKNESS
    left_axis_x = COIL_AXIS_X
    bend_radius = 0.5 * variant.height - radius
    right_edge = variant.center_x + 0.5 * variant.width
    bend_center_x = right_edge - bend_radius - radius
    if bend_radius <= 2.5 * radius:
        raise ValueError(f"{variant.name}: round-section bend radius is too tight.")
    if bend_center_x - left_axis_x <= 3.0 * radius:
        raise ValueError(f"{variant.name}: round-section straight arm is too short.")
    return {
        "section_radius_m": radius,
        "left_axis_x_m": left_axis_x,
        "bend_radius_m": bend_radius,
        "bend_center_x_m": bend_center_x,
        "right_edge_m": right_edge,
        "slot_start_m": bend_center_x + bend_radius - radius - 0.002,
        "dual_bridge_radius_m": 0.009,
    }


def add_gap_volume(
    gmsh,
    x_start: float,
    x_end: float,
    inner_gap: float,
    tip_gap: float,
    z_min: float,
    z_max: float,
) -> int:
    """Create a through-thickness tapered pole-gap cutter."""
    face = add_gap_face(gmsh, x_start, x_end, inner_gap, tip_gap, z_min)
    extruded = gmsh.model.occ.extrude([(2, face)], 0.0, 0.0, z_max - z_min)
    volumes = [tag for dim, tag in extruded if dim == 3]
    if len(volumes) != 1:
        raise RuntimeError("Pole-gap extrusion did not produce one cutter volume.")
    return volumes[0]


def build_circular_material(gmsh, variant: CoreVariant) -> tuple[list[int], dict[str, float | str]]:
    """Build a true round-section racetrack C-core from analytic OCC solids."""
    occ = gmsh.model.occ
    dims = circular_core_dimensions(variant)
    radius = dims["section_radius_m"]
    left_x = dims["left_axis_x_m"]
    bend_radius = dims["bend_radius_m"]
    bend_x = dims["bend_center_x_m"]

    parts = [
        (3, occ.addCylinder(left_x, -bend_radius, 0.0, 0.0, 2.0 * bend_radius, 0.0, radius)),
        (3, occ.addCylinder(left_x, bend_radius, 0.0, bend_x - left_x, 0.0, 0.0, radius)),
        (3, occ.addCylinder(left_x, -bend_radius, 0.0, bend_x - left_x, 0.0, 0.0, radius)),
        (3, occ.addSphere(left_x, bend_radius, 0.0, radius)),
        (3, occ.addSphere(left_x, -bend_radius, 0.0, radius)),
    ]
    torus = occ.addTorus(bend_x, 0.0, 0.0, bend_radius, radius)
    clip = occ.addBox(
        bend_x - 0.001,
        -bend_radius - radius - 0.002,
        -radius - 0.002,
        bend_radius + radius + 0.003,
        2.0 * (bend_radius + radius + 0.002),
        2.0 * (radius + 0.002),
    )
    half_torus, _ = occ.intersect(
        [(3, torus)], [(3, clip)], removeObject=True, removeTool=True
    )
    parts.extend(half_torus)
    if variant.window_topology == "dual":
        bridge_radius = dims["dual_bridge_radius_m"]
        parts.append(
            (
                3,
                occ.addCylinder(
                    RETURN_BODY_LEFT_X,
                    -bend_radius,
                    0.0,
                    0.0,
                    2.0 * bend_radius,
                    0.0,
                    bridge_radius,
                ),
            )
        )

    fused, _ = occ.fuse(parts[:1], parts[1:], removeObject=True, removeTool=True)
    volumes = [tag for dim, tag in fused if dim == 3]
    if len(volumes) != 1:
        raise RuntimeError(f"{variant.name}: circular-section union did not produce one solid.")
    gap = add_gap_volume(
        gmsh,
        dims["slot_start_m"],
        dims["right_edge_m"] + 0.008,
        variant.inner_gap,
        variant.tip_gap,
        -radius - 0.003,
        radius + 0.003,
    )
    cut, _ = occ.cut(
        [(3, volumes[0])], [(3, gap)], removeObject=True, removeTool=True
    )
    volumes = [tag for dim, tag in cut if dim == 3]
    if len(volumes) != 1:
        raise RuntimeError(f"{variant.name}: circular-section gap cut disconnected the core.")
    solid_volume = float(occ.getMass(3, volumes[0]))
    if not 1.0e-4 <= solid_volume <= 2.0e-3:
        raise RuntimeError(f"{variant.name}: implausible round-core volume {solid_volume:.6g} m^3.")
    return volumes, {
        "construction": "analytic circular cylinders, spherical elbows, and half-torus",
        "section_radius_m": radius,
        "solid_volume_m3": solid_volume,
    }


def add_smooth_unified_aperture(
    gmsh,
    variant: CoreVariant,
    inner_width: float,
    inner_height: float,
) -> int:
    """Create one smooth window/connector cutter with no Boolean cusps."""
    occ = gmsh.model.occ
    rx = 0.5 * inner_width
    ry = 0.5 * inner_height
    exponent = 7.0 if variant.profile == "rectangle" else 2.0
    theta_start = np.deg2rad(variant.transition_angle_deg)

    def profile_xy(theta: np.ndarray) -> np.ndarray:
        cosine = np.cos(theta)
        sine = np.sin(theta)
        x = variant.center_x + rx * np.sign(cosine) * np.abs(cosine) ** (2.0 / exponent)
        y = ry * np.sign(sine) * np.abs(sine) ** (2.0 / exponent)
        return np.column_stack((x, y))

    arc_theta = np.linspace(theta_start, -theta_start, 193)
    arc = profile_xy(arc_theta)
    upper_join = arc[0]
    lower_join = arc[-1]
    connector_x = -0.053
    # Match the fixed winding aperture exactly at the bridge interface, then
    # taper smoothly into the shaped return-body window.
    throat_half_height = 0.054
    fillet_radius = min(APERTURE_FILLET_RADIUS, 0.40 * (STRAIGHT_ARM_END_X - connector_x))
    upper_vertical = np.array([connector_x, throat_half_height - fillet_radius])
    upper_start = np.array([connector_x + fillet_radius, throat_half_height])
    lower_end = np.array([connector_x + fillet_radius, -throat_half_height])
    lower_vertical = np.array([connector_x, -throat_half_height + fillet_radius])

    # Join the right-hand profile arc below 90 degrees. Its upper and lower
    # tangents then point monotonically into the gap, avoiding re-entrant hooks.
    power = 2.0 / exponent
    cosine = np.cos(arc_theta[[0, -1]])
    sine = np.sin(arc_theta[[0, -1]])
    tangents = np.column_stack(
        (
            rx * power * np.abs(cosine) ** (power - 1.0) * sine,
            -ry * power * np.abs(sine) ** (power - 1.0) * cosine,
        )
    )
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    upper_tangent, lower_tangent = tangents
    upper_transition = np.array([STRAIGHT_ARM_END_X, throat_half_height])
    lower_transition = np.array([STRAIGHT_ARM_END_X, -throat_half_height])
    blend_length = min(0.018, 0.28 * (upper_join[0] - STRAIGHT_ARM_END_X))

    z = CORE_Z_MIN
    upper_points = [
        upper_transition,
        upper_transition + np.array([blend_length, 0.0]),
        upper_join - blend_length * upper_tangent,
        upper_join,
    ]
    lower_points = [
        lower_join,
        lower_join + blend_length * lower_tangent,
        lower_transition + np.array([blend_length, 0.0]),
        lower_transition,
    ]
    circle_bezier = 0.5522847498
    upper_fillet_points = [
        upper_vertical,
        upper_vertical + np.array([0.0, circle_bezier * fillet_radius]),
        upper_start - np.array([circle_bezier * fillet_radius, 0.0]),
        upper_start,
    ]
    lower_fillet_points = [
        lower_end,
        lower_end - np.array([circle_bezier * fillet_radius, 0.0]),
        lower_vertical - np.array([0.0, circle_bezier * fillet_radius]),
        lower_vertical,
    ]
    upper_fillet_tags = [occ.addPoint(float(x), float(y), z) for x, y in upper_fillet_points]
    upper_transition_tag = occ.addPoint(float(upper_transition[0]), float(upper_transition[1]), z)
    upper_tags = [upper_transition_tag] + [occ.addPoint(float(x), float(y), z) for x, y in upper_points[1:]]
    arc_tags = [upper_tags[-1]] + [
        occ.addPoint(float(x), float(y), z) for x, y in arc[1:-1]
    ]
    lower_tags = [occ.addPoint(float(x), float(y), z) for x, y in lower_points]
    lower_fillet_tags = [occ.addPoint(float(x), float(y), z) for x, y in lower_fillet_points]
    arc_tags.append(lower_tags[0])
    upper_fillet = occ.addBezier(upper_fillet_tags)
    upper_straight = occ.addLine(upper_fillet_tags[-1], upper_transition_tag)
    upper_curve = occ.addBezier(upper_tags)
    profile_curve = occ.addSpline(arc_tags)
    lower_curve = occ.addBezier(lower_tags)
    lower_straight = occ.addLine(lower_tags[-1], lower_fillet_tags[0])
    lower_fillet = occ.addBezier(lower_fillet_tags)
    closing_side = occ.addLine(lower_fillet_tags[-1], upper_fillet_tags[0])
    return occ.addPlaneSurface(
        [
            occ.addWire(
                [
                    upper_fillet,
                    upper_straight,
                    upper_curve,
                    profile_curve,
                    lower_curve,
                    lower_straight,
                    lower_fillet,
                    closing_side,
                ]
            )
        ]
    )


def add_outer_material_face(gmsh, variant: CoreVariant) -> int:
    """Build the complete outer silhouette as one loop without Boolean unions."""
    occ = gmsh.model.occ
    exponent = 7.0 if variant.profile == "rectangle" else 2.0
    power = 2.0 / exponent
    theta = np.linspace(np.deg2rad(50.0), np.deg2rad(-50.0), 257)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    outer_arc = np.column_stack(
        (
            variant.center_x
            + 0.5 * variant.width * np.sign(cosine) * np.abs(cosine) ** power,
            0.5 * variant.height * np.sign(sine) * np.abs(sine) ** power,
        )
    )
    tangent = np.column_stack(
        (
            0.5 * variant.width * power * np.abs(cosine[[0, -1]]) ** (power - 1.0) * sine[[0, -1]],
            -0.5 * variant.height * power * np.abs(sine[[0, -1]]) ** (power - 1.0) * cosine[[0, -1]],
        )
    )
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    upper_join, lower_join = outer_arc[[0, -1]]
    upper_tangent, lower_tangent = tangent
    upper_transition = np.array([STRAIGHT_ARM_END_X, 0.073])
    lower_transition = np.array([STRAIGHT_ARM_END_X, -0.073])
    blend_length = min(0.022, 0.24 * (upper_join[0] - STRAIGHT_ARM_END_X))
    upper_blend = (
        upper_transition,
        upper_transition + np.array([blend_length, 0.0]),
        upper_join - blend_length * upper_tangent,
        upper_join,
    )
    lower_blend = (
        lower_join,
        lower_join + blend_length * lower_tangent,
        lower_transition + np.array([blend_length, 0.0]),
        lower_transition,
    )

    z = CORE_Z_MIN
    upper_left = occ.addPoint(-0.080, 0.073, z)
    upper_transition_tag = occ.addPoint(*upper_transition, z)
    upper_tags = [upper_transition_tag] + [occ.addPoint(float(x), float(y), z) for x, y in upper_blend[1:]]
    arc_tags = [upper_tags[-1]] + [occ.addPoint(float(x), float(y), z) for x, y in outer_arc[1:-1]]
    lower_tags = [occ.addPoint(float(x), float(y), z) for x, y in lower_blend]
    arc_tags.append(lower_tags[0])
    lower_left = occ.addPoint(-0.080, -0.073, z)
    curves = [
        occ.addLine(upper_left, upper_transition_tag),
        occ.addBezier(upper_tags),
        occ.addSpline(arc_tags),
        occ.addBezier(lower_tags),
        occ.addLine(lower_tags[-1], lower_left),
        occ.addLine(lower_left, upper_left),
    ]
    return occ.addPlaneSurface([occ.addWire(curves)])


def build_material(gmsh, variant: CoreVariant) -> tuple[list[int], dict[str, float]]:
    occ = gmsh.model.occ
    outer_face = add_outer_material_face(gmsh, variant)

    inner_width = variant.width * variant.window_ratio_x
    inner_height = variant.height * variant.window_ratio_y
    tools: list[tuple[int, int]] = []
    if variant.window_topology == "unified":
        aperture = add_smooth_unified_aperture(
            gmsh,
            variant,
            inner_width,
            inner_height,
        )
        tools.append((2, aperture))
    else:
        winding_aperture = occ.addRectangle(
            -0.052,
            -0.054,
            CORE_Z_MIN,
            RETURN_BODY_LEFT_X - (-0.052),
            0.108,
        )
        tools.append((2, winding_aperture))
        if variant.profile == "rectangle":
            inner = occ.addRectangle(
                variant.center_x - inner_width / 2.0,
                -inner_height / 2.0,
                CORE_Z_MIN,
                inner_width,
                inner_height,
            )
        else:
            inner = ellipse_face(gmsh, variant.center_x, inner_width, inner_height)
        tools.append((2, inner))

    right_edge = variant.center_x + variant.width / 2.0
    tools.append(
        (
            2,
            add_gap_face(
                gmsh,
                gap_slot_start(variant),
                right_edge + 0.008,
                variant.inner_gap,
                variant.tip_gap,
            ),
        )
    )
    cut, _ = occ.cut([(2, outer_face)], tools, removeObject=True, removeTool=True)
    material_faces = [tag for dim, tag in cut if dim == 2]
    if len(material_faces) != 1:
        raise RuntimeError(f"{variant.name}: planar subtraction did not produce one material face.")
    cross_section_area = float(occ.getMass(2, material_faces[0]))
    if not 0.002 <= cross_section_area <= 0.030:
        raise RuntimeError(f"{variant.name}: implausible material area {cross_section_area:.6f} m^2.")
    extruded = occ.extrude([(2, material_faces[0])], 0.0, 0.0, CORE_THICKNESS)
    volumes = [tag for dim, tag in extruded if dim == 3]
    if len(volumes) != 1:
        raise RuntimeError(f"{variant.name}: cross-section extrusion did not produce one solid.")
    solid_volume = float(occ.getMass(3, volumes[0]))
    extrusion_error = abs(solid_volume - cross_section_area * CORE_THICKNESS) / solid_volume
    if extrusion_error > 1.0e-7:
        raise RuntimeError(f"{variant.name}: extrusion volume consistency failed ({extrusion_error:.3e}).")
    return volumes, {
        "construction": "explicit planar loops, one aperture cut, and one extrusion",
        "cross_section_area_m2": cross_section_area,
        "solid_volume_m3": solid_volume,
        "extrusion_relative_error": float(extrusion_error),
    }


def build_coupled_geometry(
    gmsh, variant: CoreVariant
) -> tuple[dict[str, list[int]], list[int], dict[str, float | str]]:
    """Create conformal core and air domains while leaving the fixed winding unmeshed."""
    occ = gmsh.model.occ
    if variant.section_profile == "circular":
        material, cad_quality = build_circular_material(gmsh, variant)
    else:
        material, cad_quality = build_material(gmsh, variant)

    outer_coil = occ.addCylinder(
        COIL_AXIS_X,
        -COIL_HALF_LENGTH,
        0.0,
        0.0,
        2.0 * COIL_HALF_LENGTH,
        0.0,
        COIL_OUTER_RADIUS,
    )
    inner_coil = occ.addCylinder(
        COIL_AXIS_X,
        -COIL_HALF_LENGTH - 0.001,
        0.0,
        0.0,
        2.0 * (COIL_HALF_LENGTH + 0.001),
        0.0,
        COIL_INNER_RADIUS,
    )
    coil_cut, _ = occ.cut(
        [(3, outer_coil)], [(3, inner_coil)], removeObject=True, removeTool=True
    )
    coil = [tag for dim, tag in coil_cut if dim == 3]
    if len(coil) != 1:
        raise RuntimeError(f"{variant.name}: fixed winding volume construction failed.")

    enclosure = occ.addBox(
        *AIR_MIN,
        *(AIR_MAX - AIR_MIN),
    )
    fragmented, maps = occ.fragment(
        [(3, enclosure)],
        [(3, tag) for tag in material + coil],
        removeObject=True,
        removeTool=True,
    )
    all_volumes = {tag for dim, tag in fragmented if dim == 3}
    material_after = {tag for dim, tag in maps[1] if dim == 3}
    coil_after = {tag for dim, tag in maps[2] if dim == 3}
    air_after = all_volumes - material_after - coil_after
    if len(material_after) != 1 or len(coil_after) != 1 or not air_after:
        raise RuntimeError(
            f"{variant.name}: conformal fragmentation produced invalid domains "
            f"(core={material_after}, air={air_after}, coil={coil_after})."
        )
    occ.synchronize()
    coil_faces = []
    for volume in coil_after:
        coil_faces.extend(
            tag
            for dim, tag in gmsh.model.getBoundary(
                [(3, volume)], oriented=False, recursive=False
            )
            if dim == 2
        )
    occ.remove([(3, tag) for tag in coil_after], recursive=True)
    occ.synchronize()
    remaining = {tag for _, tag in gmsh.model.getEntities(3)}
    if not material_after <= remaining or not air_after <= remaining:
        raise RuntimeError(f"{variant.name}: removing the winding damaged a solve domain.")
    return {
        "material": sorted(material_after),
        "air": sorted(air_after),
    }, sorted(set(coil_faces)), cad_quality


def configure_fine_mesh(
    gmsh,
    material: list[int],
    coil_faces: list[int],
    mesh_size_scale: float,
) -> None:
    material_faces = []
    for volume in material:
        material_faces.extend(
            tag
            for dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False, recursive=False)
            if dim == 2
        )
    material_edges = []
    for face in sorted(set(material_faces)):
        material_edges.extend(
            tag
            for dim, tag in gmsh.model.getBoundary([(2, face)], oriented=False, recursive=False)
            if dim == 1
        )

    # Select pole-gap features from the CAD's own bounding box. This avoids a
    # coordinate-aligned refinement box and, importantly, does not refine every
    # harmless OCC seam on a circular-section core.
    core_bounds = np.asarray(gmsh.model.getBoundingBox(3, material[0]), dtype=np.float64)
    core_x_span = core_bounds[3] - core_bounds[0]
    pole_threshold = core_bounds[3] - 0.24 * core_x_span
    feature_edges = []
    for edge in sorted(set(material_edges)):
        edge_bounds = gmsh.model.getBoundingBox(1, edge)
        if edge_bounds[3] >= pole_threshold:
            feature_edges.append(edge)
    if not feature_edges:
        raise RuntimeError("No pole-gap feature edges were identified for proximity refinement.")

    fields = gmsh.model.mesh.field
    edge_distance = fields.add("Distance")
    gmsh.model.mesh.field.setNumbers(edge_distance, "EdgesList", feature_edges)
    gmsh.model.mesh.field.setNumber(edge_distance, "Sampling", EDGE_DISTANCE_SAMPLING)
    proximity = fields.add("Threshold")
    gmsh.model.mesh.field.setNumber(proximity, "InField", edge_distance)
    gmsh.model.mesh.field.setNumber(proximity, "SizeMin", MESH_EDGE_SIZE * mesh_size_scale)
    gmsh.model.mesh.field.setNumber(proximity, "SizeMax", AIR_MESH_MAX_SIZE * mesh_size_scale)
    # These are physical influence distances, not mesh sizes. Keeping them
    # fixed prevents adaptive regions from shrinking as the mesh is refined.
    gmsh.model.mesh.field.setNumber(proximity, "DistMin", EDGE_REFINEMENT_DIST_MIN)
    gmsh.model.mesh.field.setNumber(proximity, "DistMax", EDGE_REFINEMENT_DIST_MAX)

    coil_distance = fields.add("Distance")
    fields.setNumbers(coil_distance, "FacesList", coil_faces)
    fields.setNumber(coil_distance, "Sampling", COIL_DISTANCE_SAMPLING)
    coil_proximity = fields.add("Threshold")
    fields.setNumber(coil_proximity, "InField", coil_distance)
    fields.setNumber(coil_proximity, "SizeMin", COIL_MESH_SIZE * mesh_size_scale)
    fields.setNumber(coil_proximity, "SizeMax", AIR_MESH_MAX_SIZE * mesh_size_scale)
    fields.setNumber(coil_proximity, "DistMin", 0.0)
    fields.setNumber(coil_proximity, "DistMax", COIL_REFINEMENT_DIST_MAX)

    # Resolve the complete material boundary independently of the coarser
    # solid interior. This gives near-quadratic surface growth without forcing
    # an unnecessary eightfold increase throughout the tetrahedral volume.
    surface_constant = fields.add("MathEval")
    fields.setString(surface_constant, "F", f"{MESH_SURFACE_MAX_SIZE * mesh_size_scale:.12g}")
    surface_restrict = fields.add("Restrict")
    fields.setNumber(surface_restrict, "InField", surface_constant)
    fields.setNumbers(surface_restrict, "SurfacesList", sorted(set(material_faces)))

    core_constant = fields.add("MathEval")
    fields.setString(core_constant, "F", f"{MESH_MAX_SIZE * mesh_size_scale:.12g}")
    core_restrict = fields.add("Restrict")
    fields.setNumber(core_restrict, "InField", core_constant)
    fields.setNumbers(core_restrict, "VolumesList", material)
    combined = fields.add("Min")
    fields.setNumbers(
        combined,
        "FieldsList",
        [proximity, coil_proximity, surface_restrict, core_restrict],
    )
    fields.setAsBackgroundMesh(combined)

    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", CURVATURE_POINTS_PER_2PI)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvatureIsotropic", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_MIN_SIZE * mesh_size_scale)
    gmsh.option.setNumber("Mesh.MeshSizeMax", AIR_MESH_MAX_SIZE * mesh_size_scale)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)


def circular_loop_h_field(
    points: np.ndarray,
    center_y: float,
) -> np.ndarray:
    """Evaluate the analytic H field of one circular filament about the y axis."""
    from scipy.special import ellipe, ellipk

    x = points[:, 0] - COIL_AXIS_X
    axial = points[:, 1] - center_y
    z = points[:, 2]
    radial = np.hypot(x, z)
    radius = COIL_CENTERLINE_RADIUS
    denominator = np.sqrt((radius + radial) ** 2 + axial**2)
    delta = (radius - radial) ** 2 + axial**2
    parameter = np.clip(
        4.0 * radius * radial / denominator**2,
        0.0,
        1.0 - 1.0e-12,
    )
    first = ellipk(parameter)
    second = ellipe(parameter)

    radial_h = np.zeros_like(radial)
    off_axis = radial > 1.0e-12
    radial_h[off_axis] = (
        COIL_CURRENT_A
        * axial[off_axis]
        / (2.0 * np.pi * radial[off_axis] * denominator[off_axis])
        * (
            -first[off_axis]
            + (
                (radius**2 + radial[off_axis] ** 2 + axial[off_axis] ** 2)
                / delta[off_axis]
            )
            * second[off_axis]
        )
    )
    axial_h = (
        COIL_CURRENT_A
        / (2.0 * np.pi * denominator)
        * (
            first
            + ((radius**2 - radial**2 - axial**2) / delta) * second
        )
    )
    field = np.zeros_like(points)
    field[:, 1] = axial_h
    field[off_axis, 0] = radial_h[off_axis] * x[off_axis] / radial[off_axis]
    field[off_axis, 2] = radial_h[off_axis] * z[off_axis] / radial[off_axis]
    return field


def impressed_coil_h_field(points: np.ndarray) -> np.ndarray:
    """Sum the fixed, equally spaced winding filaments without case metadata."""
    field = np.zeros_like(points, dtype=np.float64)
    for center_y in np.linspace(
        -0.875 * COIL_HALF_LENGTH,
        0.875 * COIL_HALF_LENGTH,
        COIL_TURNS,
    ):
        field += circular_loop_h_field(points, float(center_y))
    if not np.isfinite(field).all():
        raise RuntimeError("The analytic impressed-coil field contains non-finite values.")
    return field


def solve_air_coupled_core(
    coupled: pv.UnstructuredGrid,
    surface: pv.PolyData,
    variant: CoreVariant,
    mesh_size_scale: float,
) -> tuple[pv.UnstructuredGrid, dict[str, object]]:
    """Solve a reduced scalar-potential correction in conformal core and air."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    points = np.asarray(coupled.points, dtype=np.float64)
    tetrahedra = np.asarray(coupled.cells_dict[pv.CellType.TETRA], dtype=np.int64)
    regions = np.asarray(coupled.cell_data["region_id"], dtype=np.uint8)
    material = regions == MATERIAL_REGION
    gradients, volumes = tetra_geometry(points, tetrahedra)
    centroids = points[tetrahedra].mean(axis=1)
    impressed_h = impressed_coil_h_field(centroids)
    relative_permeability = np.where(material, 300.0, 1.0)
    products = np.einsum("nik,njk->nij", gradients, gradients)
    local = relative_permeability[:, None, None] * volumes[:, None, None] * products
    rows = np.repeat(tetrahedra, 4, axis=1).ravel()
    columns = np.tile(tetrahedra, (1, 4)).ravel()
    matrix = coo_matrix(
        (local.ravel(), (rows, columns)), shape=(coupled.n_points, coupled.n_points)
    ).tocsr()
    cell_rhs = (
        relative_permeability[:, None]
        * volumes[:, None]
        * np.einsum("ni,nji->nj", impressed_h, gradients)
    )
    rhs = np.zeros(coupled.n_points, dtype=np.float64)
    for local_node in range(4):
        np.add.at(rhs, tetrahedra[:, local_node], cell_rhs[:, local_node])

    tolerance = 2.0e-7
    fixed = np.any(
        (np.abs(points - AIR_MIN) < tolerance)
        | (np.abs(points - AIR_MAX) < tolerance),
        axis=1,
    )
    if np.count_nonzero(fixed) < 8:
        raise RuntimeError(f"{variant.name}: outer-air boundary selection failed.")
    free = ~fixed
    potential = np.zeros(coupled.n_points, dtype=np.float64)
    potential[free] = spsolve(matrix[free][:, free], rhs[free])
    residual = matrix @ potential - rhs
    residual_relative = float(
        np.linalg.norm(residual[free]) / max(np.linalg.norm(rhs[free]), 1.0e-14)
    )
    if not np.isfinite(potential).all() or residual_relative > 1.0e-10:
        raise RuntimeError(f"{variant.name}: magnetic solve failed (relative residual={residual_relative:.3e}).")

    correction_h = np.einsum("ni,nij->nj", potential[tetrahedra], gradients)
    cell_h = impressed_h - correction_h
    cell_b = MU0 * relative_permeability[:, None] * cell_h
    if not np.isfinite(cell_b).all():
        raise RuntimeError(f"{variant.name}: magnetic flux density contains non-finite values.")

    material_tetrahedra = tetrahedra[material]
    material_volumes = volumes[material]
    material_b = cell_b[material]
    point_b = np.zeros((coupled.n_points, 3), dtype=np.float64)
    point_weights = np.zeros(coupled.n_points, dtype=np.float64)
    weighted_b = material_b * material_volumes[:, None]
    for local_node in range(4):
        np.add.at(point_b, material_tetrahedra[:, local_node], weighted_b)
        np.add.at(point_weights, material_tetrahedra[:, local_node], material_volumes)
    material_points = point_weights > 0.0
    point_b[material_points] /= point_weights[material_points, None]
    coupled.point_data["B_T"] = point_b.astype(np.float32)
    map_point_fields(coupled, surface, ["B_T"])
    solid = coupled.extract_cells(np.flatnonzero(material)).clean()

    rms_components = np.sqrt(np.average(material_b**2, axis=0, weights=material_volumes))
    rms_magnitude = float(
        np.sqrt(np.average(np.sum(material_b**2, axis=1), weights=material_volumes))
    )
    bz_fraction = float(rms_components[2] / max(rms_magnitude, 1.0e-14))
    if not 0.02 <= bz_fraction <= 0.35:
        raise RuntimeError(f"{variant.name}: implausible RMS Bz fraction {bz_fraction:.3%}.")

    # The geometry and excitation are mirror-symmetric about y=0. Compare
    # interpolated nodal fields at paired interior locations, avoiding a false
    # mismatch from unrelated cellwise-constant gradients.
    solid_b = np.asarray(solid.point_data["B_T"], dtype=np.float64)
    cell_centers = np.asarray(solid.cell_centers().points, dtype=np.float64)
    sample = np.linspace(0, solid.n_cells - 1, min(20_000, solid.n_cells), dtype=np.int64)
    base_samples = pv.PolyData(cell_centers[sample]).sample(
        solid,
        pass_point_data=True,
        pass_cell_data=False,
        tolerance=2.0e-6,
    )
    mirrored = cell_centers[sample].copy()
    mirrored[:, 1] *= -1.0
    mirror_samples = pv.PolyData(mirrored).sample(
        solid,
        pass_point_data=True,
        pass_cell_data=False,
        tolerance=2.0e-6,
    )
    valid_mirror = np.asarray(base_samples["vtkValidPointMask"], dtype=bool) & np.asarray(
        mirror_samples["vtkValidPointMask"], dtype=bool
    )
    if valid_mirror.mean() < 0.98:
        raise RuntimeError(
            f"{variant.name}: only {valid_mirror.mean():.2%} of symmetry samples are valid."
        )
    b_magnitude = np.linalg.norm(solid_b, axis=1)
    symmetry_scale = max(float(np.percentile(b_magnitude, 95.0)), 1.0e-12)
    base_b_magnitude = np.linalg.norm(np.asarray(base_samples["B_T"], dtype=np.float64), axis=1)
    mirrored_b_magnitude = np.linalg.norm(np.asarray(mirror_samples["B_T"], dtype=np.float64), axis=1)
    symmetry_difference = (
        np.abs(base_b_magnitude[valid_mirror] - mirrored_b_magnitude[valid_mirror])
        / symmetry_scale
    )
    symmetry_p95 = float(np.percentile(symmetry_difference, 95.0))
    symmetry_limit = min(0.16, 0.08 * np.sqrt(mesh_size_scale))
    if symmetry_p95 > symmetry_limit:
        raise RuntimeError(f"{variant.name}: magnetic mirror-symmetry check failed ({symmetry_p95:.3f}).")

    return solid, {
        "formulation": "reduced scalar potential with analytic impressed winding field",
        "coil_magnetomotive_force_A_turn": COIL_TURNS * COIL_CURRENT_A,
        "material_relative_permeability": 300.0,
        "air_domain_included_during_solve": True,
        "residual_relative": residual_relative,
        "weak_flux_divergence_residual_relative": residual_relative,
        "B_max_T": float(np.linalg.norm(material_b, axis=1).max()),
        "B_mean_T": float(np.average(np.linalg.norm(material_b, axis=1), weights=material_volumes)),
        "B_rms_components_T": rms_components.tolist(),
        "Bz_rms_fraction": bz_fraction,
        "Bz_peak_T": float(np.max(np.abs(material_b[:, 2]))),
        "mirror_symmetry": {
            "normalized_absolute_difference_mean": float(symmetry_difference.mean()),
            "normalized_absolute_difference_p95": symmetry_p95,
            "acceptance_limit": float(symmetry_limit),
            "valid_interpolation_fraction": float(valid_mirror.mean()),
        },
    }


def generate_variant(
    variant: CoreVariant,
    output_dir: Path,
    gmsh_threads: int,
    max_cells: int,
    mesh_size_scale: float,
) -> dict[str, object]:
    import gmsh

    constraints = validate_variant(variant)
    timings: dict[str, float] = {}
    started = time.perf_counter()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", gmsh_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", gmsh_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", gmsh_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", gmsh_threads)
        gmsh.model.add(variant.name)
        regions, coil_faces, cad_quality = build_coupled_geometry(gmsh, variant)
        timings["cad_s"] = time.perf_counter() - started
        print(f"[{variant.name}] CAD ready in {timings['cad_s']:.1f}s", flush=True)
        if len(regions["material"]) != 1 or not regions["air"]:
            raise RuntimeError(f"{variant.name}: invalid synchronized solve domains.")
        configure_fine_mesh(
            gmsh,
            regions["material"],
            coil_faces,
            mesh_size_scale,
        )
        meshing_started = time.perf_counter()
        gmsh.model.mesh.generate(3)
        timings["mesh_s"] = time.perf_counter() - meshing_started
        print(f"[{variant.name}] mesh ready in {timings['mesh_s']:.1f}s", flush=True)
        optimization_started = time.perf_counter()
        gmsh.model.mesh.optimize("Netgen", niter=5)
        timings["mesh_optimization_s"] = time.perf_counter() - optimization_started
        print(
            f"[{variant.name}] mesh optimized in {timings['mesh_optimization_s']:.1f}s",
            flush=True,
        )
        extraction_started = time.perf_counter()
        coupled = extract_volume(gmsh, regions)
        surface = extract_material_surface(gmsh, regions["material"])
        timings["extraction_s"] = time.perf_counter() - extraction_started
    finally:
        gmsh.finalize()

    if coupled.n_cells > 3 * max_cells:
        raise RuntimeError(f"{variant.name}: solve mesh has {coupled.n_cells:,} cells.")
    solve_started = time.perf_counter()
    solid, solution = solve_air_coupled_core(coupled, surface, variant, mesh_size_scale)
    timings["solve_s"] = time.perf_counter() - solve_started
    print(f"[{variant.name}] field solved in {timings['solve_s']:.1f}s", flush=True)
    if solid.n_cells > max_cells:
        raise RuntimeError(f"{variant.name}: solid mesh has {solid.n_cells:,} cells.")
    if set(np.unique(solid.cell_data["region_id"]).tolist()) != {MATERIAL_REGION}:
        raise RuntimeError(f"{variant.name}: final volume contains non-solid cells.")
    mesh_quality = validate_mesh(solid, surface, mesh_size_scale)
    for name in list(solid.point_data):
        if name != "B_T":
            del solid.point_data[name]
    for name in list(solid.cell_data):
        del solid.cell_data[name]
    for name in list(surface.point_data):
        if name != "B_T":
            del surface.point_data[name]
    for name in list(surface.cell_data):
        del surface.cell_data[name]

    output_dir.mkdir(parents=True, exist_ok=True)
    surface_path = output_dir / f"{variant.name}_solid_surface.vtp"
    volume_path = output_dir / f"{variant.name}_solid_volume.vtu"
    surface.save(surface_path, binary=True)
    solid.save(volume_path, binary=True)
    return {
        "name": variant.name,
        "profile": variant.profile,
        "section_profile": variant.section_profile,
        "window_topology": variant.window_topology,
        "pole_style": variant.pole_style,
        "variant": asdict(variant),
        "geometry_constraints": constraints,
        "cad_quality": cad_quality,
        "fixed_coil": {
            "axis": "y",
            "center_x_m": COIL_AXIS_X,
            "length_m": 2.0 * COIL_HALF_LENGTH,
            "turn_count": COIL_TURNS,
            "turn_centerline_radius_m": COIL_CENTERLINE_RADIUS,
            "excluded_winding_inner_radius_m": COIL_INNER_RADIUS,
            "excluded_winding_outer_radius_m": COIL_OUTER_RADIUS,
            "current_per_turn_A": COIL_CURRENT_A,
            "magnetomotive_force_A_turn": COIL_TURNS * COIL_CURRENT_A,
        },
        "surface_vtp": surface_path.name,
        "solid_volume_vtu": volume_path.name,
        "surface_points": int(surface.n_points),
        "surface_triangles": int(surface.n_cells),
        "solid_volume_points": int(solid.n_points),
        "solid_tetrahedra": int(solid.n_cells),
        "computational_volume_points": int(coupled.n_points),
        "computational_tetrahedra": int(coupled.n_cells),
        "mesh_quality": mesh_quality,
        "solution": solution,
        "timings": {**timings, "total_s": time.perf_counter() - started},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/parsa/smart_parsa/results/c_core_shape_solution_examples"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gmsh-threads", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=1_000_000)
    parser.add_argument("--mesh-size-scale", type=float, default=1.0)
    parser.add_argument("--variants", default=",".join(variant.name for variant in VARIANTS))
    args = parser.parse_args()

    requested = [name.strip() for name in args.variants.split(",") if name.strip()]
    by_name = {variant.name: variant for variant in VARIANTS}
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    selected = [by_name[name] for name in requested]
    if len(selected) == len(VARIANTS):
        section_counts = {
            section: sum(variant.section_profile == section for variant in selected)
            for section in ("circular", "rectangular")
        }
        if section_counts != {"circular": 4, "rectangular": 4}:
            raise RuntimeError(f"Full example set is not section-balanced: {section_counts}.")
    if args.mesh_size_scale <= 0.0:
        raise ValueError("--mesh-size-scale must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(selected)), mp_context=context) as executor:
        futures = {
            executor.submit(
                generate_variant,
                variant,
                args.output_dir,
                args.gmsh_threads,
                args.max_cells,
                args.mesh_size_scale,
            ): variant
            for variant in selected
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"{record['name']}: surface={record['surface_triangles']:,}, "
                f"solid={record['solid_tetrahedra']:,}, "
                f"Bmax={record['solution']['B_max_T']:.3g} T",
                flush=True,
            )
    order = {variant.name: index for index, variant in enumerate(VARIANTS)}
    records.sort(key=lambda record: order[str(record["name"])])
    summary = {
        "description": (
            "Eight physically connected C-core variants with a fixed straight winding limb, "
            "fixed winding, and balanced circular/rectangular material sections."
        ),
        "final_files_contain": "magnetic solid only; computational air and winding are omitted",
        "reduced_physics": (
            "A conformal core-plus-air solve corrects the analytic three-dimensional field of a "
            "fixed 20 A-turn winding using a reduced magnetic scalar potential."
        ),
        "material_region_id": MATERIAL_REGION,
        "mesh": {
            "method": (
                "independent material-surface and solid-interior sizing with pole-edge proximity, "
                "winding-interface proximity, and isotropic curvature adaptation"
            ),
            "minimum_size_m": MESH_MIN_SIZE * args.mesh_size_scale,
            "edge_proximity_size_m": MESH_EDGE_SIZE * args.mesh_size_scale,
            "surface_maximum_size_m": MESH_SURFACE_MAX_SIZE * args.mesh_size_scale,
            "core_maximum_size_m": MESH_MAX_SIZE * args.mesh_size_scale,
            "air_maximum_size_m": AIR_MESH_MAX_SIZE * args.mesh_size_scale,
            "curvature_points_per_2pi": CURVATURE_POINTS_PER_2PI,
            "edge_distance_sampling": EDGE_DISTANCE_SAMPLING,
            "coil_distance_sampling": COIL_DISTANCE_SAMPLING,
            "edge_refinement_distance_range_m": [
                EDGE_REFINEMENT_DIST_MIN,
                EDGE_REFINEMENT_DIST_MAX,
            ],
            "coil_refinement_distance_max_m": COIL_REFINEMENT_DIST_MAX,
            "coordinate_based_refinement_regions": 0,
            "nominal_section_diameter_or_thickness_m": CORE_THICKNESS,
            "production_triangle_quality_p01_minimum": MIN_TRIANGLE_P01_QUALITY,
            "tetra_mean_ratio_minimum": MIN_TETRA_MEAN_RATIO,
            "tetra_mean_ratio_p01_minimum": MIN_TETRA_P01_MEAN_RATIO,
        },
        "geometry_constraints": {
            "minimum_magnetic_web_m": MIN_MAGNETIC_WEB,
            "unified_aperture_fillet_radius_m": APERTURE_FILLET_RADIUS,
            "outer_and_inner_transitions": "monotone cubic Bezier curves tangent to the profile arcs",
            "cad_construction": (
                "rectangular sections use explicit planar loops and one extrusion; circular sections "
                "use analytic cylinders, spherical elbows, and a half-torus"
            ),
            "section_profile_counts": {
                section: sum(variant.section_profile == section for variant in selected)
                for section in ("circular", "rectangular")
            },
            "inner_gap_range_m": [min(v.inner_gap for v in selected), max(v.inner_gap for v in selected)],
            "tip_gap_range_m": [min(v.tip_gap for v in selected), max(v.tip_gap for v in selected)],
            "maximum_pole_taper_half_angle_deg": 20.0,
        },
        "problems": records,
    }
    summary_path = args.output_dir / "c_core_shape_examples_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
