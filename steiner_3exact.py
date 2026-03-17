import math
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from shapely import affinity
from typing import Tuple, Optional, Any

# --- Geometric Utilities ---

def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """Calculate the angle at p2 formed by p1, p2, and p3 in degrees."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.hypot(*v1)
    mag_v2 = math.hypot(*v2)
    if mag_v1 == 0 or mag_v2 == 0: return 0.0
    cos_theta = max(-1.0, min(1.0, dot_product / (mag_v1 * mag_v2)))
    return math.degrees(math.acos(cos_theta))

def find_equilateral_triangle_vertex(a: Point, b: Point, cw: int = 1) -> Point:
    """Find the third vertex of an equilateral triangle given two vertices."""
    base_vector = LineString([a, b])
    rotated_vector = affinity.rotate(base_vector, 60 * cw, origin=(a.x, a.y))
    return Point(rotated_vector.coords[1])

def find_fermat_point(p1: Point, p2: Point, p3: Point) -> Optional[Point]:
    """Find the Fermat point (Torricelli point) of a triangle."""
    def round_point(p, digits=9): return Point(round(p.x, digits), round(p.y, digits))
    for cw in [1, -1]:
        p4 = find_equilateral_triangle_vertex(p1, p2, cw)
        p5 = find_equilateral_triangle_vertex(p2, p3, cw)
        p6 = find_equilateral_triangle_vertex(p3, p1, cw)
        l1, l2, l3 = LineString([p3, p4]), LineString([p1, p5]), LineString([p2, p6])
        i1, i2, i3 = l1.intersection(l2), l1.intersection(l3), l2.intersection(l3)
        if not (i1.is_empty or i2.is_empty or i3.is_empty):
            i1, i2, i3 = round_point(i1), round_point(i2), round_point(i3)
            if i1.distance(i2) < 1e-7 and i1.distance(i3) < 1e-7: return i1
    return None

def normalize_points(pa: Tuple[float, float], pb: Tuple[float, float], pc: Tuple[float, float], r: float = 1.0):
    """Normalize points such that pb is at origin and pc is on positive x-axis."""
    pa, pb, pc = np.array(pa), np.array(pb), np.array(pc)
    v = pc - pb
    d = np.linalg.norm(v)
    if d == 0: raise ValueError("pb and pc must be distinct")
    pa0, pc0 = pa - pb, pc - pb
    angle = -math.atan2(pc0[1], pc0[0])
    ca, sa = math.cos(angle), math.sin(angle)
    rot_matrix = np.array([[ca, -sa], [sa, ca]])
    pa1, pc1 = rot_matrix @ pa0, rot_matrix @ pc0
    scale = 1.0 / d
    return pa1 * scale, np.array([0.0, 0.0]), pc1 * scale, r * scale

def circle_circle_intersection_nearest(c1: Point, c2: Point, r1: float, r2: float, ref_p: Point) -> Optional[Point]:
    """Find the intersection point of two circles nearest to a reference point."""
    p1, p2, p_ref = np.array((c1.x, c1.y)), np.array((c2.x, c2.y)), np.array((ref_p.x, ref_p.y))
    d = np.linalg.norm(p2 - p1)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0: return None
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = math.sqrt(max(0, r1**2 - a**2))
    p_mid = p1 + a * (p2 - p1) / d
    perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]]) / d
    i1, i2 = p_mid + h * perp, p_mid - h * perp
    return Point(i1) if np.linalg.norm(i1 - p_ref) < np.linalg.norm(i2 - p_ref) else Point(i2)

# --- 3-Exact Steiner Algorithm ---

def solve_3exact_steiner(pa: Point, pb: Point, pc: Point, d_max: float) -> Tuple[int, Point]:
    """
    Find the optimal Steiner junction for three terminals under bounded-edge constraint (Equation 6).
    
    :param pa, pb, pc: Terminal points.
    :param d_max: Maximum segment length.
    :return: (repeater_count, junction_point)
    """
    pf = find_fermat_point(pa, pb, pc)
    if not pf: return 0, pa # Fallback

    def get_ijk(a, b, c, x, r):
        return math.ceil(x.distance(a)/r), math.ceil(x.distance(b)/r), math.ceil(x.distance(c)/r)

    i_f, j_f, k_f = get_ijk(pa, pb, pc, pf, d_max)
    t_base = i_f + j_f + k_f - 2

    # Trial configurations from Shin & Choi 2023
    params = [
        (i_f-2, j_f, k_f), (i_f-3, j_f, k_f+1), (i_f-4, j_f, k_f+2), (i_f-3, j_f+1, k_f),
        (i_f-1, j_f, k_f), (i_f-2, j_f, k_f+1), (i_f-3, j_f, k_f+2), (i_f-2, j_f+1, k_f)
    ]

    for i, p in enumerate(params):
        if sum(p) == (t_base + (0 if i < 4 else 1)):
            valid_p = _calc_eq6_feasibility(pa, pb, pc, p[0], p[1], p[2], t_base, d_max)
            if valid_p:
                p_j = circle_circle_intersection_nearest(pb, pc, valid_p[1]*d_max, valid_p[2]*d_max, pa)
                if p_j:
                    return (t_base + (0 if i < 4 else 1) - 2, p_j)
    
    return t_base, pf

def _calc_eq6_feasibility(pa, pb, pc, i, j, k, t, d_max):
    """Checks if a discrete hop count configuration is feasible by root finding."""
    pa_n, pb_n, pc_n, r_n = normalize_points((pa.x, pa.y), (pb.x, pb.y), (pc.x, pc.y), d_max)
    xa, ya = pa_n[0], pa_n[1]
    a = (((j - k) * r_n) + 1) / 2
    b, s, c0, m = a - xa, a * (1 - a) * r_n, r_n * (t + 2) - 1, (2 * a - 1) * r_n
    a2, a1, a0 = m**2 + 4*s*r_n - 4*r_n**2, 2*m*b + 4*s + 4*r_n*c0, b**2 + ya**2 - c0**2
    coeffs = [a2**2, 2*a2*a1, a1**2 + 2*a2*a0 - 16*ya**2*s*r_n, 2*a1*a0 - 16*ya**2*s, a0**2]
    roots = sorted([r.real for r in np.roots(coeffs) if abs(r.imag) < 1e-10])
    sub = (j + k) / 2 + 1 / (2 * r_n)
    for idx in range(len(roots) - 1):
        mid = (roots[idx] + roots[idx+1]) / 2
        if np.polyval(coeffs, mid) <= 1e-10:
            for z in range(math.ceil(roots[idx] - sub), math.floor(roots[idx+1] - sub) + 1):
                ci, cj, ck = i - 2*z, j + z, k + z
                p_j = circle_circle_intersection_nearest(Point(0,0), Point(1,0), cj*r_n, ck*r_n, Point(xa, ya))
                if p_j and p_j.distance(Point(xa, ya)) <= ci*r_n:
                    return ci, cj, ck
    return None
