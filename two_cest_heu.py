import math
from shapely import *
import numpy as np
from graph_tools import *

# def find_fermat_point(p1, p2, p3):
#     """
#     Find the Fermat point of a triangle defined by three vertices.
#     The Fermat point minimizes the total distance to the three vertices.
            
#     :param p1: First vertex as a Point object.
#     :param p2: Second vertex as a Point object.
#     :param p3: Third vertex as a Point object.
#     :return: The Fermat point as a Point object, or None if not found.
#     """
#     def round_point(point,decimal_digit = 9):
    
#         # print(point)
#         return Point(round(point.x, decimal_digit), round(point.y, decimal_digit))
    
#     def fermat_intersection(lines):

#         # Find intersections
#         intersection1 = lines[0].intersection(lines[1])
#         intersection2 = lines[0].intersection(lines[2])
#         intersection3 = lines[1].intersection(lines[2])
#         if intersection1.is_empty or intersection2.is_empty or intersection3.is_empty:
#             # print("no fermat point1")
#             return None
#         else:
#             intersection1 = round_point(intersection1)
#             intersection2 = round_point(intersection2)
#             intersection3 = round_point(intersection3)

#         if intersection_all([intersection1, intersection2, intersection3]):
#             return intersection1
#         else:
#             # print("no fermat point2")
#             return None
        
#     two_clockwise_test = [1, -1]
    
#     for cw_direction in two_clockwise_test:
         
#         p4 = find_equilateral_triangle_vertex(p1, p2, cw_direction)
#         p5 = find_equilateral_triangle_vertex(p2, p3, cw_direction)
#         p6 = find_equilateral_triangle_vertex(p3, p1, cw_direction)

#         #create Fermat line
#         p3p4 = LineString([p3, p4])
#         p1p5 = LineString([p1, p5])
#         p2p6 = LineString([p2, p6])

#         fermat_lines = [p3p4, p1p5, p2p6]

#         fermat_point = fermat_intersection(fermat_lines)
        
#         if fermat_point != None:
#             return fermat_point
    
#     return None


def find_equilateral_triangle_vertex(a, b, cw=1):
    """
    Find the third vertex of an equilateral triangle given two vertices.
    
    :param a: First vertex as a Point object.
    :param b: Second vertex as a Point object.
    :param cw: Clockwise direction (1 for clockwise, -1 for counterclockwise).
    :return: Third vertex as Point object.
    """
    base_vector = LineString([a, b])
    rotated_vector = affinity.rotate(base_vector, 60 * cw, origin=(a.x, a.y))
    return Point(rotated_vector.coords[1])


def calc_distance(p1, p2):
    """
    Calculate the Euclidean distance between two coordinate tuples.
            
    :param p1: First point as a tuple (x, y).
    :param p2: Second point as a tuple (x, y).
    :return: Euclidean distance.
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle at p2 formed by three points p1, p2, and p3.
    
    :param p1: Tuple (x1, y1) - First point
    :param p2: Tuple (x2, y2) - Second point (vertex of the angle)
    :param p3: Tuple (x3, y3) - Third point
    :return: Angle at p2 in degrees
    """
    # Create vectors
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    # Calculate dot product and magnitudes
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # Calculate cosine of the angle
    if mag_v1 == 0 or mag_v2 == 0:
        raise ValueError("Magnitude of one of the vectors is zero; points must not overlap.")
    cos_theta = dot_product / (mag_v1 * mag_v2)
    
    # Numerical stability: Clamp cosine value to [-1, 1]
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    # Calculate angle in radians and then convert to degrees
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def triangle_angles(A, B, C):
    """
    Calculate all angles of a triangle formed by points A, B, and C.
    
    :param A: Tuple (x1, y1) - First point
    :param B: Tuple (x2, y2) - Second point
    :param C: Tuple (x3, y3) - Third point
    :return: A tuple of three angles (in degrees) corresponding to A, B, and C.
    """
    angle_A = calculate_angle(B, A, C)
    angle_B = calculate_angle(A, B, C)
    angle_C = calculate_angle(A, C, B)
    return angle_A, angle_B, angle_C

def calc_steinernize_MST(G, R=1):
    """
    Calculate internal Steiner points to 'steinernize' the edges of a given graph G.
    
    According to the STP-MSPBEL problem (Steiner Tree Problem with Minimum Number of 
    Steiner Points and Bounded Edge Length), 'steinernizing' an edge means inserting 
    Steiner points to break the edge into smaller segments of length at most R. 
    This function iterates over all edges in G and calculates the coordinates for 
    these inserted points. 
    
    If G represents a Minimum Spanning Tree (MST) of three terminals, it will accurately 
    steinernize the actual edges of that MST, whether it forms a path like (Pa, Pb), (Pb, Pc) 
    or a wedge like (Pa, Pb), (Pa, Pc).
            
    :param G: A NetworkX graph whose nodes have a 'pos' attribute (x, y).
    :param R: Maximum distance constraint (default is 1).
    :return: List of coordinates (x, y) for the inserted Steiner points.
    """
    
    edges = G.edges()
    pos = nx.get_node_attributes(G, 'pos')
    steinerized_nodes = []

    for u, v in edges:
        
        x1, y1 = pos[u][0], pos[u][1]  
        x2, y2 = pos[v][0], pos[v][1]  
        dx, dy = (x2 - x1), (y2 - y1)

        # Prefer stored length; fall back to Euclidean if missing
        L = math.hypot(dx, dy)

        if L <= R:
            # Already within span; no extra discs needed
            continue

        # Number of interior points so that each gap <= lmax
        # This formula works uniformly for all L:
        # if lmax < L <= 2*lmax -> n = 1 (midpoint),
        # if 2*lmax < L <= 3*lmax -> n = 2, etc.
        n = math.ceil(L / R) - 1

        # Parametric positions along the segment at equal spacing
        step = 1.0 / (n + 1)
        for i in range(1, n + 1):
            t = i * step
            steinerized_nodes.append((x1 + dx * t, y1 + dy * t))
            
    return steinerized_nodes

def circle_circle_intersection_nearest(C1, C2, R1, R2, Pa):
    """
    Find the intersection point of two circles nearest to point Pa.
    
    :param C1: Center of first circle
    :param C2: Center of second circle
    :param R1: Radius of first circle
    :param R2: Radius of second circle
    :param Pa: Reference point
    :return: Nearest intersection point as Point object or None
    """
    C1 = np.array((C1.x, C1.y), dtype=float)
    C2 = np.array((C2.x, C2.y), dtype=float)
    Pa = np.array((Pa.x, Pa.y), dtype=float)

    # Distance between centers
    d = np.linalg.norm(C2 - C1)

    # No intersection: circles are separate or one contains the other
    if d > R1 + R2 or d < abs(R1 - R2) or d == 0:
        return None

    # Find point P2 which is the midpoint along the line between C1 and C2
    a = (R1**2 - R2**2 + d**2) / (2 * d)
    P2 = C1 + a * (C2 - C1) / d

    # Height from P2 to the intersection points
    h = np.sqrt(R1**2 - a**2)

    # Direction vector perpendicular to line C1C2
    perp = np.array([-(C2 - C1)[1], (C2 - C1)[0]]) / d

    # Two intersection points
    inter1 = P2 + h * perp
    inter2 = P2 - h * perp

    # Return the one closer to Pa
    if np.linalg.norm(inter1 - Pa) < np.linalg.norm(inter2 - Pa):
        return Point(inter1)
    else:
        return Point(inter2)
    
def calc_ijkF(Pa, Pb, Pc, Px, R):
    """
    Calculate the discrete hop counts (i, j, k) from a point Px to three terminals.
            
    :param Pa: Terminal A Point object.
    :param Pb: Terminal B Point object.
    :param Pc: Terminal C Point object.
    :param Px: Reference point (e.g., Fermat point).
    :param R: Maximum distance constraint.
    :return: Tuple of three integers (iX, jX, kX).
    """

    PaPb = math.dist((Pa.x,Pa.y),(Pb.x,Pb.y))
    PaPc = math.dist((Pa.x,Pa.y),(Pc.x,Pc.y))
    PbPc = math.dist((Pb.x,Pb.y),(Pc.x,Pc.y))
    PaPx = math.dist((Pa.x,Pa.y),(Px.x,Px.y))
    PbPx = math.dist((Pb.x,Pb.y),(Px.x,Px.y))
    PcPx = math.dist((Pc.x,Pc.y),(Px.x,Px.y))
    iX = math.ceil(PaPx/R)
    jX = math.ceil(PbPx/R)
    kX = math.ceil(PcPx/R)
    return iX, jX, kX

def normalize_points(pa, pb, pc, R=1.0):
    """
    Normalize three points such that pb is at the origin and pc is on the x-axis.
            
    :param pa: First point coordinate tuple.
    :param pb: Second point coordinate tuple (becomes origin).
    :param pc: Third point coordinate tuple (rotated to x-axis).
    :param R: Distance value to scale proportionally.
    :return: Tuple of (pa_norm, pb_norm, pc_norm, scaled_R).
    """
    pb = np.array(pb, dtype=float)
    pc = np.array(pc, dtype=float)
    pa = np.array(pa, dtype=float)
    v = pc - pb
    d = np.linalg.norm(v)
    if d == 0:
        raise ValueError("pb and pc must be distinct")
    pa0 = pa - pb
    pc0 = pc - pb
    angle = -math.atan2(pc0[1], pc0[0])
    ca, sa = math.cos(angle), math.sin(angle)
    Rm = np.array([[ca, -sa],[sa, ca]])
    pa1 = Rm @ pa0
    pc1 = Rm @ pc0
    s = 1.0 / d
    pa2 = pa1 * s
    pb2 = np.array([0.0, 0.0])
    pc2 = pc1 * s
    return pa2, pb2, pc2, s*R


def three_exact_bounded_edge_eq6(Pa, Pb, Pc, PF, R):
    """
    Compute an (approximate) optimal Steiner-like point for three terminals under
    a bounded-edge constraint using the Equation (6) approach from Shin & Choi (2023).

    - Inputs:
        * `Pa`, `Pb`, `Pc`: objects representing the three terminals.
        * `PF`: a candidate Fermat.
        * `R`: the communication/coverage radius.

    - Notation correspondence used in code:
        * `iF, jF, kF` : segments along edges when bounding by `R`.
        * `t = iF + jF + kF - 2` corresponds to the total cost baseline
            used to compare alternative placements.

    - Method summary (high level):
        1. Normalize the triangle (rotate/scale) so Pb-Pc lie on a canonical axis
           degrees of freedom and simplifies the algebraic conditions derived in
           Shin & Choi for checking feasibility of Equation (6).
        2. Construct a quartic polynomial whose root intervals indicate parameter 
           ranges where the quartic <= 0 holds. These intervals map to
           candidate integer shifts (Δz) which generate discrete candidate triples
           (Ci,Cj,Ck) representing Steinerized node counts.
        3. For each candidate triple, compute the geometric placement of the
           required junction (via circle-circle intersections) and test whether the
           placement meets the distance bounds (<= Ci*Rdat). If so, accept it as
           a valid Steiner-like point with improved cost.
        4. The routine enumerates a small finite set of parameter tuples (the
           `param_list`), applies the above check using `calc_eq6_normalize` and
           returns the best (cost, placement) pair found.

    - Return value:
        * `(optimal_cost, optimal_Ia)` where `optimal_cost` is the integer cost
            (number of discs/hops) and `optimal_Ia` is the chosen point (shapely
            Point) achieving that cost. If no better point is found, returns the
            baseline cost with `PF`.

    """

    def integers_between(r1, r2):
        if r1 > r2:
            temp = r1
            r1 = r2
            r2 = temp

        low = math.ceil(r1)
        high = math.floor(r2)
        return list(range(low, high+1))

    def integers_between_minmax_list(r1, r2):
        low = min(math.ceil(r1), math.floor(r2))
        high = max(math.ceil(r1), math.floor(r2))
        return list(range(low, high+1))

    def quartic_below_zero_check(X, A, B, C, D, E):
        formula = A*(X**4) + B*(X**3) + C*(X**2) + D*X + E
        if formula <= 0:
            return True
        else:
            return False

    def calc_eq6_normalize(Pa, Pb, Pc, i, j, k, t, R):

        # Normalize the triangle (rotate/scale) so Pb-Pc lie on a canonical axis and distances scale by `R`
        Pa, Pb, Pc, Rdat = normalize_points((Pa.x, Pa.y), (Pb.x, Pb.y), (Pc.x, Pc.y), R)
        
        # Construct a quartic polynomial
        Xa = Pa[0]
        Ya = Pa[1]

        a = (((j - k) * Rdat) + 1) / 2
        b = a - Xa
        s = a*(1 - a) * Rdat
        c0 = Rdat * (t + 2) - 1
        m = ((2*a) - 1) * Rdat
        A2 = (m**2) + (4*s*Rdat) - (4*(Rdat**2))
        A1 = (2*m*b) + (4*s) + (4*Rdat*c0)
        A0 = (b**2) + (Ya**2) - (c0**2)

        Q4 = A2**2
        Q3 = 2*A2*A1
        Q2 = (A1**2) + (2*A2*A0) - (16*(Ya**2)*s*Rdat)
        Q1 = (2*A1*A0) - (16*(Ya**2)*s)
        Q0 = A0**2

        coefficients = [Q4, Q3, Q2, Q1, Q0]

        # Find roots
        roots = np.roots(coefficients)
        roots_real = [r.real for r in roots if abs(r.imag) < 1e-10]

        # print("roots", roots)
        # print("roots_real", roots_real)
        # print("coefficients", coefficients)

        less_than_zero_interval = []
        candidate_delta_z = []
        substitute_diff = ((j+k)/2) + (1/(2*Rdat))
        for i_ter in range(len(roots_real)-1):
            mid = ((roots_real[i_ter] - roots_real[i_ter+1])/2) + roots_real[i_ter+1]
            if quartic_below_zero_check(mid,*coefficients):
                less_than_zero_interval.append((roots_real[i_ter]-substitute_diff,roots_real[i_ter+1]-substitute_diff))
                candidate_delta_z.extend(integers_between(roots_real[i_ter]-substitute_diff,roots_real[i_ter+1]-substitute_diff))

        # print("less_than_zero_interval",less_than_zero_interval)
        # print("candidate_delta_z",candidate_delta_z)

        Ci, Cj, Ck = None, None, None 
        # print(f"where i = {iF}+{jF}+{kF}-{j}-{k}-2 >> {iF+jF+kF-j-k-2}")
        # print(f"ijk setting {(i,j,k)} R={R}")
        for z in candidate_delta_z:
            Ci = i-(2*z)
            Cj = j+z
            Ck = k+z
            # print(f"Δz = {z} || CV = {(Ci, Cj, Ck)}")
            point_Pj = circle_circle_intersection_nearest(Point(Pb), Point(Pc), Cj*Rdat, Ck*Rdat, Point(Pa))
            if point_Pj:
                Pj_to_Pa = point_Pj.distance(Point(Pa))
                if Pj_to_Pa <= (Ci*Rdat):# and Ci >= 1 and Cj >= 1 and Ck >= 1:
                    # print(f"- ✅ {Pj_to_Pa} <= {Ci*Rdat}")
                    # print(f"- check {Ci+Cj+Ck}={t+t_diff}")
                    circle_circle_intersection_nearest(Point(Pb), Point(Pc), Cj*Rdat, Ck*Rdat, Point(Pa))
                    return Ci, Cj, Ck
                else:
                    # print(f"- ❌ {Pj_to_Pa} <= {Ci*Rdat}")
                    pass

        return None, None, None
    
    # STEP 1: Compute the Fermat point
    iF, jF, kF = calc_ijkF(Pa, Pb, Pc, PF, R)
    t = iF + jF + kF - 2
    cTF = t
    optimal_Ia = PF
    optimal_cost = cTF
    
    # STEP 2: Rotating the terminal points was done by calling this method with
    #         different permutations of (Pa, Pb, Pc) to cover all cases.
    # Adjusting point Pb and Pc
    if kF > jF:
        temp = kF
        kF = jF
        jF = temp
        temp = Pc
        Pc = Pb
        Pb = temp
        # print("adj PbPF and PcPF", jF, kF)
    
    no_intersec = True
    
    param_list = [(iF-2, jF, kF),
                  (iF-3, jF, kF+1),
                  (iF-4, jF, kF+2),
                  (iF-3, jF+1, kF),
                  (iF-1, jF, kF),
                  (iF-2, jF, kF+1),
                  (iF-3, jF, kF+2),
                  (iF-2, jF+1, kF)]
    
    # STEP 2-1: Solve Equation 6 with a set of parameters
    for i, param in enumerate(param_list):
        
        if (i < 4):
            t_diff = 0
        else:
            t_diff = 1
        
        if (param_list[i][0]+param_list[i][1]+param_list[i][2]) == t+t_diff:
            
            # print(f"✅ {param_list[i][0]}+{param_list[i][1]}+{param_list[i][2]} == {t}+{t_diff} ~~ {param_list[i][0]+param_list[i][1]+param_list[i][2]} == {t+t_diff}")
            
            convex_i, convex_j, convex_k = calc_eq6_normalize(Pa, Pb, Pc, param_list[i][0], param_list[i][1], param_list[i][2], t, R)
            
            if convex_i or convex_j or convex_k:
                Pj = circle_circle_intersection_nearest(Pb, Pc, convex_j*R, convex_k*R, Pa)
                if Pj:
                    optimal_cost = t+t_diff-2
                    optimal_Ia = Pj
                    # print("Pj cost",optimal_cost)
                    return optimal_cost, optimal_Ia
        else:
            # print(f"❌ {param_list[i][0]}+{param_list[i][1]}+{param_list[i][2]} == {t}+{t_diff} ~~ {param_list[i][0]+param_list[i][1]+param_list[i][2]} == {t+t_diff}")
            pass
        
    # print("PF cost",optimal_cost)
    return optimal_cost, optimal_Ia