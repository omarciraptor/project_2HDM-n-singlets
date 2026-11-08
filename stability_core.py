"""
=============================================================================
TITLE: Autonomous Vacuum Stability Framework for 2HDM + n Real Singlets
AUTHOR: Omar Moises Asto Rojas
DESCRIPTION: 
    Core algebraic and numerical engine using SageMath for analyzing the 
    asymptotic vacuum stability (bounded-from-below) of extended Higgs 
    potentials via Minkowski space embedding, topological domain decomposition, 
    Lagrange multipliers, and Newton-Puiseux local expansions.
=============================================================================
"""

from sage.all import *
from itertools import product
import math
from sage.interfaces.phc import phc
from sage.rings.complex_double import CDF


# =============================================================================
# 1. SYMMETRIC TENSOR VARIABLE CREATION
# =============================================================================

def create_symmetric_var(base_name, indices):
    """
    Creates symbolic variables with symmetric indices to handle permutation 
    invariance in multi-singlet self-interactions (e.g., m_ab = m_ba).
    
    Args:
        base_name (str): Prefix string for the variable (e.g., 'm', 'kappa', 'lamS').
        indices (tuple/list): Indices to append to the base name.
        
    Returns:
        Expression: A SageMath symbolic variable.
    """
    indices = tuple(sorted(indices))

    if any(i >= 10 for i in indices):
        suffix = "_".join(str(i) for i in indices)
    else:
        suffix = "".join(str(i) for i in indices)

    return var(f"{base_name}_{suffix}")


# =============================================================================
# 2. GENERAL POTENTIAL BUILDER (2HDM + n REAL SINGLETS)
# =============================================================================

def build_general_potential(n_singlets):
    """
    Constructs the most general renormalizable scalar potential for the 
    Two-Higgs-Doublet Model (2HDM) extended with 'n' real scalar singlets.
    
    Constraint:
        n_singlets >= 1
        
    Returns:
        tuple: (V_total, total_params)
            - V_total: Full symbolic potential expression.
            - total_params: Exact count of independent real parameters.
    """
    if (
        not isinstance(n_singlets, (int, Integer))
        or isinstance(n_singlets, bool)
        or n_singlets < 1
    ):
        raise ValueError("n_singlets must be an integer greater than or equal to 1.")

    n = int(n_singlets)

    # Gauge-invariant doublet bilinear fields
    P11, P22, P12, P21 = var("P11 P22 P12 P21")

    # Real singlet scalar fields
    s = [var(f"s_{a}") for a in range(1, n + 1)]

    # Parameter counting breakdown
    params_2hdm = 14
    params_singlet = (
        n
        + math.comb(n + 1, 2)
        + math.comb(n + 2, 3)
        + math.comb(n + 3, 4)
    )
    params_portal = 4 * n + 4 * math.comb(n + 1, 2)
    total_params = params_2hdm + params_singlet + params_portal

    # --- 2HDM Sector ---
    m11_2, m22_2 = var("m11_2 m22_2")
    lam1, lam2, lam3, lam4 = var("lam1 lam2 lam3 lam4")
    Re_m12, Im_m12 = var("Re_m12 Im_m12")
    Re_lam5, Im_lam5 = var("Re_lam5 Im_lam5")
    Re_lam6, Im_lam6 = var("Re_lam6 Im_lam6")
    Re_lam7, Im_lam7 = var("Re_lam7 Im_lam7")

    m12 = Re_m12 + I * Im_m12
    m12_c = Re_m12 - I * Im_m12
    lam5 = Re_lam5 + I * Im_lam5
    lam5_c = Re_lam5 - I * Im_lam5
    lam6 = Re_lam6 + I * Im_lam6
    lam6_c = Re_lam6 - I * Im_lam6
    lam7 = Re_lam7 + I * Im_lam7
    lam7_c = Re_lam7 - I * Im_lam7

    V_2HDM = (
        m11_2 * P11
        + m22_2 * P22
        - (m12 * P12 + m12_c * P21)
        + QQ(1) / 2 * lam1 * P11**2
        + QQ(1) / 2 * lam2 * P22**2
        + lam3 * P11 * P22
        + lam4 * P12 * P21
        + QQ(1) / 2 * (lam5 * P12**2 + lam5_c * P21**2)
        + (lam6 * P11 * P12 + lam6_c * P11 * P21)
        + (lam7 * P22 * P12 + lam7_c * P22 * P21)
    )

    # --- Pure Singlet Sector ---
    V_singlet_lin = sum(var(f"tau_{a}") * s[a - 1] for a in range(1, n + 1))
    V_singlet_quad = sum(
        QQ(1) / 2 * create_symmetric_var("m", (a, b)) * s[a - 1] * s[b - 1]
        for a, b in product(range(1, n + 1), repeat=2)
    )
    V_singlet_cubic = sum(
        QQ(1) / 6 * create_symmetric_var("kappa", (a, b, c)) * s[a - 1] * s[b - 1] * s[c - 1]
        for a, b, c in product(range(1, n + 1), repeat=3)
    )
    V_singlet_quartic = sum(
        QQ(1) / 4 * create_symmetric_var("lamS", (a, b, c, d)) * s[a - 1] * s[b - 1] * s[c - 1] * s[d - 1]
        for a, b, c, d in product(range(1, n + 1), repeat=4)
    )

    V_singlet = V_singlet_lin + V_singlet_quad + V_singlet_cubic + V_singlet_quartic

    # --- Portal Sector (Doublet-Singlet Interactions) ---
    Re_P12 = (P12 + P21) / 2
    Im_P12 = (P12 - P21) / (2 * I)
    doublet_bases = [P11, P22, Re_P12, Im_P12]
    basis_names = ["11", "22", "Re12", "Im12"]

    V_portal_cubic = 0
    V_portal_quartic = 0

    for basis_name, base in zip(basis_names, doublet_bases):
        V_portal_cubic += sum(
            QQ(1) / 2 * var(f"sigma_{basis_name}_{a}") * s[a - 1] * base
            for a in range(1, n + 1)
        )
        V_portal_quartic += sum(
            QQ(1) / 2 * create_symmetric_var(f"rho_{basis_name}", (a, b)) * s[a - 1] * s[b - 1] * base
            for a, b in product(range(1, n + 1), repeat=2)
        )

    V_portal = V_portal_cubic + V_portal_quartic
    V_total = V_2HDM + V_singlet + V_portal

    return V_total, total_params


# =============================================================================
# 3. MINKOWSKI BASIS MAPPING
# =============================================================================

def transform_to_minkowski(potential, n_singlets):
    """
    Maps the potential from the bilinear P_ij basis to the Minkowski 4-vector 
    basis (K_mu), projecting the doublet orbit space onto the forward light cone (LC+).
    """
    P11, P22, P12, P21 = var("P11 P22 P12 P21")
    K0, K1, K2, K3 = var("K0 K1 K2 K3")
    singlets = [var(f"s_{a}") for a in range(1, n_singlets + 1)]

    candidate_fields = [P11, P22, P12, P21] + singlets
    potential_vars = list(potential.variables())
    fields = [x for x in candidate_fields if x in potential_vars]
    parameters = [x for x in potential_vars if x not in fields]

    transformation_rules = {
        P11: (K0 + K3) / 2,
        P22: (K0 - K3) / 2,
        P12: (K1 + I * K2) / 2,
        P21: (K1 - I * K2) / 2,
    }

    V_minkowski = potential.subs(transformation_rules)
    V_minkowski = expand(V_minkowski).simplify_full()

    return V_minkowski, fields, parameters


# =============================================================================
# 4. HOMOGENEOUS SECTOR DECOMPOSITION (t -> infinity limit)
# =============================================================================

def decompose_potential(potential, n_singlets, numeric_values):
    """
    Evaluates numerical parameters (assigning zero to unspecified couplings) 
    and extracts homogeneous field sectors (J4, J3, J2, J1) via asymptotic scaling.
    """
    n = int(n_singlets)
    if numeric_values is None:
        numeric_values = {}

    K0, K1, K2, K3 = var("K0 K1 K2 K3")
    K_fields = [K0, K1, K2, K3]
    singlet_fields = [var(f"s_{a}") for a in range(1, n + 1)]
    fields = K_fields + singlet_fields

    potential = SR(potential)
    potential_vars = list(potential.variables())
    parameters = sorted([v for v in potential_vars if v not in fields], key=str)

    param_rules = {}
    for param in parameters:
        if param in numeric_values:
            val = numeric_values[param]
        elif str(param) in numeric_values:
            val = numeric_values[str(param)]
        else:
            val = 0
        param_rules[param] = SR(val)

    V_numeric = potential.subs(param_rules).expand().simplify_full()
    
    t = var("t")
    scaling_rules = {f: t**2 * f for f in K_fields}
    for f in singlet_fields:
        scaling_rules[f] = t * f

    V_scaled = V_numeric.subs(scaling_rules).expand()

    def extract_sector(expression, degree):
        deriv = diff(expression, t, degree).subs({t: 0})
        return (deriv / factorial(degree)).expand().simplify_full()

    J4 = extract_sector(V_scaled, 4)
    J3 = extract_sector(V_scaled, 3)
    J2 = extract_sector(V_scaled, 2)
    J1 = extract_sector(V_scaled, 1)

    return {
        "V": V_numeric,
        "J4": J4,
        "J3": J3,
        "J2": J2,
        "J1": J1,
        "parameters": param_rules,
        "K": K_fields,
        "s": singlet_fields,
    }


# =============================================================================
# 5. STATIONARITY EQUATIONS BY TOPOLOGICAL DOMAIN
# =============================================================================

def build_stationarity_equations(J_expr, K_vars, s_vars):
    """
    Builds Lagrange multiplier stationarity equations across topological domains 
    (Timelike interior, Lightlike boundary, and Tip of the cone).
    """
    K_vars = list(K_vars)
    s_vars = list(s_vars)
    J_expr = SR(J_expr).expand()
    K0, K1, K2, K3 = K_vars

    lam_N, lam_K, lam_S = var("lam_N lam_K lam_S")
    K_sq = K0**2 - K1**2 - K2**2 - K3**2
    s_sq = sum(s**2 for s in s_vars)
    norm_constraint = K0 + s_sq - 1
    all_fields = K_vars + s_vars

    # Timelike domain
    L_timelike = J_expr - lam_N * norm_constraint
    eqs_timelike = [diff(L_timelike, f).expand() for f in all_fields]
    eqs_timelike.append(norm_constraint)

    # Lightlike domain
    L_lightlike = J_expr - lam_N * norm_constraint - lam_K * K_sq
    eqs_lightlike = [diff(L_lightlike, f).expand() for f in all_fields]
    eqs_lightlike.extend([norm_constraint, K_sq])

    # Tip domain
    tip_rules = {K0: ZZ(0), K1: ZZ(0), K2: ZZ(0), K3: ZZ(0)}
    J_tip = J_expr.subs(tip_rules).expand()
    tip_constraint = s_sq - 1
    L_tip = J_tip - lam_S * tip_constraint
    eqs_tip = [diff(L_tip, s).expand() for s in s_vars]
    eqs_tip.append(tip_constraint)

    return {
        "timelike": eqs_timelike,
        "lightlike": eqs_lightlike,
        "tip": eqs_tip,
    }


# =============================================================================
# 6. SOLVERS & FILTERS
# =============================================================================

def filter_real_solutions(solutions, tol=1e-6):
    """Filters numerical solutions, retaining only strictly real physical roots."""
    filtered = []
    for sol in solutions:
        clean_sol = {}
        discard = False
        for k, v in sol.items():
            val = complex(v)
            if abs(val.imag) > tol:
                discard = True
                break
            clean_sol[k] = val.real
        if not discard:
            filtered.append(clean_sol)
    return filtered


def solve_domain_safely(equations, variables, tol=1e-6, domain_name="Unknown", method="groebner"):
    """Safely solves polynomial equations using Lexicographical Gröbner bases."""
    try:
        if method == "groebner":
            R = PolynomialRing(QQ, names=[str(v) for v in variables], order="lex")
            

            polys = [R(SR(eq)) for eq in equations]
            I = R.ideal(polys)
            
            if I.dimension() != 0:
                return []
            

            raw_solutions = I.variety(ring=AA)
            
        else:
            R = PolynomialRing(QQ, names=[str(v) for v in variables])
            sol = phc.blackbox(equations, R)
            raw_solutions = sol.solution_dicts()

        return filter_real_solutions(raw_solutions, tol=tol)
        
    except Exception as e:

        print(f"[WARNING] Solver exception in {domain_name} domain: {e}")
        return []


def solve_all_domains(equations_dict, K_vars, s_vars, method="groebner"):
    """Unions real stationary solutions across all domains."""
    lam_N, lam_K, lam_S = var("lam_N lam_K lam_S")
    soluciones = {"timelike": [], "lightlike": [], "tip": []}

    def normalize(sol, variables):
        nueva = {}
        for var_sym in variables:
            target = str(var_sym)
            for k, val in sol.items():
                if str(k) == target:
                    nueva[var_sym] = val
                    break
        return nueva

    # Timelike
    sols_tl = solve_domain_safely(equations_dict["timelike"], K_vars + s_vars + [lam_N], "Timelike", method)
    for s in sols_tl:
        soluciones["timelike"].append(normalize(s, K_vars + s_vars))

    # Lightlike
    sols_ll = solve_domain_safely(equations_dict["lightlike"], K_vars + s_vars + [lam_N, lam_K], "Lightlike", method)
    for s in sols_ll:
        soluciones["lightlike"].append(normalize(s, K_vars + s_vars))

    # Tip
    sols_tip = solve_domain_safely(equations_dict["tip"], s_vars + [lam_S], "Tip", method)
    for s in sols_tip:
        nueva = normalize(s, s_vars)
        for K in K_vars:
            nueva[K] = ZZ(0)
        soluciones["tip"].append(nueva)

    return soluciones


def solve_all_domains(equations_dict, K_vars, s_vars,
                      method="groebner"):
    """Unions real stationary solutions across all domains."""

    # ========================================================
    # 1. VARIABLES
    # ========================================================

    K_vars = list(K_vars)
    s_vars = list(s_vars)

    lam_N, lam_K, lam_S = var("lam_N lam_K lam_S")

    soluciones = {"timelike": [], "lightlike": [], "tip": []}


    def normalize(sol, variables):
        nueva = {}
        for var_sym in variables:
            target = str(var_sym)
            for k, val in sol.items():
                if str(k) == target:
                    nueva[var_sym] = val
                    break
        return nueva

    # Timelike
    sols_tl = solve_domain_safely(equations_dict["timelike"], K_vars + s_vars + [lam_N], domain_name="Timelike", method=method)
    for s in sols_tl:
        soluciones["timelike"].append(normalize(s, K_vars + s_vars))

    # Lightlike
    sols_ll = solve_domain_safely(equations_dict["lightlike"], K_vars + s_vars + [lam_N, lam_K], domain_name="Lightlike", method=method)
    for s in sols_ll:
        soluciones["lightlike"].append(normalize(s, K_vars + s_vars))

    # Tip
    sols_tip = solve_domain_safely(equations_dict["tip"], s_vars + [lam_S], domain_name="Tip", method=method)

    for s in sols_tip:
        nueva = normalize(s, s_vars)
        for K in K_vars:
            nueva[K] = ZZ(0)
        soluciones["tip"].append(nueva)

    return soluciones






# =============================================================================
# 7. GLOBAL STABILITY ANALYSIS (J4 Evaluation)
# =============================================================================

def analyze_j4_stability(J4, solutions, K_vars, s_vars, tol=1e-10):
    """Evaluates J4 across stationary points to classify global stability."""
    fields = list(K_vars) + list(s_vars)
    j4_values = {"timelike": [], "lightlike": [], "tip": []}
    all_values = []
    has_negatives = False
    has_zeros = False
    zero_points = []

    for domain, sol_list in solutions.items():
        for sol in sol_list:
            value = SR(J4).subs({v: sol[v] for v in fields}).simplify_full()
            num = float(value)
            print("J4 value is:")
            print(value)

            if num < -tol:
                sign = "NEGATIVE"
                has_negatives = True
            elif abs(num) <= tol:
                sign = "ZERO"
                has_zeros = True
                zero_points.append({"solucion": sol, "domain": domain})
            else:
                sign = "POSITIVE"

            record = {"solution": sol, "j4_value": value, "sign": sign, "domain": domain}
            j4_values[domain].append(record)
            all_values.append((num, value))

    if not all_values:
        raise ValueError("No stationary solutions found to evaluate J4.")

    
    min_j4 = min(all_values, key=lambda x: x[0])[1]

    if has_negatives:
        status = "UNSTABLE"
    elif has_zeros:
        status = "LOCAL_ANALYSIS"
    else:
        status = "STRONGLY_STABLE" 

    return {
        "status": status,
        "j4_values": j4_values,
        "min_j4": min_j4,
        "trigger_local_test": status == "LOCAL_ANALYSIS",
        "zeros": zero_points
    }

# =============================================================================
# 8. LOCAL ANALYSIS (Fast Verification, Young Test & Rays)
# =============================================================================

def fast_local_analysis(J1, J2, J3, J4, zero_points, tol=1e-10):
    """
    First local analysis verification. Checks if J2(p) > 0 at marginal points.
    """

    # Step 1: Check whether the odd sectors vanish identically.
    if SR(J1).expand() != 0 or SR(J3).expand() != 0:
        return {"status": "CONTINUE_ANALYSIS", "evaluations": []}

    if not zero_points:
        raise ValueError("No quartic zero points were provided.")

    evaluations = []

    # Step 2: Evaluate the quadratic sector at each quartic zero.
    for i, record in enumerate(zero_points, start=1):

        point = record.get("solucion", record.get("solution", record))
        domain = record.get("domain", record.get("dominio", "unknown"))

        J4_value = SR(J4).subs(point).simplify_full()
        J2_value = SR(J2).subs(point).simplify_full()

        if J4_value.variables() or J2_value.variables():
            raise ValueError(f"Incomplete substitution at point {i}.")

        if abs(float(J4_value)) > tol:
            raise ValueError(f"Point {i} does not satisfy J4 = 0.")

        if float(J2_value) < -tol:
            status = "UNSTABLE"
        elif float(J2_value) > tol:
            status = "POSITIVE"
        else:
            status = "CONTINUE_ANALYSIS"

        evaluations.append({
            "point": point,
            "domain": domain,
            "J4": J4_value,
            "J2": J2_value,
            "status": status
        })

        print(f"\nPoint {i} ({domain})")
        print(f"  J4 = {J4_value}")
        print(f"  J2 = {J2_value}")
        print(f"  Status = {status}")

    # Step 3: Determine the overall result.
    statuses = [entry["status"] for entry in evaluations]

    if "UNSTABLE" in statuses:
        global_status = "UNSTABLE"
    elif all(status == "POSITIVE" for status in statuses):
        global_status = "WEAKLY_STABLE"
    else:
        global_status = "CONTINUE_ANALYSIS"

    print("\nFAST LOCAL ANALYSIS:", global_status)

    return {
        "status": global_status,
        "evaluations": evaluations
    }


def youngs_local_test(J1, J2, J3, J4, zero_points, tol=1e-10):
    """
    Young's sufficient algebraic test for stability at marginal directions.
    """

    if not zero_points:
        raise ValueError("No quartic zero points provided.")

    J1, J2, J3, J4 = map(SR, (J1, J2, J3, J4))

    # Check whether an expression vanishes identically.
    def is_zero(f):
        return f.expand() == 0

    # Verify polynomial divisibility over the rational field.
    def divisible(numerator, denominator):
        if is_zero(numerator):
            return True

        try:
            variables = sorted(
                set(numerator.variables())
                | set(denominator.variables()),
                key=str
            )

            R = PolynomialRing(
                QQ, names=[str(v) for v in variables]
            )

            P = R(numerator.expand())
            Q = R(denominator.expand())

            return Q != 0 and P.reduce([Q]) == 0

        except (TypeError, ValueError, ArithmeticError):
            return False

    # Test the cubic and linear ratios.
    bound_R3 = (
        is_zero(J3) or divisible(J3**4, J4**3)
    )

    bound_R1 = (
        is_zero(J1) or divisible(J1**4, J4)
    )

    results = []

    for i, record in enumerate(zero_points, start=1):

        # Support the output format used by report_J4.
        point = record.get(
            "solucion",
            record.get("solution", record)
        )

        domain = record.get(
            "domain",
            record.get("dominio", "unknown")
        )

        # Substitute only field variables.
        rules = {
            v: val for v, val in point.items()
            if not isinstance(v, str)
        }

        val_J4 = J4.subs(rules).simplify_full()
        val_J2 = J2.subs(rules).simplify_full()

        if val_J4.variables() or val_J2.variables():
            raise ValueError(
                f"Incomplete field substitution at point {i}."
            )

        a4 = float(val_J4)
        a2 = float(val_J2)

        if abs(a4) > tol:
            raise ValueError(
                f"Point {i} does not satisfy J4 = 0."
            )

        # If J2(p) > 0, its negative part vanishes
        # throughout a sufficiently small neighborhood.
        if a2 > tol:
            bound_R2 = True
        else:
            bound_R2 = (
                is_zero(J2) or divisible(J2**2, J4)
            )

        certified = bound_R3 and bound_R2 and bound_R1

        status = (
            "YOUNG_CERTIFIED"
            if certified else "CONTINUE_ANALYSIS"
        )

        results.append({
            "point": point,
            "domain": domain,
            "J4": val_J4,
            "J2": val_J2,
            "R3_bounded": bound_R3,
            "R2_bounded": bound_R2,
            "R1_bounded": bound_R1,
            "status": status
        })

        print(f"\nPoint {i} ({domain})")
        print("  J4 =", val_J4)
        print("  J2 =", val_J2)
        print("  R3 bounded:", bound_R3)
        print("  R2 bounded:", bound_R2)
        print("  R1 bounded:", bound_R1)
        print("  Status:", status)

    global_status = (
        "YOUNG_CERTIFIED"
        if all(r["status"] == "YOUNG_CERTIFIED"
               for r in results)
        else "CONTINUE_ANALYSIS"
    )

    print("\nYOUNG TEST:", global_status)

    return {
        "status": global_status,
        "evaluations": results
    }


def generate_perturbed_rays(domain, zero_points, J1, J2, J3, J4, K_vars=None, s_vars=None):
    """
    Generates local parameterizations using EXACTLY ONE generic perturbation 
    parameter 'x' that excites all directions simultaneously to avoid blind spots.
    """
    if K_vars is None:
        K_vars = list(var("K0 K1 K2 K3"))
    else:
        K_vars = list(K_vars)

    points = []
    if isinstance(zero_points, dict):
        for dom, lst in zero_points.items():
            for p in lst:
                points.append((dom, p))
    else:
        for p in zero_points:
            dom = p.get("domain", p.get("dominio", domain))
            points.append((dom, p))

    if domain != "all":
        points = [(d, p) for d, p in points if d == domain]

    if not points:
        raise ValueError("No points found for the requested domain.")

    if s_vars is None:
        example = points[0][1]
        example = example.get("solution", example.get("solucion", example))
        s_vars = sorted([k for k in example if str(k).startswith("s_")], key=str)
    else:
        s_vars = list(s_vars)

    K0, K1, K2, K3 = K_vars
    fields = K_vars + s_vars

    def dot_product(a, b):
        return sum(x*y for x, y in zip(a, b))

    def norm2(a):
        return dot_product(a, a)

    results = []
    for index, (dom, record) in enumerate(points, 1):
        point = record.get("solution", record.get("solucion", record))
        p = {k: SR(point[k]) for k in fields}
        pK = [p[k] for k in K_vars]
        ps = [p[s] for s in s_vars]

        rules = {}
        
        # A SINGLE GENERIC PARAMETER
        x = var(f"x_{index}")
        parameters = [x]
        constraints = []

        if dom == "timelike":
            # Perturb the singlet with x
            ss = [ps[a] + x for a in range(len(ps))]
            k0 = 1 - norm2(ss)
            
            # Perturb the spatial proportions with x
            r0 = [pK[a]/pK[0] for a in range(1, 4)]
            rules[K0] = k0
            for a, K in enumerate(K_vars[1:]):
                rules[K] = k0 * (r0[a] + x)
            for a, s in enumerate(s_vars):
                rules[s] = ss[a]

        elif dom == "lightlike":
            # Perturb the singlet
            ss = [ps[a] + x for a in range(len(ps))]
            k0 = 1 - norm2(ss)
            
            # Perturb the rational sphere with the same parameter x across all angles
            n0 = [pK[a]/pK[0] for a in range(1, 4)]
            m = len(n0)
            j = max(range(m), key=lambda k: abs(float(n0[k])))
            indices = [k for k in range(m) if k != j]
            v_vec = [SR(0)] * m
            for k in indices:
                for h in range(m):
                    v_vec[h] += x*((1 if h == k else 0) - n0[k]*n0[h])
            q_val = norm2(v_vec)
            nvec = [((1-q_val)*n0[h] + 2*v_vec[h])/(1+q_val) for h in range(m)]
            
            # Use x^2 for the depth (d) to ensure K0^2 - K^2 >= 0
            rules[K0] = k0
            for a, K in enumerate(K_vars[1:]):
                rules[K] = k0 * (1 - x**2) * nvec[a]
            for a, s in enumerate(s_vars):
                rules[s] = ss[a]

        elif dom == "tip":
            m = len(ps)
            if m == 1:
                sphere_s = [1] 
            else:
                j = max(range(m), key=lambda k: abs(float(ps[k])))
                indices = [k for k in range(m) if k != j]
                v_vec = [SR(0)] * m
                for k in indices:
                    for h in range(m):
                        v_vec[h] += x*((1 if h == k else 0) - ps[k]*ps[h])
                q_val = norm2(v_vec)
                sphere_s = [((1-q_val)*ps[h] + 2*v_vec[h])/(1+q_val) for h in range(m)]

            factor_s = (1 - x**2)/(1 + x**2)
            k0 = 4*x**2 / (1+x**2)**2
            rules[K0] = k0
            for K in K_vars[1:]:
                rules[K] = k0 * x
            for a, s in enumerate(s_vars):
                rules[s] = factor_s * sphere_s[a]
        else:
            raise ValueError(f"Unknown domain: {dom}")

        J_pert = {
            "J1": SR(J1).subs(rules).simplify_full(),
            "J2": SR(J2).subs(rules).simplify_full(),
            "J3": SR(J3).subs(rules).simplify_full(),
            "J4": SR(J4).subs(rules).simplify_full()
        }
        
        u_as = var(f"q_as_{index}")
        Phi = J_pert["J4"] + u_as*J_pert["J3"] + u_as**2*J_pert["J2"] + u_as**3*J_pert["J1"]

        results.append({
            "domain": dom,
            "point": p,
            "trajectory": rules,
            "parameters": parameters,
            "constraints": constraints,
            "perturbed_sectors": J_pert,
            "asymptotic_variable": u_as,
            "Phi": Phi
        })

    return results

import sympy as sp

def newton_puiseux_analysis(chart):
    """
    Compute leading Newton-Puiseux balances for a one-parameter physical perturbation.
    """
    if not chart["parameters"]:
        return {"status": "CONTINUE_ANALYSIS", "branches": []}

    # Read the primary perturbation parameter
    x_name = str(chart["parameters"][0])
    q_name = str(chart["asymptotic_variable"])

    x, q, c, z = sp.symbols("x q c z", real=True)

    phi = sp.sympify(
        str(chart["Phi"]).replace("^", "**"),
        locals={x_name: x, q_name: q}
    )

    # 1. BASE RAY VERIFICATION (x = 0)
    # If the unperturbed direction is already unstable, Newton-Puiseux is not needed.
    phi_base = sp.cancel(phi.subs(x, 0))
    if phi_base != 0 and phi_base.has(q):
        # Extract the term with the lowest degree in q
        q_poly = sp.Poly(phi_base, q)
        min_q = min(k[0] for k, _ in q_poly.terms())
        leading_coeff = q_poly.nth(min_q)
        if leading_coeff.is_negative and min_q < 4:
            print("\nNEWTON-PUISEUX RESULT: UNSTABLE (Base ray is unbounded)")
            return {"status": "UNSTABLE", "branches": []}

    # 2. STATIONARY POLYNOMIAL CONSTRUCTION
    F = sp.cancel(sp.diff(phi, x))
    if F == 0:
        return {"status": "CONTINUE_ANALYSIS", "branches": []}

    numerator = sp.fraction(F)[0]
    P = sp.Poly(numerator, x, q, domain=sp.QQ)

    print("Stationary polynomial:")
    print(P.as_expr())

    terms = [(powers[0], powers[1], coeff) for powers, coeff in P.terms()]
    slopes = set()

    for i, (a, b, _) in enumerate(terms):
        for A, B, _ in terms[i+1:]:
            if a != A:
                alpha = sp.Rational(B-b, a-A)
                if alpha > 0:
                    slopes.add(alpha)

    print("\nCandidate exponents:", sorted(slopes))

    branches = []

    # If no slopes exist (e.g., P = 4*x**3), the asymptotic minimum stays at x=0.
    if not slopes:
        print("\nNo Newton exponents found. Minimum stays at origin.")
        # Stability was already verified in step 1 (phi_base)
        return {"status": "CONTINUE_ANALYSIS", "branches": []}

    # 3. BRANCH EVALUATION
    for alpha in sorted(slopes):
        weights = [b + alpha*a for a, b, _ in terms]
        minimum = min(weights)

        face = [
            (a, coeff)
            for (a, b, coeff), weight in zip(terms, weights)
            if weight == minimum
        ]

        if len(face) < 2:
            continue

        initial = sp.Poly(sum(coeff*c**a for a, coeff in face), c)

        print("\nNewton exponent:", alpha)
        print("Initial equation:", initial.as_expr())

        for root in sp.real_roots(initial):
            if root == 0:
                continue

            if chart["domain"] == "tip" and not root.is_positive:
                continue

            m, n = int(alpha.p), int(alpha.q)
            phi_z = sp.cancel(phi.subs({x: root*z**m, q: z**n}))
            num, den = sp.fraction(phi_z)

            if num == 0:
                continue

            A_poly = sp.Poly(num, z)
            B_poly = sp.Poly(den, z)

            kA = min(k[0] for k, _ in A_poly.terms())
            kB = min(k[0] for k, _ in B_poly.terms())

            coefficient = sp.simplify(A_poly.nth(kA) / B_poly.nth(kB))
            gamma = sp.Rational(kA-kB, n)

            status = (
                "UNSTABLE"
                if coefficient.is_negative and gamma < 4
                else "CONTINUE_ANALYSIS"
            )

            branches.append({
                "alpha": alpha,
                "c": root,
                "gamma": gamma,
                "coefficient": coefficient,
                "status": status
            })

    overall = "UNSTABLE" if any(b["status"] == "UNSTABLE" for b in branches) else "CONTINUE_ANALYSIS"

    print("\nNEWTON-PUISEUX RESULT:", overall)
    return {"status": overall, "branches": branches}


# =============================================================================
# 9. FINITE VACUA STATIONARITY (Physical Minima)
# =============================================================================

def build_finite_vacua_equations(V_expr, K_vars, s_vars):
    """
    Builds the exact stationarity equations for finite vacua across all physical 
    domains (Timelike, Lightlike, Tip) without scale-fixing constraints.
    Ref: Equations (4.1), (4.2), and (4.3).
    """

    # V_expr = (K0 - 1)**2 + K1**2 + 2*K2**2 + 3*K3**2+ (s_1**2 - 1)**2
    K_vars = list(K_vars)
    s_vars = list(s_vars)
    V_expr = SR(V_expr).expand()
    K0, K1, K2, K3 = K_vars

    lam_K = var("lam_K")
    K_sq = K0**2 - K1**2 - K2**2 - K3**2
    all_fields = K_vars + s_vars

    # (4.1) Timelike interior: No active constraints
    eqs_timelike = [diff(V_expr, f).expand() for f in all_fields]

    # (4.2) Lightlike boundary: Constraint KgK = 0 (using Lagrange multiplier lam_K)
    # L = V - lam_K * (K0^2 - K1^2 - K2^2 - K3^2)
    L_lightlike = V_expr - lam_K * K_sq
    eqs_lightlike = [diff(L_lightlike, f).expand() for f in all_fields]
    eqs_lightlike.append(K_sq)

    # (4.3) Tip of the cone: K_mu = 0
    tip_rules = {K0: ZZ(0), K1: ZZ(0), K2: ZZ(0), K3: ZZ(0)}
    V_tip = V_expr.subs(tip_rules).expand()
    eqs_tip = [diff(V_tip, s).expand() for s in s_vars]

    return {
        "timelike": eqs_timelike,
        "lightlike": eqs_lightlike,
        "tip": eqs_tip,
    }

def solve_finite_vacua(equations_dict, K_vars, s_vars, method="groebner", tol=1e-6):
    """
    Solves the finite vacua equations and filters only strictly real roots.
    """
    lam_K = var("lam_K")
    solutions = {"timelike": [], "lightlike": [], "tip": []}

    def normalize(sol, variables):
        new_sol = {}
        for var_sym in variables:
            target = str(var_sym)
            for k, val in sol.items():
                if str(k) == target:
                    new_sol[var_sym] = val
                    break
        return new_sol

    # Solve Timelike
    sols_tl = solve_domain_safely(equations_dict["timelike"], K_vars + s_vars, tol=tol, domain_name="Vacua_Timelike", method=method)
    for s in sols_tl:
        solutions["timelike"].append(normalize(s, K_vars + s_vars))

    # Solve Lightlike
    sols_ll = solve_domain_safely(equations_dict["lightlike"], K_vars + s_vars + [lam_K], tol=tol, domain_name="Vacua_Lightlike", method=method)
    for s in sols_ll:
        solutions["lightlike"].append(normalize(s, K_vars + s_vars))

    # Solve Tip
    sols_tip = solve_domain_safely(equations_dict["tip"], s_vars, tol=tol, domain_name="Vacua_Tip", method=method)
    for s in sols_tip:
        new_sol = normalize(s, s_vars)
        for K in K_vars:
            new_sol[K] = ZZ(0)
        solutions["tip"].append(new_sol)

    return solutions

def filter_physical_vacua(solutions_dict, K_vars, tol=1e-8):
    """
    Enforces the geometric physical constraints on the solved mathematical roots.
    - Timelike: K0 > 0 and K0^2 - K_vec^2 > 0
    - Lightlike: K0 > 0 (K0^2 - K_vec^2 = 0 is guaranteed by the solver)
    - Tip: K_mu = 0 (guaranteed)
    """
    K0, K1, K2, K3 = K_vars
    physical_vacua = {"timelike": [], "lightlike": [], "tip": solutions_dict["tip"]}

    # Filter Timelike solutions
    for sol in solutions_dict["timelike"]:
        k0_val = float(sol.get(K0, 0))
        k_sq_val = k0_val**2 - float(sol.get(K1, 0))**2 - float(sol.get(K2, 0))**2 - float(sol.get(K3, 0))**2
        
        # Condition: K0 > 0 AND KgK > 0
        if k0_val > tol and k_sq_val > tol:
            physical_vacua["timelike"].append(sol)

    # Filter Lightlike solutions
    for sol in solutions_dict["lightlike"]:
        k0_val = float(sol.get(K0, 0))
        
        # Condition: K0 > 0 (KgK = 0 is already part of the equation system)
        if k0_val > tol:
            physical_vacua["lightlike"].append(sol)

    return physical_vacua