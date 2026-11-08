#!/usr/bin/env python3
"""
=============================================================================
TITLE: Interactive Vacuum Stability CLI for 2HDM + n Singlets
AUTHOR: Omar Moises Asto Rojas
DESCRIPTION:
    Command-line executable script that guides the user through model selection,
    parameter input, potential display, and automated vacuum stability verdicts.
=============================================================================
"""

import sys
from sage.all import show
from stability_core import (
    build_general_potential,
    transform_to_minkowski,
    decompose_potential,
    build_stationarity_equations,
    solve_all_domains,
    analyze_j4_stability,
    fast_local_analysis,
    youngs_local_test,
    generate_perturbed_rays,
    newton_puiseux_analysis
)

def print_prerequisites():
    print("=" * 70)
    print(" SYSTEM PREREQUISITES & ENVIRONMENT CHECK")
    print("=" * 70)
    print(" To run this stability framework successfully, ensure you have:")
    print("   1. SageMath installed (recommended v9.5 or higher).")
    print("   2. Python 3.10+ compatible environment.")
    print("   3. PHCpack library configured in SageMath (optional, for homotopy solver).")
    print("=" * 70)
    print()

def main():
    print_prerequisites()

    # Step 1: Request model specification (2HDM + n singlets)
    try:
        n_input = input("Enter the number of real scalar singlets (n >= 1): ").strip()
        n_singlets = int(n_input)
        if n_singlets < 1:
            raise ValueError
    except ValueError:
        print("[ERROR] Please enter a valid positive integer >= 1.")
        sys.exit(1)

    print(f"\n[INFO] Building General Potential for 2HDM + {n_singlets} Singlet(s)...")
    V_total, total_params = build_general_potential(n_singlets)
    print(f"-> Total independent real parameters in theory: {total_params}")

    # Step 2: Display general symbolic potential
    print("\n" + "=" * 70)
    print(" GENERAL SYMBOLIC SCALAR POTENTIAL (P_ij Basis)")
    print("=" * 70)

    show(V_total)

    # Step 3: Minkowski transformation
    V_K, K_fields, parameters = transform_to_minkowski(V_total, n_singlets)

    # Step 4: Interactive parameter assignment
    print("\n" + "=" * 70)
    print(" PARAMETER CONFIGURATION")
    print("=" * 70)
    print("You can now assign numerical values to specific parameters.")
    print("Unspecified parameters will be automatically set to 0.")
    print("Example input format: {'lam1': 2, 'lamS_1111': 4, 'm11_2': 3, 'm_11': 2, 'tau_1': 1}")
    
    user_eval_choice = input("\nWould you like to use the default benchmark test values? (y/n): ").strip().lower()

    if user_eval_choice == 'y':
        if n_singlets == 1:
            numeric_values = {
                                'lam1': 0.78722,
                                'lam2': 0.78722,
                                'lam3': 0,
                                'lam4': 0,
                                'Re_lam5': -0.78722,
                                'Im_lam5': 0,
                                'Re_lam6': 0,
                                'Im_lam6': 0,
                                'Re_lam7': 0,
                                'Im_lam7': 0,

                                'm11_2': 276680.483352,
                                'm22_2': 51305.343768,
                                'Re_m12': 131194.330848,
                                'Im_m12': 0,

                                'lamS_1111': 0.20,
                                'm_11': 4597.42,
                                'tau_1': 0,
                                'kappa_111': 0,

                                'rho_11_11': 0.010,
                                'rho_22_11': 0.010,
                                'rho_Re12_11': 0,
                                'rho_Im12_11': 0,

                                'sigma_11_1': 0,
                                'sigma_22_1': 0,
                                'sigma_Re12_1': 0,
                                'sigma_Im12_1': 0
                            }
            #numeric_values = {'lam1': 2, 'lam2': 2, 'lam3': 1, 'lam4': 0, 'lamS_1111': 0, 'rho_11_11': 0, 'm11_2': 3, 'm22_2': 3, 'm_11': -2, 'tau_1': 0}
        else:
            numeric_values = {'lam1': 3, 'lam2': 3, 'lam3': 1, 'm11_2': 5, 'm22_2': 5}
        print("-> Loaded benchmark parameters successfully.")
    else:
        print("Enter custom values as a python dictionary string:")
        raw_dict = input("Dictionary: ").strip()
        try:
            numeric_values = eval(raw_dict)
            if not isinstance(numeric_values, dict):
                raise ValueError
        except Exception:
            print("[WARNING] Invalid format. Falling back to empty dict (all parameters = 0).")
            numeric_values = {}

    # Step 5: Potential decomposition and sector extraction
    print("\n[INFO] Decomposing potential into homogeneous sectors (J4, J3, J2, J1)...")
    results = decompose_potential(V_K, n_singlets, numeric_values)
    
    J1, J2, J3, J4, s_vars = results["J1"], results["J2"], results["J3"], results["J4"], results["s"]
    s_vars = results["s"]
    K_vars_correctos = results["K"]  # <-- EXTRACCIÓN CORREGIDA DE K0, K1, K2, K3

    print("\n" + "=" * 70)
    print(" EXTRACTED QUARTIC SECTOR (J4)")
    print("=" * 70)
    show(J4)

    # Step 6: Stationarity analysis & Gröbner resolution
    print("\n[INFO] Building stationarity equations across topological domains...")
    # Usamos K_vars_correctos en lugar de K_fields
    ecuaciones_J4 = build_stationarity_equations(J4, K_vars_correctos, s_vars)

    print("[INFO] Solving stationarity systems using Lexicographical Gröbner Bases...")
    soluciones_J4 = solve_all_domains(ecuaciones_J4, K_vars_correctos, s_vars, method="groebner")
    print("[INFO] The solutions are ...")
    show(soluciones_J4)
    # Step 7: Global Stability Evaluation (J4 verdict)
    print("\n" + "=" * 70)
    print(" GLOBAL STABILITY VERDICT (J4 Analysis)")
    print("=" * 70)
    
    report_J4 = analyze_j4_stability(J4, soluciones_J4, K_vars_correctos, s_vars)


    print("\n" + "=" * 70)
    print(" GLOBAL STABILITY VERDICT")
    print("=" * 70)

    print(report_J4)
    
    if not report_J4['trigger_local_test']:
        if report_J4['status'] == 'STRONGLY_STABLE':
            print(" RESULT: The potential is STRONGLY STABLE.")
            print(" SUMMARY: J4 evaluation alone was sufficient (Global minimum > 0).")
        else:
            print(" RESULT: The potential is UNSTABLE.")
            print(" SUMMARY: J4 evaluation alone was sufficient (Negative direction detected).")
    else:
        print(" RESULT: The potential is MARGINAL (J4 = 0 detected).")
        print(" SUMMARY: Activating Local Analysis (Young's Test)...\n")
        
        zero_points = report_J4['zeros']
        
        print("[INFO] Running Fast Local Analysis...")
        fast_report = fast_local_analysis(J1, J2, J3, J4, zero_points)
        
        if fast_report['status'] == 'UNSTABLE':
            print(" RESULT: The potential is UNSTABLE (Determined by local quadratic behavior).")
        elif fast_report['status'] == 'WEAKLY_STABLE_CANDIDATE':
            print("[INFO] Fast test passed. Verifying completely with Young's Test...")
            young_report = youngs_local_test(J1, J2, J3, J4, zero_points)
            if young_report['status'] == 'YOUNG_CERTIFIED':
                print(" RESULT: The potential is STABLE (Certified by Young's Sufficient Test).")
            else:
                print(" RESULT: INCONCLUSIVE (Advanced Puiseux Ray Analysis Required).")
                
                # ====================================================================
                # NEWTON-PUISEUX ACTIVATION TRIGGER
                # ====================================================================
                print("\n" + "=" * 70)
                print(" NEWTON-PUISEUX ASYMPTOTIC RAY ANALYSIS")
                print("=" * 70)
                print("[INFO] Generating one-parameter diagonal physical perturbations...")
                
                try:
                    rays = generate_perturbed_rays("all", zero_points, J1, J2, J3, J4, K_vars_correctos, s_vars)
                    all_stable = True
                    
                    for i, ray in enumerate(rays, 1):
                        print(f"\n--- Analyzing Ray {i} (Domain: {ray['domain']}) ---")
                        
                        # Call the Newton-Puiseux solver
                        puiseux_report = newton_puiseux_analysis(ray)
                        
                        if puiseux_report['status'] == 'UNSTABLE':
                            print(f"\n RESULT: The potential is UNSTABLE (Determined by Newton-Puiseux on Ray {i}).")
                            all_stable = False
                            break # Halt execution, one unstable ray is enough to kill the potential
                            
                    if all_stable:
                        print("\n RESULT: STABLE AT FIRST ORDER (Or requires higher cascade depth. Marked as stable).")
                        
                except Exception as e:
                    print(f"\n[ERROR] Newton-Puiseux evaluation failed: {e}")

    print("=" * 70)

    # ====================================================================
    # FINITE VACUA ANALYSIS (Stationary Minima of V)
    # ====================================================================
    print("\n" + "=" * 70)
    print(" FINITE VACUA ANALYSIS (Physical Stationary Points)")
    print("=" * 70)

    # Importar las nuevas funciones
    from stability_core import build_finite_vacua_equations, solve_finite_vacua, filter_physical_vacua

    V_numeric = results["V"]

    print(results["V"])

    print("[INFO] Building finite stationarity equations...")
    vacua_eqs = build_finite_vacua_equations(V_numeric, K_vars_correctos, s_vars)

    print("[INFO] Solving exact vacua equations via Gröbner Bases...")
    raw_vacua_sols = solve_finite_vacua(vacua_eqs, K_vars_correctos, s_vars, method="groebner")

    print("[INFO] Filtering for physical domain constraints (K0 > 0, KgK >= 0)...")
    physical_vacua = filter_physical_vacua(raw_vacua_sols, K_vars_correctos)

    total_physical = sum(len(lst) for lst in physical_vacua.values())

    if total_physical == 0:
        print("\n RESULT: No non-trivial physical finite vacua found (Only the origin might exist).")
    else:
        print(f"\n RESULT: Found {total_physical} physical finite vacua.")
        for domain, sols in physical_vacua.items():
            if sols:
                print(f"\n--- {domain.upper()} VACUA ---")
                for idx, sol in enumerate(sols, 1):
                    depth = float(V_numeric.subs(sol))
                    print(f" Vacuum {idx}:")
                    for k, v in sol.items():
                        print(f"   {k} = {v}")
                    print(f"   Potential Depth V(min) = {depth}")

if __name__ == "__main__":
    main()
