import time

import dolfinx.fem as fem
import dolfinx.mesh as mesh
from mpi4py import MPI
import numpy as np
import pandas as pd
import ufl

import msfem
import msfem_new


def error_norm(u, uh, norm_type="L2"):
    error = ufl.inner(u - uh, u - uh) * ufl.dx
    if norm_type == "H1":
        error += ufl.inner(ufl.grad(u - uh), ufl.grad(u - uh)) * ufl.dx
    return float(np.sqrt(fem.assemble_scalar(fem.form(error))))


def trace_jump_statistics(u):
    """Measure spreads among DG values attached to the same coordinate."""
    coordinates = u.function_space.tabulate_dof_coordinates()
    groups = {}
    for coordinate, value in zip(coordinates, u.x.array):
        key = tuple(np.round(coordinate[:2], 12))
        groups.setdefault(key, []).append(float(np.real(value)))

    jumps = np.array([
        np.ptp(values)
        for values in groups.values()
        if len(values) > 1
    ])
    return float(np.max(jumps)), float(np.sqrt(np.mean(jumps**2)))


def run_case(r, k, method_name):
    refinement_levels = 6 - r
    num_layers = int(k * 2 ** (6 - r))
    coarse_mesh = mesh.create_unit_square(
        MPI.COMM_WORLD,
        2**r,
        2**r,
        cell_type=mesh.CellType.triangle,
        ghost_mode=mesh.GhostMode.none,
    )
    data = msfem.EllipticProblemData(epsilon=5e-2)

    if method_name == "original":
        method = msfem.MsFEM(
            coarse_mesh,
            data,
            refinement_levels,
            num_layers,
        )
    elif method_name == "cutoff_filter":
        method = msfem_new.MsFEM(
            coarse_mesh,
            data,
            refinement_levels,
            num_layers,
            use_filter=True,
            filter_width=0.25,
            filter_order=1.0,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    start = time.perf_counter()
    K_H, F_H = method.offline(is_parabolic=False)
    u_ms = method.online(K_H, F_H)
    elapsed_seconds = time.perf_counter() - start

    u_exact = data.exact_solution(u_ms.function_space)
    max_jump, rms_jump = trace_jump_statistics(u_ms)
    result = {
        "method": method_name,
        "r": r,
        "H": 2.0 ** (-r),
        "k": k,
        "refinement_levels": refinement_levels,
        "num_layers": num_layers,
        "l2_error": error_norm(u_exact, u_ms),
        "h1_error": error_norm(u_exact, u_ms, "H1"),
        "max_coordinate_jump": max_jump,
        "rms_coordinate_jump": rms_jump,
        "elapsed_seconds": elapsed_seconds,
    }

    K_H.destroy()
    F_H.destroy()
    return result


def main():
    records = []
    rs = [2, 3, 4, 5]
    ks = [0.0, 0.5, 1.0, 1.5]
    methods = ["original", "cutoff_filter"]
    num_cases = len(rs) * len(ks) * len(methods)

    for case, (r, k, method_name) in enumerate(
        (
            (r, k, method_name)
            for r in rs
            for k in ks
            for method_name in methods
        ),
        start=1,
    ):
        print(
            f"[{case:02d}/{num_cases}] method={method_name:13s} "
            f"r={r}, k={k:g}",
            flush=True,
        )
        record = run_case(r, k, method_name)
        records.append(record)
        pd.DataFrame.from_records(records).to_csv(
            "filter_comparison_results.csv",
            index=False,
        )
        print(
            f"    L2={record['l2_error']:.6e}, "
            f"H1={record['h1_error']:.6e}, "
            f"jump={record['max_coordinate_jump']:.3e}, "
            f"time={record['elapsed_seconds']:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
