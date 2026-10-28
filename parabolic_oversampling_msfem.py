"""Petrov-Galerkin MsFEM with oversampled parabolic correctors.

The local corrector is computed on an oversampling patch ``U(T)`` by

    (d_t z_k, v) + (A grad(z_k), grad(v)) = 0,
    (z_k(0), v) = -(A e_k, grad(v)),

and ``w_k(T) = integral_0^T z_k(t) dt`` is approximated with backward Euler
and the trapezoidal rule.  The corrected basis is formed on ``U(T)``, but the
Petrov-Galerkin element contributions and the reconstructed solution use only
the central coarse cell ``T``.

``strategy=2`` uses the full zero-boundary patch space and is the direct
parabolic-oversampling variant. ``strategy=1`` additionally constrains the
corrector at the vertices of ``T``; this combines the reduced-space idea of
Henning--Peterseim Strategy 1 with the parabolic evolution.

The MPI backend uses coarse-cell task parallelism.  Every rank owns a replicated
``COMM_SELF`` coarse/fine mesh, constructs only its assigned patches, and adds
their element contributions to a distributed coarse matrix on ``COMM_WORLD``.
The reconstructed DG field is reduced across ranks.  This avoids truncated
cross-rank patches, but intentionally trades memory scalability for a simple and
reliable first MPI implementation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import dolfinx
from dolfinx import fem, mesh, plot
from dolfinx.fem import petsc as fem_petsc
from dolfinx.fem.petsc import LinearProblem
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl


CORE_CELL_TAG = 1


@dataclass
class PatchBasis:
    """Corrected basis and the cell map needed for reconstruction."""

    functions: list[fem.Function]
    core_patch_cells: np.ndarray
    core_fine_cells: np.ndarray


@dataclass
class ConvergenceResult:
    num_cells_per_side: int
    mesh_size: float
    l2_error: float
    broken_h1_error: float
    reference_l2_error: float
    l2_rate: float = math.nan
    broken_h1_rate: float = math.nan


@dataclass(frozen=True)
class ErrorSample:
    num_cells_per_side: int
    mesh_size: float
    oversampling_layers: int
    final_time: float
    num_time_steps: int
    strategy: int
    l2_error: float
    broken_h1_error: float
    reference_l2_error: float


class EllipticProblemData:
    """Coefficient, source and exact solution for the model problem."""

    def __init__(self, epsilon: float = 5.0e-2):
        self.epsilon = epsilon

    def coefficient(self, msh: mesh.Mesh):
        x = ufl.SpatialCoordinate(msh)[0]
        theta = 2 * ufl.pi * x / self.epsilon
        return (
            1
            / (8 * ufl.pi**2)
            * ufl.as_matrix(
                [
                    [2 / (2 + ufl.cos(theta)), 0],
                    [0, 1 + 0.5 * ufl.cos(theta)],
                ]
            )
        )

    def source(self, msh: mesh.Mesh):
        x, y = ufl.SpatialCoordinate(msh)
        return ufl.sin(2 * ufl.pi * x) * ufl.sin(2 * ufl.pi * y)

    def exact_expression(self, msh: mesh.Mesh):
        x, y = ufl.SpatialCoordinate(msh)
        return ufl.sin(2 * ufl.pi * x) * ufl.sin(
            2 * ufl.pi * y
        ) + self.epsilon / 2 * ufl.cos(2 * ufl.pi * x) * ufl.sin(
            2 * ufl.pi * y
        ) * ufl.sin(2 * ufl.pi * x / self.epsilon)


def refine_with_coarse_cell_tags(
    coarse_mesh: mesh.Mesh, refinement_levels: int
) -> tuple[mesh.Mesh, mesh.MeshTags]:
    """Uniformly refine and retain the parent coarse-cell id on fine cells."""

    tdim = coarse_mesh.topology.dim
    cell_map = coarse_mesh.topology.index_map(tdim)
    coarse_cells = np.arange(cell_map.size_local, dtype=np.int32)
    coarse_cell_tags = mesh.meshtags(
        coarse_mesh,
        tdim,
        coarse_cells,
        cell_map.local_to_global(coarse_cells).astype(np.int32),
    )

    fine_mesh = coarse_mesh
    fine_cell_tags = coarse_cell_tags
    for _ in range(refinement_levels):
        fine_mesh.topology.create_entities(1)
        fine_mesh, parent_cells, _ = mesh.refine(
            fine_mesh,
            partitioner=None,
            option=mesh.RefinementOption.parent_cell,
        )
        fine_cell_tags = mesh.transfer_meshtag(fine_cell_tags, fine_mesh, parent_cells)

    return fine_mesh, fine_cell_tags


def coarse_cell_patch(
    coarse_mesh: mesh.Mesh, coarse_cell: int, num_layers: int
) -> np.ndarray:
    """Return a vertex-connected patch measured in coarse-cell layers."""

    if num_layers < 0:
        raise ValueError("num_layers must be non-negative")

    tdim = coarse_mesh.topology.dim
    coarse_mesh.topology.create_connectivity(tdim, 0)
    coarse_mesh.topology.create_connectivity(0, tdim)
    cell_to_vertices = coarse_mesh.topology.connectivity(tdim, 0)
    vertex_to_cells = coarse_mesh.topology.connectivity(0, tdim)

    patch = {int(coarse_cell)}
    for _ in range(num_layers):
        expanded = set(patch)
        for cell in patch:
            for vertex in cell_to_vertices.links(cell):
                expanded.update(int(c) for c in vertex_to_cells.links(vertex))
        patch = expanded

    return np.asarray(sorted(patch), dtype=np.int32)


def boundary_dofs(V: fem.FunctionSpace) -> np.ndarray:
    """Locate degrees of freedom on the exterior boundary of ``V.mesh``."""

    tdim = V.mesh.topology.dim
    fdim = tdim - 1
    V.mesh.topology.create_connectivity(fdim, tdim)
    exterior_facets = mesh.exterior_facet_indices(V.mesh.topology)
    return fem.locate_dofs_topological(V, fdim, exterior_facets)


def dofs_at_points(
    V: fem.FunctionSpace, points: np.ndarray, atol: float = 1.0e-12
) -> np.ndarray:
    """Locate scalar nodal degrees of freedom at specified physical points."""

    gdim = V.mesh.geometry.dim
    point_columns = np.asarray(points)[:, :gdim].T

    def marker(x):
        matches = np.isclose(
            x[:gdim, :, None],
            point_columns[:, None, :],
            rtol=0.0,
            atol=atol,
        )
        return np.any(np.all(matches, axis=0), axis=1)

    return fem.locate_dofs_geometrical(V, marker)


def get_affine_basis_extensions(
    V_H: fem.FunctionSpace, V_patch: fem.FunctionSpace, coarse_cell: int
) -> tuple[list[fem.Function], np.ndarray]:
    """Extend a simplex P1 coarse-cell basis affinely over a patch."""

    coarse_dofs = V_H.dofmap.cell_dofs(coarse_cell)
    coarse_points = V_H.tabulate_dof_coordinates()[coarse_dofs]
    patch_points = V_patch.tabulate_dof_coordinates()
    gdim = V_H.mesh.geometry.dim
    num_basis = len(coarse_dofs)

    if num_basis != gdim + 1:
        raise NotImplementedError(
            "Affine basis extensions require scalar P1 elements on simplices."
        )

    coarse_vandermonde = np.column_stack((np.ones(num_basis), coarse_points[:, :gdim]))
    coefficients = np.linalg.solve(coarse_vandermonde, np.eye(num_basis)).T

    patch_vandermonde = np.column_stack(
        (np.ones(len(patch_points)), patch_points[:, :gdim])
    )
    values = patch_vandermonde @ coefficients.T

    basis = []
    for i, nodal_values in enumerate(values.T):
        phi = fem.Function(V_patch, name=f"phi_{coarse_cell}_{i}")
        phi.x.array[:] = nodal_values
        phi.x.scatter_forward()
        basis.append(phi)

    return basis, coefficients[:, 1:]


def create_lu_solver(A: PETSc.Mat, prefix: str) -> PETSc.KSP:
    """Create a direct solver for a matrix that is reused repeatedly."""

    solver = PETSc.KSP().create(A.comm)
    solver.setOptionsPrefix(prefix)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    preconditioner = solver.getPC()
    preconditioner.setType(PETSc.PC.Type.LU)
    if A.comm.getSize() > 1:
        preconditioner.setFactorSolverType("mumps")
    solver.setErrorIfNotConverged(True)
    solver.setUp()
    return solver


def get_parabolic_corrector_basis(
    V_patch: fem.FunctionSpace,
    A_patch,
    final_time: float,
    num_time_steps: int,
    additional_zero_dofs: np.ndarray | None = None,
    solver_prefix: str = "parabolic_corrector_",
) -> list[fem.Function]:
    """Compute time-integrated coordinate correctors by backward Euler."""

    if final_time <= 0:
        raise ValueError("final_time must be positive")
    if num_time_steps <= 0:
        raise ValueError("num_time_steps must be positive")

    constrained_dofs = boundary_dofs(V_patch)
    if additional_zero_dofs is not None:
        constrained_dofs = np.unique(
            np.concatenate((constrained_dofs, additional_zero_dofs))
        ).astype(np.int32)

    zero_bc = fem.dirichletbc(dolfinx.default_scalar_type(0), constrained_dofs, V_patch)
    bcs = [zero_bc]

    trial = ufl.TrialFunction(V_patch)
    test = ufl.TestFunction(V_patch)
    dx_patch = ufl.dx(domain=V_patch.mesh)
    mass_form = fem.form(trial * test * dx_patch)
    stiffness = ufl.inner(A_patch * ufl.grad(trial), ufl.grad(test)) * dx_patch

    time_step = final_time / num_time_steps
    step_form = fem.form(trial * test * dx_patch + time_step * stiffness)

    mass = fem_petsc.assemble_matrix(mass_form)
    mass.assemble()
    constrained_mass = fem_petsc.assemble_matrix(mass_form, bcs=bcs)
    constrained_mass.assemble()
    step_matrix = fem_petsc.assemble_matrix(step_form, bcs=bcs)
    step_matrix.assemble()

    mass_solver = create_lu_solver(constrained_mass, f"{solver_prefix}initial_")
    step_solver = create_lu_solver(step_matrix, f"{solver_prefix}step_")
    rhs = mass.createVecRight()

    gdim = V_patch.mesh.geometry.dim
    correctors: list[fem.Function] = []
    for direction in range(gdim):
        unit_vector = ufl.as_vector([1 if i == direction else 0 for i in range(gdim)])
        initial_form = fem.form(
            ufl.inner(-A_patch * unit_vector, ufl.grad(test)) * dx_patch
        )
        initial_rhs = fem_petsc.assemble_vector(initial_form)
        initial_rhs.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        fem_petsc.set_bc(initial_rhs, bcs)

        previous = fem.Function(V_patch)
        current = fem.Function(V_patch)
        mass_solver.solve(initial_rhs, previous.x.petsc_vec)
        previous.x.scatter_forward()

        corrector = fem.Function(V_patch, name=f"parabolic_corrector_{direction}")
        corrector.x.array[:] = 0.5 * time_step * previous.x.array

        for step in range(num_time_steps):
            mass.mult(previous.x.petsc_vec, rhs)
            fem_petsc.set_bc(rhs, bcs)
            step_solver.solve(rhs, current.x.petsc_vec)
            current.x.scatter_forward()

            weight = 0.5 if step == num_time_steps - 1 else 1.0
            corrector.x.array[:] += weight * time_step * current.x.array
            previous, current = current, previous

        corrector.x.scatter_forward()
        correctors.append(corrector)
        initial_rhs.destroy()

    rhs.destroy()
    mass_solver.destroy()
    step_solver.destroy()
    mass.destroy()
    constrained_mass.destroy()
    step_matrix.destroy()
    return correctors


def get_multiscale_basis(
    affine_basis: Sequence[fem.Function],
    basis_gradients: np.ndarray,
    correctors: Sequence[fem.Function],
    coarse_cell: int,
) -> list[fem.Function]:
    """Construct ``Psi_i = Phi_i + grad(Phi_i) dot w`` on a patch."""

    corrector_values = np.column_stack([corrector.x.array for corrector in correctors])
    basis = []
    for i, phi in enumerate(affine_basis):
        psi = fem.Function(phi.function_space, name=f"psi_{coarse_cell}_{i}")
        psi.x.array[:] = phi.x.array + corrector_values @ basis_gradients[i]
        psi.x.scatter_forward()
        basis.append(psi)
    return basis


class ParabolicOversamplingMsFEM:
    """Petrov-Galerkin MsFEM with MPI-parallel independent patch tasks."""

    def __init__(
        self,
        coarse_mesh: mesh.Mesh,
        problem_data: EllipticProblemData,
        refinement_levels: int,
        oversampling_layers: int,
        final_time: float,
        num_time_steps: int,
        strategy: int = 2,
        execution_comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        if coarse_mesh.comm.size != 1:
            raise NotImplementedError(
                "The local-task backend requires a replicated COMM_SELF mesh."
            )
        if strategy not in (1, 2):
            raise ValueError("strategy must be 1 or 2")

        self.coarse_mesh = coarse_mesh
        self.mesh_comm = coarse_mesh.comm
        self.execution_comm = execution_comm
        self.problem_data = problem_data
        self.refinement_levels = refinement_levels
        self.oversampling_layers = oversampling_layers
        self.final_time = final_time
        self.num_time_steps = num_time_steps
        self.strategy = strategy

        self.tdim = coarse_mesh.topology.dim
        if self.tdim != 2 or coarse_mesh.topology.cell_type != mesh.CellType.triangle:
            raise NotImplementedError(
                "This prototype currently supports two-dimensional triangles."
            )

        self.V_H = fem.functionspace(coarse_mesh, ("CG", 1))
        self.num_coarse_cells = coarse_mesh.topology.index_map(self.tdim).size_local
        self.num_coarse_dofs = self.V_H.dofmap.index_map.size_local
        self.fine_mesh, self.coarse_cell_tags = refine_with_coarse_cell_tags(
            coarse_mesh, refinement_levels
        )
        self.local_coarse_cells = np.arange(
            execution_comm.rank,
            self.num_coarse_cells,
            execution_comm.size,
            dtype=np.int32,
        )
        self.patch_bases: dict[int, PatchBasis] = {}

    def _patch_fine_cells(self, coarse_cell: int) -> np.ndarray:
        patch_coarse_cells = coarse_cell_patch(
            self.coarse_mesh, coarse_cell, self.oversampling_layers
        )
        fine_cell_blocks = [
            self.coarse_cell_tags.find(int(cell)) for cell in patch_coarse_cells
        ]
        return np.unique(np.concatenate(fine_cell_blocks)).astype(np.int32)

    def _strategy_one_dofs(
        self, V_patch: fem.FunctionSpace, coarse_cell: int
    ) -> np.ndarray | None:
        if self.strategy == 2:
            return None

        coarse_dofs = self.V_H.dofmap.cell_dofs(coarse_cell)
        coarse_vertices = self.V_H.tabulate_dof_coordinates()[coarse_dofs]
        vertex_dofs = dofs_at_points(V_patch, coarse_vertices)
        if len(vertex_dofs) != len(coarse_dofs):
            raise RuntimeError(
                "Could not identify every coarse-cell vertex in the patch space."
            )
        return vertex_dofs

    def offline(self) -> tuple[PETSc.Mat, PETSc.Vec]:
        """Compute assigned patch bases and assemble the distributed PG system."""

        A_H = PETSc.Mat().createAIJ(
            (self.num_coarse_dofs, self.num_coarse_dofs),
            nnz=12,
            comm=self.execution_comm,
        )
        A_H.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        b_H = A_H.createVecRight()
        b_H.set(0)
        self.patch_bases.clear()

        for coarse_cell_value in self.local_coarse_cells:
            coarse_cell = int(coarse_cell_value)
            core_fine_cells = self.coarse_cell_tags.find(coarse_cell)
            patch_fine_cells = self._patch_fine_cells(coarse_cell)
            patch_mesh, patch_to_fine, _, _ = mesh.create_submesh(
                self.fine_mesh, self.tdim, patch_fine_cells
            )
            core_patch_cells = patch_to_fine.sub_topology_to_topology(
                core_fine_cells, inverse=True
            ).astype(np.int32)
            if np.any(core_patch_cells < 0):
                raise RuntimeError("The oversampling patch does not contain T.")

            V_patch = fem.functionspace(patch_mesh, ("CG", 1))
            A_patch = self.problem_data.coefficient(patch_mesh)
            f_patch = self.problem_data.source(patch_mesh)

            affine_basis, basis_gradients = get_affine_basis_extensions(
                self.V_H, V_patch, coarse_cell
            )
            extra_dofs = self._strategy_one_dofs(V_patch, coarse_cell)
            correctors = get_parabolic_corrector_basis(
                V_patch,
                A_patch,
                self.final_time,
                self.num_time_steps,
                additional_zero_dofs=extra_dofs,
                solver_prefix=f"patch_{coarse_cell}_",
            )
            if extra_dofs is not None:
                for corrector in correctors:
                    if not np.allclose(
                        corrector.x.array[extra_dofs], 0.0, atol=1.0e-12
                    ):
                        raise RuntimeError(
                            "A Strategy 1 corrector is nonzero at a coarse vertex."
                        )
            multiscale_basis = get_multiscale_basis(
                affine_basis,
                basis_gradients,
                correctors,
                coarse_cell,
            )

            partition_values = np.sum([psi.x.array for psi in multiscale_basis], axis=0)
            if not np.allclose(partition_values, 1.0, atol=1.0e-10):
                raise RuntimeError(
                    "The local multiscale basis lost partition of unity."
                )

            tagged_core_cells = np.sort(core_patch_cells)
            core_tags = mesh.meshtags(
                patch_mesh,
                self.tdim,
                tagged_core_cells,
                np.full(len(tagged_core_cells), CORE_CELL_TAG, dtype=np.int32),
            )
            dx_core = ufl.Measure("dx", domain=patch_mesh, subdomain_data=core_tags)(
                CORE_CELL_TAG
            )

            coarse_dofs = self.V_H.dofmap.cell_dofs(coarse_cell)
            for i, phi in enumerate(affine_basis):
                local_load = fem.assemble_scalar(fem.form(f_patch * phi * dx_core))
                b_H.setValue(int(coarse_dofs[i]), local_load, addv=True)
                for j, psi in enumerate(multiscale_basis):
                    local_stiffness = fem.assemble_scalar(
                        fem.form(
                            ufl.inner(A_patch * ufl.grad(psi), ufl.grad(phi)) * dx_core
                        )
                    )
                    A_H.setValue(
                        int(coarse_dofs[i]),
                        int(coarse_dofs[j]),
                        local_stiffness,
                        addv=True,
                    )

            self.patch_bases[coarse_cell] = PatchBasis(
                functions=multiscale_basis,
                core_patch_cells=core_patch_cells,
                core_fine_cells=core_fine_cells.astype(np.int32),
            )

        A_H.assemble()
        b_H.assemble()
        coarse_boundary_dofs = boundary_dofs(self.V_H)
        row_start, row_end = A_H.getOwnershipRange()
        owned_boundary_dofs = coarse_boundary_dofs[
            (coarse_boundary_dofs >= row_start) & (coarse_boundary_dofs < row_end)
        ]
        A_H.zeroRowsColumns(owned_boundary_dofs, diag=1.0)
        b_H.setValues(
            owned_boundary_dofs,
            np.zeros(len(owned_boundary_dofs), dtype=PETSc.ScalarType),
        )
        b_H.assemble()
        return A_H, b_H

    def _gather_coarse_solution(self, solution: PETSc.Vec) -> np.ndarray:
        """Replicate a distributed PETSc vector in serial coarse-dof ordering."""

        ownership = solution.getOwnershipRange()
        local_values = solution.getArray(readonly=True).copy()
        pieces = self.execution_comm.allgather((ownership, local_values))
        global_values = np.empty(self.num_coarse_dofs, dtype=PETSc.ScalarType)
        for (start, end), values in pieces:
            global_values[start:end] = values
        return global_values

    def online(
        self, A_H: PETSc.Mat, b_H: PETSc.Vec
    ) -> tuple[fem.Function, fem.Function]:
        """Solve globally and reconstruct the replicated broken fine-scale field."""

        coarse_solver = create_lu_solver(A_H, "coarse_problem_")
        coarse_solution = A_H.createVecRight()
        coarse_solver.solve(b_H, coarse_solution)
        global_coefficients = self._gather_coarse_solution(coarse_solution)

        u_H = fem.Function(self.V_H, name="Coarse_Solution")
        u_H.x.array[:] = global_coefficients
        u_H.x.scatter_forward()

        V_ms = fem.functionspace(self.fine_mesh, ("DG", 1))
        u_ms = fem.Function(V_ms, name="Parabolic_Oversampling_MsFEM")
        num_fine_cells = self.fine_mesh.topology.index_map(self.tdim).size_local
        reconstructed_cells = np.zeros(num_fine_cells, dtype=bool)

        for coarse_cell, patch_basis in self.patch_bases.items():
            if np.any(reconstructed_cells[patch_basis.core_fine_cells]):
                raise RuntimeError(
                    "A fine cell is assigned by more than one coarse-cell reconstruction."
                )
            reconstructed_cells[patch_basis.core_fine_cells] = True

            cell_coefficients = u_H.x.array[self.V_H.dofmap.cell_dofs(coarse_cell)]
            u_patch = fem.Function(patch_basis.functions[0].function_space)
            u_patch.x.array[:] = (
                np.column_stack([psi.x.array for psi in patch_basis.functions])
                @ cell_coefficients
            )
            u_patch.x.scatter_forward()

            u_ms.interpolate(
                u_patch,
                patch_basis.core_patch_cells,
                patch_basis.core_fine_cells,
            )

        coverage = np.empty(num_fine_cells, dtype=np.int32)
        self.execution_comm.Allreduce(
            reconstructed_cells.astype(np.int32),
            coverage,
            op=MPI.SUM,
        )
        if not np.all(coverage == 1):
            raise RuntimeError(
                "MPI coarse-cell tasks must cover every fine cell exactly once."
            )

        self.execution_comm.Allreduce(MPI.IN_PLACE, u_ms.x.array, op=MPI.SUM)
        u_ms.x.scatter_forward()
        coarse_solution.destroy()
        coarse_solver.destroy()
        return u_H, u_ms


def solve_reference_problem(
    fine_mesh: mesh.Mesh, problem_data: EllipticProblemData
) -> fem.Function:
    """Solve the original elliptic problem in fine-grid CG1."""

    V_ref = fem.functionspace(fine_mesh, ("CG", 1))
    trial = ufl.TrialFunction(V_ref)
    test = ufl.TestFunction(V_ref)
    A = problem_data.coefficient(fine_mesh)
    source = problem_data.source(fine_mesh)
    dx = ufl.dx(domain=fine_mesh)
    zero_bc = fem.dirichletbc(
        dolfinx.default_scalar_type(0), boundary_dofs(V_ref), V_ref
    )
    problem = LinearProblem(
        ufl.inner(A * ufl.grad(trial), ufl.grad(test)) * dx,
        source * test * dx,
        bcs=[zero_bc],
        petsc_options_prefix="reference_problem_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    solution = problem.solve()
    solution.name = "Reference_Solution"
    return solution


def error_norms(
    approximation: fem.Function,
    exact_expression,
) -> tuple[float, float]:
    """Return the L2 error and broken H1 seminorm error."""

    msh = approximation.function_space.mesh
    error = approximation - exact_expression
    local_l2 = fem.assemble_scalar(
        fem.form(ufl.inner(error, error) * ufl.dx(domain=msh))
    )
    local_h1 = fem.assemble_scalar(
        fem.form(ufl.inner(ufl.grad(error), ufl.grad(error)) * ufl.dx(domain=msh))
    )
    global_l2 = msh.comm.allreduce(local_l2, op=MPI.SUM)
    global_h1 = msh.comm.allreduce(local_h1, op=MPI.SUM)
    return math.sqrt(float(np.real(global_l2))), math.sqrt(float(np.real(global_h1)))


def _refinement_levels(fine_size: int, num_cells_per_side: int) -> int:
    """Return the binary refinement depth connecting the two meshes."""

    if fine_size % num_cells_per_side != 0:
        raise ValueError("fine_size must be divisible by every coarse size")
    refinement_ratio = fine_size // num_cells_per_side
    refinement_levels = int(round(math.log2(refinement_ratio)))
    if 2**refinement_levels != refinement_ratio:
        raise ValueError("fine_size / coarse_size must be an integer power of two")
    return refinement_levels


def _function_triangulation(function: fem.Function) -> mtri.Triangulation:
    """Create a Matplotlib triangulation in finite-element dof ordering."""

    topology, _, geometry = plot.vtk_mesh(function.function_space)
    vtk_cells = topology.reshape(-1, 4)
    if not np.all(vtk_cells[:, 0] == 3):
        raise NotImplementedError("Solution plots currently require triangle cells.")
    return mtri.Triangulation(
        geometry[:, 0],
        geometry[:, 1],
        vtk_cells[:, 1:],
    )


def save_solution_comparison(
    u_ms: fem.Function,
    reference: fem.Function,
    output_path: Path,
    title: str,
) -> None:
    """Plot the broken MsFEM solution, fine reference and their difference."""

    reference_dg = fem.Function(u_ms.function_space, name="Reference_on_DG1")
    reference_dg.interpolate(reference)
    reference_dg.x.scatter_forward()

    error = fem.Function(u_ms.function_space, name="MsFEM_minus_reference")
    error.x.array[:] = u_ms.x.array - reference_dg.x.array
    error.x.scatter_forward()

    solution_range = (
        min(float(np.min(u_ms.x.array)), float(np.min(reference.x.array))),
        max(float(np.max(u_ms.x.array)), float(np.max(reference.x.array))),
    )
    error_limit = float(np.max(np.abs(error.x.array)))
    if error_limit == 0.0:
        error_limit = 1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    panels = (
        (u_ms, "MsFEM solution", "viridis", solution_range),
        (reference, "Fine reference", "viridis", solution_range),
        (
            error,
            "MsFEM - reference",
            "coolwarm",
            (-error_limit, error_limit),
        ),
    )
    for axis, (function, panel_title, colour_map, limits) in zip(axes, panels):
        colour_plot = axis.tripcolor(
            _function_triangulation(function),
            np.real(function.x.array),
            shading="gouraud",
            cmap=colour_map,
            vmin=limits[0],
            vmax=limits[1],
        )
        axis.set_title(panel_title)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(colour_plot, ax=axis, shrink=0.82)
    figure.suptitle(title)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_error_curves(
    series: dict[str, Sequence[tuple[float, ErrorSample]]],
    x_label: str,
    output_path: Path,
    title: str,
    *,
    x_scale: str = "linear",
    reverse_x_axis: bool = False,
) -> None:
    """Save paired L2 and broken-H1 error curves."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    metrics = (
        ("l2_error", r"$\|u_{ms}-u_{ex}\|_{L^2}$", axes[0]),
        ("broken_h1_error", r"$|u_{ms}-u_{ex}|_{H^1(\mathcal{T}_H)}$", axes[1]),
    )

    all_x_values: set[float] = set()
    for label, samples in series.items():
        ordered = sorted(samples, key=lambda item: item[0])
        x_values = np.asarray([item[0] for item in ordered], dtype=float)
        all_x_values.update(x_values.tolist())
        for metric, _, axis in metrics:
            y_values = np.asarray(
                [getattr(item[1], metric) for item in ordered], dtype=float
            )
            axis.plot(x_values, y_values, marker="o", linewidth=1.8, label=label)

    for _, y_label, axis in metrics:
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_xscale(x_scale)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
        if reverse_x_axis:
            axis.invert_xaxis()
        if x_scale == "linear" and all(float(x).is_integer() for x in all_x_values):
            axis.set_xticks(sorted(all_x_values))

    for metric, _, axis in metrics:
        y_values = [
            getattr(sample, metric)
            for samples in series.values()
            for _, sample in samples
        ]
        minimum, maximum = min(y_values), max(y_values)
        if np.isclose(minimum, maximum, rtol=1.0e-12, atol=0.0):
            axis.set_ylim(0.9 * minimum, 1.1 * maximum)

    figure.suptitle(title)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run_configuration(
    num_cells_per_side: int,
    fine_size: int,
    oversampling_layers: int,
    final_time: float,
    num_time_steps: int,
    epsilon: float,
    strategy: int,
    solution_output: Path | None = None,
) -> ErrorSample:
    """Solve and measure one point in a parameter sweep."""

    execution_comm = MPI.COMM_WORLD
    refinement_levels = _refinement_levels(fine_size, num_cells_per_side)
    coarse_mesh = mesh.create_unit_square(
        MPI.COMM_SELF,
        num_cells_per_side,
        num_cells_per_side,
        cell_type=mesh.CellType.triangle,
        ghost_mode=mesh.GhostMode.none,
    )
    problem_data = EllipticProblemData(epsilon)
    method = ParabolicOversamplingMsFEM(
        coarse_mesh,
        problem_data,
        refinement_levels,
        oversampling_layers,
        final_time,
        num_time_steps,
        strategy,
        execution_comm,
    )

    if execution_comm.rank == 0:
        print(
            f"N_H={num_cells_per_side:>3d}, L={oversampling_layers}, "
            f"T={final_time:g}, steps={num_time_steps}, strategy={strategy}, "
            f"MPI ranks={execution_comm.size}",
            flush=True,
        )
    A_H, b_H = method.offline()
    _, u_ms = method.online(A_H, b_H)

    sample = None
    if execution_comm.rank == 0:
        reference = solve_reference_problem(method.fine_mesh, problem_data)
        exact = problem_data.exact_expression(method.fine_mesh)
        l2_error, broken_h1_error = error_norms(u_ms, exact)
        reference_l2_error, _ = error_norms(reference, exact)

        sample = ErrorSample(
            num_cells_per_side=num_cells_per_side,
            mesh_size=1 / num_cells_per_side,
            oversampling_layers=oversampling_layers,
            final_time=final_time,
            num_time_steps=num_time_steps,
            strategy=strategy,
            l2_error=l2_error,
            broken_h1_error=broken_h1_error,
            reference_l2_error=reference_l2_error,
        )
        print(
            f"  L2={l2_error:.6e}, broken-H1={broken_h1_error:.6e}, "
            f"fine-reference L2={reference_l2_error:.6e}",
            flush=True,
        )

        if solution_output is not None:
            save_solution_comparison(
                u_ms,
                reference,
                solution_output,
                (
                    f"Parabolic oversampling MsFEM: strategy {strategy}, "
                    f"H={1 / num_cells_per_side:g}, L={oversampling_layers}, "
                    f"T={final_time:g}"
                ),
            )
        del reference

    sample = execution_comm.bcast(sample, root=0)

    A_H.destroy()
    b_H.destroy()
    del u_ms, method, coarse_mesh
    gc.collect()
    return sample


def run_convergence_experiment(
    coarse_sizes: Sequence[int],
    fine_size: int,
    oversampling_layers: int,
    final_time: float,
    num_time_steps: int,
    epsilon: float,
    strategy: int,
) -> list[ConvergenceResult]:
    """Run an H-sweep while keeping the global fine mesh size fixed."""

    results: list[ConvergenceResult] = []
    for num_cells_per_side in coarse_sizes:
        sample = run_configuration(
            num_cells_per_side,
            fine_size,
            oversampling_layers,
            final_time,
            num_time_steps,
            epsilon,
            strategy,
        )
        result = ConvergenceResult(
            num_cells_per_side=num_cells_per_side,
            mesh_size=1 / num_cells_per_side,
            l2_error=sample.l2_error,
            broken_h1_error=sample.broken_h1_error,
            reference_l2_error=sample.reference_l2_error,
        )

        if results:
            previous = results[-1]
            denominator = math.log(previous.mesh_size / result.mesh_size)
            result.l2_rate = math.log(previous.l2_error / result.l2_error) / denominator
            result.broken_h1_rate = (
                math.log(previous.broken_h1_error / result.broken_h1_error)
                / denominator
            )
        results.append(result)

    return results


def write_samples_csv(samples: Sequence[ErrorSample], output_path: Path) -> None:
    """Write every unique sweep point so plots can be reproduced externally."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in ErrorSample.__dataclass_fields__.values()]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        for sample in sorted(
            samples,
            key=lambda item: (
                item.strategy,
                item.num_cells_per_side,
                item.oversampling_layers,
                item.final_time,
            ),
        ):
            writer.writerow({name: getattr(sample, name) for name in field_names})


def run_plot_suite(
    coarse_sizes: Sequence[int],
    fine_size: int,
    oversampling_layers: int,
    final_time: float,
    num_time_steps: int,
    epsilon: float,
    strategies: Sequence[int],
    layer_values: Sequence[int],
    time_values: Sequence[float],
    plot_directory: Path,
    experiment: str = "all",
) -> list[ErrorSample]:
    """Run cached parameter sweeps and save solution/error figures."""

    execution_comm = MPI.COMM_WORLD
    coarse_sizes = tuple(sorted(set(coarse_sizes)))
    strategies = tuple(sorted(set(strategies)))
    layer_values = tuple(sorted({0, *layer_values}))
    time_values = tuple(sorted(set(time_values)))
    if not coarse_sizes or not strategies or not layer_values or not time_values:
        raise ValueError("Every parameter sweep must contain at least one value")
    if any(layer < 0 for layer in layer_values):
        raise ValueError("oversampling layers must be non-negative")
    if any(time <= 0 for time in time_values):
        raise ValueError("final times must be positive")
    if final_time <= 0 or num_time_steps <= 0:
        raise ValueError("final_time and num_time_steps must be positive")

    if execution_comm.rank == 0:
        plot_directory.mkdir(parents=True, exist_ok=True)
    execution_comm.Barrier()
    base_time_step = final_time / num_time_steps
    cache: dict[tuple[int, int, float, int, int], ErrorSample] = {}

    def evaluate(
        num_cells: int,
        layers: int,
        time: float,
        strategy: int,
        solution_output: Path | None = None,
    ) -> ErrorSample:
        steps = max(1, int(round(time / base_time_step)))
        key = (num_cells, layers, float(time), steps, strategy)
        if key not in cache:
            cache[key] = run_configuration(
                num_cells,
                fine_size,
                layers,
                time,
                steps,
                epsilon,
                strategy,
                solution_output,
            )
        return cache[key]

    finest_coarse_size = max(coarse_sizes)
    solution_layers = oversampling_layers
    for strategy in strategies:
        evaluate(
            finest_coarse_size,
            solution_layers,
            final_time,
            strategy,
            plot_directory / f"solution_strategy_{strategy}.png",
        )

    if experiment in ("all", "layers"):
        layer_coarse_sizes = coarse_sizes[-2:]
        layer_times = tuple(dict.fromkeys((time_values[0], time_values[-1])))
        comparison: dict[str, list[tuple[float, ErrorSample]]] = {}
        for strategy in strategies:
            series: dict[str, list[tuple[float, ErrorSample]]] = {}
            for num_cells in layer_coarse_sizes:
                for time in layer_times:
                    label = f"H={1 / num_cells:g}, T={time:g}"
                    series[label] = [
                        (layers, evaluate(num_cells, layers, time, strategy))
                        for layers in layer_values
                    ]
            if execution_comm.rank == 0:
                save_error_curves(
                    series,
                    "oversampling layers L (L=0: none)",
                    plot_directory / f"error_vs_layers_strategy_{strategy}.png",
                    f"Oversampling-layer study: strategy {strategy}",
                )
            comparison[f"strategy {strategy}"] = [
                (
                    layers,
                    evaluate(
                        finest_coarse_size,
                        layers,
                        layer_times[-1],
                        strategy,
                    ),
                )
                for layers in layer_values
            ]
        if execution_comm.rank == 0:
            save_error_curves(
                comparison,
                "oversampling layers L (L=0: none)",
                plot_directory / "error_vs_layers_strategy_comparison.png",
                (
                    "Strategy comparison versus oversampling layers: "
                    f"H={1 / finest_coarse_size:g}, T={layer_times[-1]:g}"
                ),
            )

    if experiment in ("all", "time"):
        comparison = {}
        for strategy in strategies:
            series = {}
            for layers in layer_values:
                layer_label = "L=0 (none)" if layers == 0 else f"L={layers}"
                series[layer_label] = [
                    (time, evaluate(finest_coarse_size, layers, time, strategy))
                    for time in time_values
                ]
            if execution_comm.rank == 0:
                save_error_curves(
                    series,
                    "parabolic final time T",
                    plot_directory / f"error_vs_T_strategy_{strategy}.png",
                    (
                        f"Parabolic-time study: strategy {strategy}, "
                        f"H={1 / finest_coarse_size:g}"
                    ),
                )
            comparison[f"strategy {strategy}"] = [
                (
                    time,
                    evaluate(
                        finest_coarse_size,
                        layer_values[-1],
                        time,
                        strategy,
                    ),
                )
                for time in time_values
            ]
        if execution_comm.rank == 0:
            save_error_curves(
                comparison,
                "parabolic final time T",
                plot_directory / "error_vs_T_strategy_comparison.png",
                (
                    "Strategy comparison versus parabolic time: "
                    f"H={1 / finest_coarse_size:g}, L={layer_values[-1]}"
                ),
            )

    if experiment in ("all", "h"):
        h_times = tuple(dict.fromkeys((time_values[0], time_values[-1])))
        h_layers = tuple(dict.fromkeys((0, layer_values[-1])))
        comparison = {}
        for strategy in strategies:
            series = {}
            for layers in h_layers:
                for time in h_times:
                    layer_label = "none" if layers == 0 else str(layers)
                    label = f"L={layer_label}, T={time:g}"
                    series[label] = [
                        (
                            1 / num_cells,
                            evaluate(num_cells, layers, time, strategy),
                        )
                        for num_cells in coarse_sizes
                    ]
            if execution_comm.rank == 0:
                save_error_curves(
                    series,
                    "coarse mesh size H",
                    plot_directory / f"error_vs_H_strategy_{strategy}.png",
                    f"Coarse-mesh study: strategy {strategy}",
                    x_scale="log",
                    reverse_x_axis=True,
                )
            comparison[f"strategy {strategy}"] = [
                (
                    1 / num_cells,
                    evaluate(
                        num_cells,
                        h_layers[-1],
                        h_times[-1],
                        strategy,
                    ),
                )
                for num_cells in coarse_sizes
            ]
        if execution_comm.rank == 0:
            save_error_curves(
                comparison,
                "coarse mesh size H",
                plot_directory / "error_vs_H_strategy_comparison.png",
                (f"Strategy comparison versus H: L={h_layers[-1]}, T={h_times[-1]:g}"),
                x_scale="log",
                reverse_x_axis=True,
            )

    samples = list(cache.values())
    if execution_comm.rank == 0:
        write_samples_csv(samples, plot_directory / "sweep_results.csv")
    return samples


def print_convergence_table(results: Sequence[ConvergenceResult]) -> None:
    header = (
        "| N_H | H | L2 error | rate | broken H1 error | rate | fine reference L2 |"
    )
    separator = "|---:|---:|---:|---:|---:|---:|---:|"
    print("\n" + header)
    print(separator)
    for result in results:
        l2_rate = "-" if math.isnan(result.l2_rate) else f"{result.l2_rate:.3f}"
        h1_rate = (
            "-" if math.isnan(result.broken_h1_rate) else f"{result.broken_h1_rate:.3f}"
        )
        print(
            f"| {result.num_cells_per_side} | {result.mesh_size:.5f} | "
            f"{result.l2_error:.6e} | {l2_rate} | "
            f"{result.broken_h1_error:.6e} | {h1_rate} | "
            f"{result.reference_l2_error:.6e} |"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plotting and convergence studies for parabolic oversampling MsFEM"
    )
    parser.add_argument("--coarse-sizes", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--fine-size", type=int, default=32)
    parser.add_argument("--oversampling-layers", type=int, default=1)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--time-steps", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=5.0e-2)
    parser.add_argument(
        "--strategies", nargs="+", type=int, choices=(1, 2), default=[1, 2]
    )
    parser.add_argument("--layer-values", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--time-values", nargs="+", type=float, default=[0.25, 0.5, 1.0]
    )
    parser.add_argument(
        "--experiment",
        choices=("all", "layers", "time", "h"),
        default="all",
        help="Select all plots or one parameter sweep; solution plots are always saved.",
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=Path("plots/parabolic_oversampling")
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    samples = run_plot_suite(
        coarse_sizes=arguments.coarse_sizes,
        fine_size=arguments.fine_size,
        oversampling_layers=arguments.oversampling_layers,
        final_time=arguments.final_time,
        num_time_steps=arguments.time_steps,
        epsilon=arguments.epsilon,
        strategies=arguments.strategies,
        layer_values=arguments.layer_values,
        time_values=arguments.time_values,
        plot_directory=arguments.plot_dir,
        experiment=arguments.experiment,
    )
    if MPI.COMM_WORLD.rank == 0:
        print(
            f"\nSaved {len(samples)} unique samples and plots to {arguments.plot_dir}"
        )
