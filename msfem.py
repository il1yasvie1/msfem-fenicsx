import dolfinx
from dolfinx import mesh
from dolfinx import fem
from dataclasses import dataclass
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
from mpi4py import MPI
import pyvista

from typing import List


pyvista.OFF_SCREEN = True


@dataclass
class LocalBasisData:
    """Multiscale basis values without retaining the local submesh."""

    fine_dofs: np.ndarray
    values: np.ndarray


def refine(coarse_mesh, refinement_levels):
    tdim = coarse_mesh.topology.dim
    imap = coarse_mesh.topology.index_map(tdim)
    local_cells = np.arange(imap.size_local, dtype=np.int32)

    coarse_to_fine_cells = mesh.meshtags(
        coarse_mesh,
        tdim,
        local_cells,
        imap.local_to_global(local_cells).astype(np.int32)
    )
    refined_mesh = coarse_mesh

    for _ in range(refinement_levels):
        refined_mesh.topology.create_entities(1)
        refined_mesh, parent_cell, _ = mesh.refine(refined_mesh,
                                                   partitioner=None,
                                                   option=mesh.RefinementOption.parent_cell)
        coarse_to_fine_cells = mesh.transfer_meshtag(
                        coarse_to_fine_cells,
                        refined_mesh,
                        parent_cell)
    return refined_mesh, coarse_to_fine_cells


def oversampling(fine_mesh, num_layers, local_cell_list):
    fine_mesh.topology.create_connectivity(2, 0)
    fine_mesh.topology.create_connectivity(0, 2)
    cell_to_vertices = fine_mesh.topology.connectivity(2, 0)
    vertex_to_cells  = fine_mesh.topology.connectivity(0, 2)
    cell_to_vertices = cell_to_vertices.array.reshape(-1, 3)
    oversampled_cell_list = np.copy(local_cell_list)
    for _ in range(num_layers):
        oversampled_cell_set = set(oversampled_cell_list)
        for i in np.unique(cell_to_vertices[oversampled_cell_list].reshape(-1)):
            oversampled_cell_set = oversampled_cell_set.union(set(vertex_to_cells.links(i)))
        oversampled_cell_list = np.array(sorted(oversampled_cell_set), dtype=np.int32)
    return oversampled_cell_list


def boundary_dofs(V):
    tdim = V.mesh.topology.dim
    fdim = tdim - 1
    V.mesh.topology.create_connectivity(fdim, tdim)
    exterior_facets = mesh.exterior_facet_indices(V.mesh.topology)
    return fem.locate_dofs_topological(V, fdim, exterior_facets)


def plot_function(u):
    u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))
    u_grid.point_data[u.name] = u.x.array
    u_grid.set_active_scalars(u.name)
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid,
                       show_edges=False,
                       edge_color="gray",
                       line_width=0.1)
    u_plotter.view_xy()
    if pyvista.OFF_SCREEN:
        u_plotter.screenshot(f"plots/{u.name}.png")
    else:
        u_plotter.show()


def plot_function_warped(u):
    u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))
    u_grid.point_data[u.name] = u.x.array
    u_grid.set_active_scalars(u.name)
    warped = u_grid.warp_by_scalar()
    plotter = pyvista.Plotter()
    plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True)
    if pyvista.OFF_SCREEN:
        plotter.screenshot(f"plots/{u.name}_warped.png")
    else:
        plotter.show()


def plot_oversampling(coarse_mesh, fine_mesh, local_mesh):
    plotter = pyvista.Plotter()
    tdim = coarse_mesh.topology.dim

    fine_mesh.topology.create_connectivity(tdim, tdim)
    fine_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(fine_mesh, tdim))
    plotter.add_mesh(fine_grid,
            show_edges=True,
            color="w",
            edge_color="gray",
            line_width=1,
            label="Fine mesh")

    coarse_mesh.topology.create_connectivity(tdim, tdim)
    coarse_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(coarse_mesh, tdim))
    plotter.add_mesh(coarse_grid,
            show_edges=True,
            color="w",
            edge_color="red",
            line_width=3,
            opacity=0.5,
            label="Coarse mesh")

    local_mesh.topology.create_connectivity(tdim, tdim)
    local_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(local_mesh, tdim))
    plotter.add_mesh(local_grid,
            show_edges=True,
            color="lightskyblue",
            edge_color="blue",
            line_width=2,
            opacity=1,
            label="Local mesh")

    plotter.view_xy()
    plotter.show_axes()
    if pyvista.OFF_SCREEN:
        plotter.screenshot("plots/Oversampling.png")
    else:
        plotter.show()


class EllipticProblemData:
    def __init__(self, epsilon=5e-2):
        self.epsilon = epsilon

    def coefficient(self, msh):
        x, y = ufl.SpatialCoordinate(msh)
        theta = 2 * ufl.pi * x / self.epsilon

        return 1/(8*ufl.pi**2) * ufl.as_matrix([
            [2 / (2 + ufl.cos(theta)), 0],
            [0,  1 + .5 * ufl.cos(theta)]])

    def source(self, msh):
        x, y = ufl.SpatialCoordinate(msh)
        return ufl.sin(2 * ufl.pi * x) * ufl.sin(2 * ufl.pi * y)

    def exact_solution(self, V):
        def exact_solution_expression(x):
            return (
                    np.sin(2 * np.pi * x[0])
                    * np.sin(2 * np.pi * x[1])
                    + self.epsilon / 2
                    * np.cos(2 * np.pi * x[0])
                    * np.sin(2 * np.pi * x[1])
                    * np.sin(2 * np.pi * x[0] / self.epsilon)
                )
        u_exact = fem.Function(V, name="Exact_Solution")
        u_exact.interpolate(exact_solution_expression)
        return u_exact


def get_affine_basis_extensions(V_H: fem.FunctionSpace,
                                V_h: fem.FunctionSpace,
                                c: int):
    vandermonde_matrix = [[1, i, j] for i, j, _ in V_H.tabulate_dof_coordinates()[V_H.dofmap.cell_dofs(c)]]
    basis_coefs = np.linalg.inv(vandermonde_matrix).T
    Phi = [fem.Function(V_h, name=f"phi_0_{c}"), fem.Function(V_h, name=f"phi_1_{c}"), fem.Function(V_h, name=f"phi_2_{c}")]
    X = V_h.tabulate_dof_coordinates()
    for i in range(3):
        Phi[i].x.array[:] = 1 * basis_coefs[i, 0] \
            + X[:, 0] * basis_coefs[i, 1] \
            + X[:, 1] * basis_coefs[i, 2]
    return Phi


def get_local_corrector_basis(V: fem.FunctionSpace, A):
    wh, vh = ufl.TrialFunction(V), ufl.TestFunction(V)
    lhs = ufl.inner(A * ufl.grad(wh), ufl.grad(vh)) * ufl.dx
    ws = []
    for k, ek in enumerate([ufl.as_vector([1, 0]), ufl.as_vector([0, 1])]):
        rhs = ufl.inner(-A * ek, ufl.grad(vh)) * ufl.dx
        linear_problem = LinearProblem(
            lhs,
            rhs,
            bcs=[fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs(V), V)],
            petsc_options_prefix="local_corrector_problem_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            })
        wk = linear_problem.solve()
        wk.name = f"elliptic_corrector_{k}"
        ws.append(wk)
    return ws


def get_local_corrector_basis_parabolic(
    V: fem.FunctionSpace,
    A,
    T,
    Nt):

    dt = dolfinx.default_scalar_type(T / Nt)
    wh, vh = ufl.TrialFunction(V), ufl.TestFunction(V)
    lhs = (
        wh * vh
        + dt * ufl.inner(A * ufl.grad(wh), ufl.grad(vh))
    ) * ufl.dx
    bc = fem.dirichletbc(
        dolfinx.default_scalar_type(0),
        boundary_dofs(V),
        V)
    ws = []
    for k, ek in enumerate([ufl.as_vector([1, 0]), ufl.as_vector([0, 1])]):
        w0 = fem.Function(V, name=f"w0_{k}")
        rhs = (
            w0 * vh
            + dt * ufl.inner(-A * ek, ufl.grad(vh))
        ) * ufl.dx

        linear_problem = LinearProblem(
            lhs,
            rhs,
            bcs=[bc],
            petsc_options_prefix=f"wt_{k}_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )

        for _ in range(Nt):
            wt = linear_problem.solve()
            wt.x.scatter_forward()
            w0.x.array[:] = wt.x.array

        wk = fem.Function(V, name=f"parabolic_corrector_{k}")
        wk.x.array[:] = w0.x.array
        ws.append(wk)

    return ws


def get_local_multiscale_basis(V_H: fem.FunctionSpace,
                           V_h: fem.FunctionSpace,
                           Phi: List[fem.Function],
                           c: int,
                           w: List[fem.Function]):
    vandermonde_matrix = [[1, i, j] for i, j, _ in V_H.tabulate_dof_coordinates()[V_H.dofmap.cell_dofs(c)]]
    basis_coefs = np.linalg.inv(vandermonde_matrix).T
    Psi = [fem.Function(V_h, name=f"psi_0_{c}"), fem.Function(V_h, name=f"psi_1_{c}"), fem.Function(V_h, name=f"psi_2_{c}")]
    for i in range(3):
        Psi[i].x.array[:] = Phi[i].x.array \
            + w[0].x.array * basis_coefs[i, 1] \
            + w[1].x.array * basis_coefs[i, 2]
    return Psi


class MsFEM:
    """
    refine: two scale refinement algorithm
    oversampling: 
    """
    def __init__(self,
                 coarse_mesh: mesh.Mesh,
                 elliptic_problem_data: EllipticProblemData,
                 refinement_levels: int = 1,
                 num_layers: int = 0):

        tdim = coarse_mesh.topology.dim
        assert tdim == 2
        assert coarse_mesh.topology.cell_type == mesh.CellType.triangle
        
        self.coarse_mesh = coarse_mesh
        self.comm = coarse_mesh.comm
        self.num_cells_coarse = coarse_mesh.topology.index_map(tdim).size_global
        self.V_H = fem.functionspace(coarse_mesh, ("CG", 1))
        self.num_dofs_coarse = self.V_H.dofmap.index_map.size_global
        self.fine_mesh, self.coarse_to_fine_cells = refine(coarse_mesh,
                                                           refinement_levels)
        self.V_ms = fem.functionspace(self.fine_mesh, ("DG", 1))
        self.num_layers = num_layers
        self.data = elliptic_problem_data
        self.local_multiscale_basis: List[LocalBasisData] = []


    def offline(self, is_parabolic=False, T=1, Nt=1):
        self.local_multiscale_basis.clear()

        K_H = PETSc.Mat().create(MPI.COMM_WORLD)
        K_H.setSizes([self.num_dofs_coarse, self.num_dofs_coarse])
        K_H.setUp()
        F_H = PETSc.Vec().create(MPI.COMM_WORLD)
        F_H.setSizes(self.num_dofs_coarse)
        F_H.setUp()

        tdim = self.coarse_mesh.topology.dim

        for c in range(self.num_cells_coarse):
            fine_local_cell_list = self.coarse_to_fine_cells.find(c)
            fine_oversampled_cell_list = oversampling(self.fine_mesh, self.num_layers, fine_local_cell_list)
            oversampled_mesh, oversampled_to_fine_entity_map, _, _ = mesh.create_submesh(self.fine_mesh, tdim, fine_oversampled_cell_list)

            V_os = fem.functionspace(oversampled_mesh, ("CG", 1))

            Phi = get_affine_basis_extensions(self.V_H, V_os, c)
            A_os = self.data.coefficient(oversampled_mesh)
            if is_parabolic:
                w = get_local_corrector_basis_parabolic(V_os, A_os, T, Nt)
            else:
                w = get_local_corrector_basis(V_os, A_os)

            Psi = get_local_multiscale_basis(self.V_H, V_os, Phi, c, w)

            oversampled_to_local_cell_list = oversampled_to_fine_entity_map.sub_topology_to_topology(fine_local_cell_list, inverse=True)
            local_mesh, local_to_oversampled_entity_map, _, _ = mesh.create_submesh(oversampled_mesh, tdim, oversampled_to_local_cell_list)


            V_loc = fem.functionspace(local_mesh, ("CG", 1))
            A_loc = self.data.coefficient(local_mesh)
            f_loc = self.data.source(local_mesh)

            Phi_loc = [fem.Function(V_loc, name=f"phi_0_loc_{c}"),
                       fem.Function(V_loc, name=f"phi_1_loc_{c}"),
                       fem.Function(V_loc, name=f"phi_2_loc_{c}")]

            Psi_loc = [fem.Function(V_loc, name=f"psi_0_loc_{c}"),
                       fem.Function(V_loc, name=f"psi_1_loc_{c}"),
                       fem.Function(V_loc, name=f"psi_2_loc_{c}")]
            
            cells1 = np.arange(len(fine_local_cell_list), dtype=np.int32)
            cells0 = local_to_oversampled_entity_map.sub_topology_to_topology(cells1, False)
           
            for i in range(3):
                Phi_loc[i].interpolate(Phi[i], cells0, cells1)
                Psi_loc[i].interpolate(Psi[i], cells0, cells1)

            for i in range(3):
                fc = fem.assemble_scalar(fem.form(f_loc * Phi_loc[i] * ufl.dx))
                F_H.setValue(self.V_H.dofmap.cell_dofs(c)[i], fc, addv=True)
                for j in range(3):
                    kc = fem.assemble_scalar(fem.form(ufl.inner(A_loc * ufl.grad(Psi_loc[j]), ufl.grad(Phi_loc[i])) * ufl.dx))
                    K_H.setValue(self.V_H.dofmap.cell_dofs(c)[i], self.V_H.dofmap.cell_dofs(c)[j], kc, addv=True)

            fine_dofs = np.concatenate([
                self.V_ms.dofmap.cell_dofs(int(cell))
                for cell in fine_local_cell_list
            ]).astype(np.int32)

            basis_values = np.empty(
                (len(fine_dofs), len(Psi_loc)),
                dtype=dolfinx.default_scalar_type,
            )

            for i, psi_loc in enumerate(Psi_loc):
                psi_fine = fem.Function(self.V_ms)
                psi_fine.interpolate(
                    psi_loc,
                    cells1,
                    fine_local_cell_list,
                )
                basis_values[:, i] = psi_fine.x.array[fine_dofs]

            self.local_multiscale_basis.append(
                LocalBasisData(
                    fine_dofs=fine_dofs,
                    values=basis_values,
                )
            )

        K_H.assemble()
        K_H.zeroRowsColumns(boundary_dofs(self.V_H), diag=1.0)
        F_H.setValues(boundary_dofs(self.V_H), np.zeros(len(boundary_dofs(self.V_H))))
        F_H.assemble()
        K_H.setUp()
        F_H.setUp()
        return K_H, F_H

    def online(self, K_H, F_H):
        ksp = PETSc.KSP().create()
        ksp.setOperators(K_H)
        ksp.setFromOptions()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")

        u_H = fem.Function(self.V_H)
        ksp.solve(F_H, u_H.x.petsc_vec)

        u_ms = fem.Function(self.V_ms, name="Multiscale_Solution")

        for c, basis_data in enumerate(self.local_multiscale_basis):
            coarse_coefficients = u_H.x.array[self.V_H.dofmap.cell_dofs(c)]
            u_ms.x.array[basis_data.fine_dofs] = (
                basis_data.values @ coarse_coefficients
            )

        ksp.destroy()
        return u_ms
