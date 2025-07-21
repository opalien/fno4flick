from mpi4py import MPI
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem


import torch
from torch import Tensor


from collections.abc import Callable
import os


class DataGenerator:
    def __init__(self,
                 R: float, r_max: float,
                 C_in: float, C_out: float,
                 D_in: float, D_out: float,
                 T1_in: float, T1_out: float,
                 P0_in: float, P0_out: float,
                 Tfinal: float = 10.,
                 Nr: int = 100,
                 Nt: int = 100,
                 tanh_slope: float = 0.,
                 ):

        self.R = R
        self.r_max = r_max
        self.Tfinal = Tfinal
        self.Nr = Nr
        self.Nt = Nt
        self.dt = self.Tfinal / self.Nt
        self.tanh_slope = tanh_slope

        self.C_in = C_in
        self.C_out = C_out
        self.D_in = D_in
        self.D_out = D_out
        self.T1_in = T1_in
        self.T1_out = T1_out
        self.P0_in = P0_in
        self.P0_out = P0_out

        if tanh_slope == 0.:
            self.C: Callable = lambda r: ufl.conditional(ufl.lt(r, self.R), C_in, C_out)
            self.D: Callable = lambda r: ufl.conditional(ufl.lt(r, self.R), D_in, D_out)
            self.T1: Callable = lambda r: ufl.conditional(ufl.lt(r, self.R), T1_in, T1_out)
            self.P0: Callable = lambda r: ufl.conditional(ufl.lt(r, self.R), P0_in, P0_out)

        else:
            mid   = 0.5*(C_out + C_in)
            amp   = 0.5*(C_out - C_in)
            self.C  = lambda r: mid + amp * ufl.tanh((r - self.R)/tanh_slope)

            mid   = 0.5*(D_out + D_in)
            amp   = 0.5*(D_out - D_in)
            self.D  = lambda r: mid + amp * ufl.tanh((r - self.R)/tanh_slope)

            mid   = 0.5*(T1_out + T1_in)
            amp   = 0.5*(T1_out - T1_in)
            self.T1 = lambda r: mid + amp * ufl.tanh((r - self.R)/tanh_slope)

            mid   = 0.5*(P0_out + P0_in)
            amp   = 0.5*(P0_out - P0_in)
            self.P0 = lambda r: mid + amp * ufl.tanh((r - self.R)/tanh_slope)
            #self.C: Callable = lambda r: ufl.tanh((r - self.R) / tanh_slope) * (C_out - C_in) + C_in
            #self.D: Callable = lambda r: ufl.tanh((r - self.R) / tanh_slope) * (D_out - D_in) + D_in
            #self.T1: Callable = lambda r: ufl.tanh((r - self.R) / tanh_slope) * (T1_out - T1_in) + T1_in
            #self.P0: Callable = lambda r: ufl.tanh((r - self.R) / tanh_slope) * (P0_out - P0_in) + P0_in


        self.msh = None
        self.V = None
        self.P_time = []
        self.r_sorted = None
        self.t_vec = None

    
    def solve(self):

        self.msh = mesh.create_interval(MPI.COMM_WORLD, self.Nr, [0.0, self.r_max])

        self.V = fem.functionspace(self.msh, ("Lagrange", 1))

        x = ufl.SpatialCoordinate(self.msh)[0]
        r = x
        w = r**2  # Jacobian for spherical coordinates

        C_expr = self.C(r)
        D_expr = self.D(r)
        T1_expr = self.T1(r)
        P0_expr = self.P0(r)

        # ----------------------------------------------------------
        # Variables for the problem
        # ----------------------------------------------------------

        P_n1 = ufl.TrialFunction(self.V)  # P^{n+1}
        v = ufl.TestFunction(self.V)  # Test function
        P_n = fem.Function(self.V)  # P^{n}

        P_n.x.array[:] = 0.0  # Initial condition: P(r, 0) = 0

        self.P_time.append(P_n)
        dx = ufl.dx

        # ----------------------------------------------------------
        # Weak form
        # ----------------------------------------------------------

        mass_lhs = (C_expr / ScalarType(self.dt)) * w * P_n1 * v * dx
        mass_rhs = (C_expr / ScalarType(self.dt)) * w * P_n * v * dx

        diff_lhs = D_expr * C_expr * w * ufl.dot(ufl.grad(P_n1), ufl.grad(v)) * dx

        react_lhs = (C_expr / T1_expr) * w * P_n1 * v * dx
        react_rhs = (C_expr / T1_expr) * w * P0_expr * v * dx

        a_form = mass_lhs + diff_lhs + react_lhs
        L_form = mass_rhs + react_rhs

        # ----------------------------------------------------------
        # Linear problem setup | Neumann comes from the weak form
        # ----------------------------------------------------------

        problem = LinearProblem(
            a_form, L_form, [], 
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )


        n_loc = self.V.dofmap.index_map.size_local
        r_loc = self.V.tabulate_dof_coordinates()[:n_loc, 0]
        r_all = self.msh.comm.gather(r_loc, root=0)

        if self.msh.comm.rank == 0:
            r_glob = np.concatenate(r_all)
            order = np.argsort(r_glob)
            self.r_sorted = r_glob[order]
            P_hist = []

        t = 0.0
        for _ in range(self.Nt):
            t += self.dt

            P_new = problem.solve()

            P_loc = P_new.x.array[:n_loc]
            P_all = self.msh.comm.gather(P_loc, root=0)
            if self.msh.comm.rank == 0:
                P_glob = np.concatenate(P_all)
                P_hist.append(P_glob[order].copy())

            P_n.x.array[:] = P_new.x.array

        if self.msh.comm.rank == 0:
            self.P_time = np.array(P_hist)
            self.t_vec = np.linspace(self.dt, self.Tfinal, self.Nt)


    def plot(self, dir: str = "data/plot"):
        if self.msh.comm.rank == 0:
            if self.P_time is None or self.r_sorted is None or self.t_vec is None:
                print("No data to plot. Please run solve() first.")
                return

            import matplotlib.pyplot as plt
            os.makedirs(dir, exist_ok=True)

            plt.figure(figsize=(8, 5))
            plt.imshow(self.P_time,
                       extent=[0.0, self.r_max, 0.0, self.Tfinal],
                       origin="lower",
                       aspect="auto",
                       cmap="viridis")

            plt.colorbar(label=r"$P(r,t)$")
            plt.xlabel(r"$r$ (m)")
            plt.ylabel(r"$t$ (s)")
            plt.title("Évolution de la polarisation P(r,t)")
            plt.tight_layout()
            filename = f"{dir}/{self.R}_{self.C_in}_{self.C_out}_{self.D_in}_{self.D_out}_{self.P0_in}_{self.P0_out}_{self.T1_in}_{self.T1_out}.png"
            plt.savefig(filename, dpi=180)
            # plt.show()

    def get(self) -> dict[str, Tensor | float | int]:
        if self.msh.comm.rank == 0:
            P_tensor = torch.tensor(self.P_time, dtype=torch.float32)

            to_send: dict[str, Tensor | float | int] = {
                "P": P_tensor,
                "r_max": self.r_max,
                "Tfinal": self.Tfinal,
                "Nr": self.Nr,
                "Nt": self.Nt,
                "dt": self.dt,
                "C_in": self.C_in,
                "C_out": self.C_out,
                "D_in": self.D_in,
                "D_out": self.D_out,
                "T1_in": self.T1_in,
                "T1_out": self.T1_out,
                "P0_in": self.P0_in,
                "P0_out": self.P0_out,
                "R": self.R
            }
            return to_send
        return {}
        


