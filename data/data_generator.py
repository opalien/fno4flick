from mpi4py import MPI
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import matplotlib.pyplot as plt

import torch
from torch import Tensor


from collections.abc import Callable


class DataGenerator:
    def __init__(self,
                 R: float,
                 C_cuve: float, C_ball: float,
                 D_cuve: float, D_ball: float,
                 Tre_cuve: float, Tre_ball: float,
                 P0_cuve: float, P0_ball: float,
                 cuve_width: float = 10.,
                 Tfinal: float = 10.,
                 Nr: int = 100,
                 Nt: int = 100,
                 ):

        self.R = R
        self.C: Callable = lambda r: ufl.conditional(ufl.lt(r, self.R), C_ball, C_cuve)
        self.D: Callable = lambda r: ufl.conditional(ufl.lt(r, self.R), D_ball, D_cuve)
        self.Tre: Callable = lambda r: ufl.conditional(ufl.lt(r, self.R), Tre_ball, Tre_cuve)
        self.P0: Callable = lambda r: ufl.conditional(ufl.lt(r, self.R), P0_ball, P0_cuve)

        self.cuve_width = cuve_width
        self.Tfinal = Tfinal
        self.Nr = Nr
        self.Nt = Nt
        self.dt = self.Tfinal / self.Nt

        self.C_cuve = C_cuve
        self.C_ball = C_ball
        self.D_cuve = D_cuve
        self.D_ball = D_ball
        self.Tre_cuve = Tre_cuve
        self.Tre_ball = Tre_ball
        self.P0_cuve = P0_cuve
        self.P0_ball = P0_ball

        self.f: Callable[[float], float] = lambda t: 1000.0 if t > 0. else 0.0

        self.mesh = None
        self.V = None
        self.P_time = []
        self.r_sorted = None
        self.t_vec = None


    def solve(self):

        self.mesh = mesh.create_interval(MPI.COMM_WORLD, self.Nr, [0.0, self.cuve_width])
        self.V = fem.functionspace(self.mesh, ("Lagrange", 1))

        x = ufl.SpatialCoordinate(self.mesh)[0]
        r = x
        w = r**2

        C_expr = self.C(r)
        D_expr = self.D(r)
        Tre_expr = self.Tre(r)
        P0_expr = self.P0(r)

        #fem.Constant(self.mesh, ScalarType(0.0))


        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        P_old = fem.Function(self.V, name="P_old")
        P_old.interpolate(lambda x: np.where(x[0] < self.R, self.P0_ball, self.P0_cuve))
        # P_old.interpolate(lambda x: np.zeros(x.shape[1]))

        a = ((C_expr * w / self.dt) * u * v
             + D_expr * C_expr * w * ufl.dot(ufl.grad(u), ufl.grad(v))
             + (C_expr / Tre_expr) * w * u * v) * ufl.dx

        L = ((C_expr * w / self.dt) * P_old * v
             + (C_expr / Tre_expr) * w * P0_expr * v) * ufl.dx


        def on_wall(x):
            return np.isclose(x[0], self.cuve_width)

        dofs_wall = fem.locate_dofs_geometrical(self.V, on_wall)
        P_wall_bc = fem.Function(self.V)
        bc = fem.dirichletbc(P_wall_bc, dofs_wall)


        problem = LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )


        n_owned = self.V.dofmap.index_map.size_local
        r_local = self.V.tabulate_dof_coordinates()[:n_owned, 0]
        all_r = self.mesh.comm.gather(r_local, root=0)

        order = None
        if self.mesh.comm.rank == 0:
            r_glob = np.concatenate(all_r)
            order = np.argsort(r_glob)
            self.r_sorted = r_glob[order] 
            self.P_time = []
        
        t = 0.0
        with io.XDMFFile(self.mesh.comm, "polarisation_P.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.mesh)

            for _ in range(self.Nt):
                t += self.dt
                
                P_wall_bc.x.array.fill(self.f(t))
                P_wall_bc.x.scatter_forward()


                P_new = problem.solve()
                P_new.name = "P"
                xdmf.write_function(P_new, t)


                P_local = P_new.x.array[:n_owned]
                all_P = self.mesh.comm.gather(P_local, root=0)

                if self.mesh.comm.rank == 0:
                    P_glob = np.concatenate(all_P)
                    P_sorted = P_glob[order]
                    self.P_time.append(P_sorted.copy())


                P_old.x.array[:] = P_new.x.array
        
        if self.mesh.comm.rank == 0:
            self.t_vec = np.linspace(self.dt, self.Tfinal, self.Nt)


    def plot(self, dir: str = "data/plot"):
        if self.mesh.comm.rank == 0:
            if not self.P_time or self.r_sorted is None or self.t_vec is None:
                print("No data to plot. Please run solve() first.")
                return

            P_mat = np.array(self.P_time)
            
            plt.figure(figsize=(8, 5))
            plt.imshow(P_mat,
                       extent=[0.0, self.cuve_width, 0.0, self.Tfinal],
                       origin="lower",
                       aspect="auto",
                       cmap="viridis")

            plt.colorbar(label=r"$P(r,t)$")
            plt.xlabel(r"$r$ (m)")
            plt.ylabel(r"$t$ (s)")
            plt.title("Évolution de la polarisation P(r,t)")
            plt.tight_layout()
            plt.savefig(f"{dir}/P_rt_{self.C_cuve}_{self.C_ball}_{self.Tre_cuve}_{self.R}.png", dpi=180)
            #plt.show()


            #plt.figure(figsize=(8, 5))
            #t_slice = 0.5
            #if t_slice > self.Tfinal:
            #    time_index = -1
            #else:
            #    time_index = np.argmin(np.abs(self.t_vec - t_slice))
            #
            #P_slice = P_mat[time_index, :]
            #t_val = self.t_vec[time_index]
#
            #plt.plot(self.r_sorted, P_slice)
            #plt.xlabel(r"$r$ (m)")
            #plt.ylabel(r"$P(r, t)$")
            #plt.title(f"Tranche de polarisation à t = {t_val:.2f} s")
            #plt.grid(True)
            #plt.tight_layout()
            #plt.savefig(f"P_r_slice_t_{t_slice:.2f}.png", dpi=180)
            #plt.show()


    def get(self):
        P_tensor = torch.tensor(self.P_time)

        to_send: dict[str, Tensor | float | int] = {
            "P": P_tensor,
            "cuve_width": self.cuve_width,
            "Tfinal": self.Tfinal,
            "Nr": self.Nr,
            "Nt": self.Nt,
            "dt": self.dt,
            "C_cuve": self.C_cuve,
            "C_ball": self.C_ball,
            "D_cuve": self.D_cuve,
            "D_ball": self.D_ball,
            "Tre_cuve": self.Tre_cuve,
            "Tre_ball": self.Tre_ball,
            "P0_cuve": self.P0_cuve,
            "P0_ball": self.P0_ball,
            "R": self.R
        }

        return to_send


