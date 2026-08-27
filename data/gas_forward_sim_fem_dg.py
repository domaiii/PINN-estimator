# %%
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm

import numpy as np
import pandas as pd
import ufl
from basix.ufl import element
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree

from data.visualizer import Visualizer

def _read_physical_name_map(
    meshfile: str | Path, dim: int | None = None
) -> dict[str, int]:
    """Read physical groups with name (str) and group id (int). """
    import gmsh

    meshfile = Path(meshfile).resolve(strict=True)
    gmsh.initialize()
    try:
        gmsh.open(str(meshfile))
        groups = gmsh.model.getPhysicalGroups(dim)
        return {gmsh.model.getPhysicalName(dim, tag): tag for (dim, tag) in groups}
    finally:
        gmsh.finalize()

case_dir = Path("/app/data/example_apartment")
mesh_path = case_dir / "apartment_2d.msh"
wind_path = case_dir / "wind_gt.csv"

D_value = 1e-2
source_xy = (4.3, 3.5)
source_sigma = 0.1
source_rate = 1.0

meshdata = io.gmsh.read_from_msh(
    str(mesh_path), MPI.COMM_WORLD, gdim=2
)

domain = meshdata.mesh
cell_tags = meshdata.cell_tags
facet_tags = meshdata.facet_tags

V = fem.functionspace(domain, element("DG", domain.basix_cell(), 0))
W = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), 1, shape=(2,)))

wind = pd.read_csv(wind_path)
wind_xy = wind[["Points:0", "Points:1"]].to_numpy(float)
wind_uv = wind[["U:0", "U:1"]].to_numpy(float)
tree = cKDTree(wind_xy)

beta = fem.Function(W, name="wind")

def wind_interp(x):
    xy = np.column_stack([x[0], x[1]])
    nearest = tree.query(xy)[1]
    values = np.zeros((2, x.shape[1]))
    values[0] = wind_uv[nearest, 0]
    values[1] = wind_uv[nearest, 1]
    return values

beta.interpolate(wind_interp)




source = fem.Function(V, name="source")

def source_interp(x):
    r2 = (x[0] - source_xy[0]) ** 2 + (x[1] - source_xy[1]) ** 2
    return source_rate * np.exp(-r2 / (2.0 * source_sigma**2))

source.interpolate(source_interp)

name_to_id = _read_physical_name_map(mesh_path, dim=1)
inlet_l = name_to_id["Inflow_left"]
inlet_r = name_to_id["Inflow_lower"]

c = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
D = fem.Constant(domain, PETSc.ScalarType(D_value))
h = ufl.CellDiameter(domain)
beta_norm = ufl.sqrt(ufl.dot(beta, beta) + PETSc.ScalarType(1e-30))

# Cell Peclet number: Pe_h = |beta| h / (2 D)
Q_pe = fem.functionspace(domain, element("DG", domain.basix_cell(), 0))
pe = fem.Function(Q_pe, name="cell_Peclet")
pe_expr = fem.Expression(
    beta_norm * h / (2.0 * D),
    Q_pe.element.interpolation_points,
)
pe.interpolate(pe_expr)

num_owned_pe = Q_pe.dofmap.index_map.size_local * Q_pe.dofmap.index_map_bs
pe_local = pe.x.array[:num_owned_pe].real
pe_min = domain.comm.allreduce(float(np.min(pe_local)), op=MPI.MIN)
pe_max = domain.comm.allreduce(float(np.max(pe_local)), op=MPI.MAX)
pe_sum = domain.comm.allreduce(float(np.sum(pe_local)), op=MPI.SUM)
pe_count = domain.comm.allreduce(pe_local.size, op=MPI.SUM)
pe_above_one = domain.comm.allreduce(
    int(np.count_nonzero(pe_local > 1.0)), op=MPI.SUM
)

if domain.comm.rank == 0:
    print(
        f"Cell Peclet number: min={pe_min:.3e}, "
        f"mean={pe_sum / pe_count:.3e}, max={pe_max:.3e}"
    )
    print(f"Cells with Pe_h > 1: {pe_above_one}/{pe_count}")

n = ufl.FacetNormal(domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
dS = ufl.Measure("dS", domain=domain)

# First-order finite-volume-like DG0 diffusion through interior facets.
diffusion_penalty = PETSc.ScalarType(1.0)
diffusion_flux = (
    D
    * diffusion_penalty
    / ufl.avg(h)
    * (c("+") - c("-"))
    * (v("+") - v("-"))
    * dS
)

# Weak c=0 condition for diffusive inflow; walls and outlet retain zero
# diffusive flux.
diffusive_inlet = (
    D * diffusion_penalty / h * c * v * ds(inlet_l)
    + D * diffusion_penalty / h * c * v * ds(inlet_r)
)

# Conservative upwind advection. On exterior inflow facets the prescribed
# concentration is zero; on outflow facets c leaves with the wind.
beta_n_plus = ufl.dot(beta("+"), n("+"))
c_upwind = ufl.conditional(beta_n_plus >= 0.0, c("+"), c("-"))
interior_advection = (
    beta_n_plus * c_upwind * (v("+") - v("-")) * dS
)

beta_n = ufl.dot(beta, n)
outflow_speed = ufl.conditional(beta_n > 0.0, beta_n, 0.0)
boundary_advection = outflow_speed * c * v * ds

a = diffusion_flux + diffusive_inlet + interior_advection + boundary_advection
L = source * v * ufl.dx

problem = LinearProblem(
    a,
    L,
    petsc_options_prefix="gas_dg0_",
    bcs=[],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
concentration = problem.solve()
concentration.name = "concentration"

# Export one DG0 concentration value per cell at the cell center.
num_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
coordinates_local = V.tabulate_dof_coordinates()[:num_owned, :2]
concentration_local = concentration.x.array[:num_owned].real
export_local = np.column_stack([coordinates_local, concentration_local])
export_parts = domain.comm.gather(export_local, root=0)

if domain.comm.rank == 0:
    export_data = np.vstack(export_parts)
    order = np.lexsort((export_data[:, 0], export_data[:, 1]))
    gas_csv_path = case_dir / f"gas_gt{source_xy[0]:.1f}_{source_xy[1]:.1f}.csv"
    pd.DataFrame(
        export_data[order],
        columns=["Points:0", "Points:1", "C"],
    ).to_csv(gas_csv_path, index=False, float_format="%.7e")
    print(f"Saved gas distribution to {gas_csv_path}")

vis = Visualizer(domain)
vis.add_background_mesh()
vis.add_scalar_field("concentration", concentration, cmap="plasma")
vis.show(colorbar_label="Concentration")



# %%
