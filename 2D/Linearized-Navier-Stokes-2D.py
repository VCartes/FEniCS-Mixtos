from dolfin import *

import sympy as sp
import numpy as np

from prettytable import PrettyTable



# ---- GLOBAL VARIABLES ----

# FUNCTIONS

x, y = sp.symbols("x[0] x[1]")

mu_val = 1

u1_ex = sp.cos(sp.pi*x) * sp.sin(sp.pi*y)
u2_ex = -sp.sin(sp.pi*x) * sp.cos(sp.pi*y)

p_ex = sp.cos(2*sp.pi*x) * sp.exp(y)

chi11_ex = u1_ex.diff(x, 1)
chi12_ex = u1_ex.diff(y, 1)
chi21_ex = u2_ex.diff(x, 1)
chi22_ex = u2_ex.diff(y, 1)

sigma11_ex = mu_val * chi11_ex - 1/2 * u1_ex * u1_ex - p_ex
sigma12_ex = mu_val * chi12_ex - 1/2 * u1_ex * u2_ex
sigma21_ex = mu_val * chi21_ex - 1/2 * u2_ex * u1_ex
sigma22_ex = mu_val * chi22_ex - 1/2 * u2_ex * u2_ex - p_ex


f1 = - mu_val * (u1_ex.diff(x, 2) + u1_ex.diff(y, 2)) \
        + u1_ex.diff(x, 1) * u1_ex + u1_ex.diff(y, 1) * u2_ex \
        + p_ex.diff(x, 1)

f2 = - mu_val * (u2_ex.diff(x, 2) + u2_ex.diff(y, 2)) \
        + u2_ex.diff(x, 1) * u1_ex + u2_ex.diff(y, 1) * u2_ex \
        + p_ex.diff(y, 1)



# EXPRESSIONS

u_ex        = Expression((sp.printing.ccode(u1_ex), sp.printing.ccode(u2_ex)), degree = 5)
p_ex        = Expression(sp.printing.ccode(p_ex), degree = 5)
chi_ex      = Expression([[sp.printing.ccode(chi11_ex), sp.printing.ccode(chi12_ex)], 
                          [sp.printing.ccode(chi21_ex), sp.printing.ccode(chi22_ex)]], degree = 5)
sigma_ex    = Expression([[sp.printing.ccode(sigma11_ex), sp.printing.ccode(sigma12_ex)], 
                          [sp.printing.ccode(sigma21_ex), sp.printing.ccode(sigma22_ex)]], degree = 5)

f = Expression((sp.printing.ccode(f1), sp.printing.ccode(f2)), degree = 5)


def build_trace_free_tensor(components):
    tensor = as_tensor(
            [[components[0], components[1]], 
             [components[2], -components[0]]]
            )

    return tensor

def build_RT_tensor(components):
    tensor = as_tensor([components[0], components[1]])
    return tensor

def L2_norm(f):
    return np.sqrt(assemble(f**2 * dx))

def L4_vector_norm(f):
    return np.pow(assemble(sum(f[i]**4 for i in range(2)) * dx(metadata={"quadrature_degree": 5})), 1/4)

def L2_tensor_norm(f):
    return np.sqrt(assemble(inner(f, f) * dx(metadata={"quadrature_degree": 5})))

def Hdiv43_tensor_norm(f):
    norm = L2_tensor_norm(f)

    divf = div(f)
    norm += np.pow(assemble(sum(divf[i]**(4/3) for i in range(2)) * dx(metadata={"quadrature_degree": 5})), 3/4)

    return norm

def error_rates(err_list, h_list):
    l = [0]

    for i in range(1, len(err_list)):
        l.append(ln(err_list[i]/err_list[i-1]) / ln(h_list[i]/h_list[i-1]))

    return l


def solve_variational_problem(mesh):
    n = FacetNormal(mesh)

    mu = Constant(mu_val)

    order = 1
    R   = FiniteElement("R", mesh.ufl_cell(), 0)
    DG  = FiniteElement("DG", mesh.ufl_cell(), order)

    # For L^4(\Omega)
    Hu = VectorElement("DG", mesh.ufl_cell(), order)

    # For L_2^{\tr}(\Omega)
    Hchi = MixedElement([DG for _ in range(3)])

    # For H_0(div_{4/3}, \Omega)
    RT = FiniteElement("RT", mesh.ufl_cell(), order+1)
    Hsigma = RT * RT

    Xh = FunctionSpace(mesh, MixedElement([MixedElement([Hu, Hchi, R]), Hsigma]))
    dim = Xh.dim()

    FSz = FunctionSpace(mesh, Hu)
    z = Function(FSz) # For linearization


    # SOLVE

    (u_vec, sigma_comps)  = TrialFunctions(Xh)
    (v_vec, tau_comps)    = TestFunctions(Xh)

    (u, chi_comps, xi)      = split(u_vec)
    (v, eta_comps, lambdaa) = split(v_vec)


    # Build tensor from components
    sigma   = build_RT_tensor(split(sigma_comps))
    tau     = build_RT_tensor(split(tau_comps))

    chi = build_trace_free_tensor(split(chi_comps))
    eta = build_trace_free_tensor(split(eta_comps))

    # FORMS

    A   = mu * inner(chi, eta) * dx + 1/2 * (dot(chi*z, v) - dot(eta*z, u)) * dx
    B   = - inner(eta, sigma) * dx - dot(v, div(sigma)) * dx + lambdaa * tr(sigma) * dx
    Bt  = - inner(chi, tau) * dx - dot(u, div(tau)) * dx + xi * tr(tau) * dx

    AA = A + B + Bt

    RH1 = dot(f, v) * dx
    RH2 = - dot(tau * n, u_ex) * ds

    FF = RH1 + RH2

    sol = Function(Xh)



    # PICARD ITERATIONS

    its = 0
    tol = 1e-12
    tol_c = tol + 1

    u_vec_sol = sigma_sol_comps = u_sol = chi_sol_comps = xi_sol = None

    while its < 30 and tol_c > tol:
        solve(AA == FF, sol, solver_parameters = {'linear_solver':'umfpack'})

        (u_vec_sol, sigma_sol_comps)    = sol.split(deepcopy=True)
        (u_sol, chi_sol_comps, xi_sol)  = u_vec_sol.split(deepcopy=True)

        tol_c = L4_vector_norm(u_sol - z) / L4_vector_norm(u_sol)

        z.assign(u_sol)
        its += 1



    # BUILD SOLUTIONS

    sigma_sol = build_RT_tensor(split(sigma_sol_comps))

    V = FunctionSpace(mesh, "DG", order)
    tensor_V = TensorFunctionSpace(mesh, "DG", order, shape = (2, 2))

    chi_sol = build_trace_free_tensor(split(chi_sol_comps))
    chi_sol = project(chi_sol, tensor_V)

    meas = assemble(Constant(1) * dx(mesh))

    sigma_full_sol = sigma_sol - (1/(4*meas) * assemble(dot(u_sol, u_sol) * dx)) * Identity(2)
    sigma_full_sol = project(sigma_full_sol, tensor_V)

    p_sol = -1/2 * tr(sigma_full_sol + 1/2 * outer(u_sol, u_sol))
    p_sol = project(p_sol, V)
    

    # ERROR CALCULATIONS

    err_u       = L4_vector_norm(u_sol - u_ex)
    err_p       = L2_norm(p_sol - p_ex)
    err_chi     = L2_tensor_norm(chi_sol - chi_ex)
    err_sigma   = L2_tensor_norm(sigma_full_sol - sigma_ex)

    return u_sol, chi_sol, sigma_full_sol, p_sol, err_u, err_chi, err_sigma, err_p, dim, its



# MAIN PROGRAM

if __name__ == "__main__":

    # DATA FOR TABLE

    hh          = list()
    n_elem      = list()
    dofs        = list()
    errs_u      = list()
    errs_chi    = list()
    errs_sigma  = list()
    errs_p      = list()
    iters       = list()

    
    # SOLVE VARIATIONAL PROBLEMS

    u =  chi = sigma = p = None

    NN = [4, 8, 16, 32]
    for N in NN:
        print(f"N = {N}")

        mesh = UnitSquareMesh(N, N)

        u, chi, sigma, p, err_u, err_chi, err_sigma, err_p, dim, its = solve_variational_problem(mesh)

        hh.append(mesh.hmax())
        n_elem.append(mesh.num_cells())
        dofs.append(dim)
        errs_u.append(err_u)
        errs_chi.append(err_chi)
        errs_sigma.append(err_sigma)
        errs_p.append(err_p)
        iters.append(its)


    # TABLES

    u_rates     = error_rates(errs_u, hh)
    chi_rates   = error_rates(errs_chi, hh)
    sigma_rates = error_rates(errs_sigma, hh)
    p_rates     = error_rates(errs_p, hh)

    table = PrettyTable(["#elements", "h", "dofs", \
            "err_u", "u_rate", \
            "err_chi", "chi_rate", \
            "err_sigma", "sigma_rate", \
            "err_p", "p_rate", \
            "iters"])

    table.add_rows([
        ["%d" % n_elem[i], "%2.4f" % hh[i], "%d" % dofs[i], \
                "%2.2e" % errs_u[i], "%2.4f" % u_rates[i], \
                "%2.2e" % errs_chi[i], "%2.4f" % chi_rates[i], \
                "%2.2e" % errs_sigma[i], "%2.4f" % sigma_rates[i], \
                "%2.2e" % errs_p[i], "%2.4f" % p_rates[i], \
                "%d" % iters[i]] \
                for i in range(len(hh))
                ])

    table.border = True
    print(table)


    # SAVE FILES

    u_file      = File("Data_Paraview_2D/u.pvd")
    chi_file    = File("Data_Paraview_2D/chi.pvd")
    sigma_file  = File("Data_Paraview_2D/sigma.pvd")
    p_file      = File("Data_Paraview_2D/p.pvd")

    u_file      << u
    chi_file    << chi
    sigma_file  << sigma
    p_file      << p


    print("DONE")


