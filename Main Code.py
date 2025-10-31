'''
tunneling_verification_suite.py

Comprehensive verification package for "Analytical and Numerical Modeling of Electron
Transmission Probability through Quantum Tunneling Barriers in Nanoscale Diodes".

Contains:
 - Transfer Matrix Method (TMM) for single and multilayer barriers
 - Finite-difference Schrödinger Equation (SE) eigen-solver for bound states (benchmark)
 - WKB transmission approximation for smooth barriers
 - Landauer current integration for J(V)
 - Simple 1D Schrödinger-Poisson self-consistent loop (finite-difference Poisson)
 - Monte-Carlo sampling of barrier width and statistics
 - Phenomenological inelastic scattering options:
      (A) Imaginary potential (optical potential) inside barrier layers
      (B) Lorentzian broadening of T(E)

Usage:
  python tunneling_verification_suite.py   # runs a demo that reproduces main plots / stats
Requirements:
  numpy, scipy, matplotlib
'''

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import hbar, m_e, e, k as k_B, h
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from math import sqrt
import os
import json
from dataclasses import dataclass

# ----------------------------
# Physical constants & helpers
# ----------------------------
eV_to_J = e
nm_to_m = 1e-9
default_T = 300.0  # K

def energy_to_joule(EeV):
    return EeV * eV_to_J

def joule_to_eV(J):
    return J / eV_to_J

# ----------------------------
# Basic models & tools
# ----------------------------

@dataclass
class DeviceParams:
    V0_eV: float = 0.9       # nominal barrier height
    t0_nm: float = 1.0       # nominal barrier thickness (single barrier)
    m_eff: float = 0.25      # effective mass relative to m0
    T_kelvin: float = default_T
    dE: float = 0.001        # energy step in eV (1 meV)
    E_min: float = -1.0      # integration window relative to mu (eV)
    E_max: float = 1.0
    dx_nm: float = 0.005     # spatial grid spacing for FD (nm) (used in SE/Poisson)
    convergence_tol_eV: float = 1e-5

# ----------------------------
# 1) Transfer Matrix Method (single rectangular barrier & multilayer)
# ----------------------------

def k_from_E(E_eV, V_eV_complex, m_eff):
    """Return complex wavevector k (1/m) for energy E in region of potential V."""
    E = energy_to_joule(E_eV)
    V = energy_to_joule(V_eV_complex)
    val = 2.0 * (m_eff * m_e) * (E - V)
    # allow complex sqrt for evanescent waves
    return np.sqrt(val + 0j) / hbar

def tmm_transmission_rectangular(E_eV, V0_eV, thickness_nm, m_eff=0.25):
    """
    Transmission T through a single rectangular barrier using analytic interface formulas.
    Leads assumed to have V=0.
    """
    kL = k_from_E(E_eV, 0.0, m_eff)
    kB = k_from_E(E_eV, V0_eV, m_eff)
    kR = k_from_E(E_eV, 0.0, m_eff)
    # reflection coefficients
    rLB = (kL - kB) / (kL + kB)
    rBR = (kB - kR) / (kB + kR)
    d = thickness_nm * nm_to_m
    denom = np.abs(1.0 - rLB * rBR * np.exp(-2j * kB * d))**2
    num = (np.abs(2.0 * kL / (kL + kB))**2) * (np.abs(2.0 * kB / (kB + kR))**2)
    v_ratio = np.real(kR) / (np.real(kL) if np.real(kL)!=0 else 1.0)
    T = (num / denom) * v_ratio
    T = np.real(T)
    T = max(0.0, min(1.0, T))
    return T

def tmm_multilayer(E_eV, regions, m_eff=0.25):
    """
    regions: list of tuples (V_eV_complex, thickness_nm)
      - first and last region are leads (thickness may be 0)
    Returns transmission coefficient (0..1)
    """
    ks = [k_from_E(E_eV, V, m_eff) for (V, d) in regions]
    M = np.array([[1+0j, 0+0j],[0+0j, 1+0j]])
    for i in range(len(regions)-1):
        k_i = ks[i]; k_j = ks[i+1]
        # interface matrix
        # avoid division by zero
        if np.isclose(k_i + k_j, 0):
            S = np.eye(2, dtype=complex)
        else:
            # continuity of psi and (1/m*) dpsi/dx assumed uniform m* here
            eta = k_i / k_j
            S = 0.5 * np.array([[1+eta, 1-eta],[1-eta, 1+eta]])
        # propagation in region j
        thickness_nm = regions[i+1][1]
        if thickness_nm is not None and thickness_nm > 0:
            d = thickness_nm * nm_to_m
            P = np.array([[np.exp(-1j * ks[i+1] * d), 0],[0, np.exp(1j * ks[i+1] * d)]])
        else:
            P = np.eye(2, dtype=complex)
        M = M @ S @ P
    k0 = ks[0]; kN = ks[-1]
    if np.isclose(M[0,0], 0):
        return 0.0
    t_amp = 1.0 / M[0,0]
    kr0 = np.real(k0) if np.real(k0)!=0 else 1e-30
    krN = np.real(kN) if np.real(kN)!=0 else 1e-30
    T = (krN / kr0) * np.abs(t_amp)**2
    if not np.isfinite(T): return 0.0
    T = max(0.0, min(1.0, np.real(T)))
    return T

# ----------------------------
# 2) Finite-difference Schrödinger eigen-solver (1D)
# ----------------------------
def build_hamiltonian_1d(V_profile_eV, dx_nm, m_eff):
    """
    Build sparse Hamiltonian matrix (in Joules) for 1D finite-difference Schrödinger:
      -V''/(2m) + V(x)
    V_profile_eV: array of potential (eV) on grid
    dx_nm: spacing in nm
    Returns H (N x N sparse), x_grid (nm)
    """
    dx = dx_nm * nm_to_m
    N = len(V_profile_eV)
    VJ = energy_to_joule(np.array(V_profile_eV))
    pref = -(hbar**2) / (2.0 * m_eff * m_e * dx**2)
    diag = np.full(N, -2.0 * pref) + VJ
    off = np.full(N-1, pref)
    H = sp.diags([off, diag, off], offsets=[-1, 0, 1], format='csr')
    x = np.arange(N) * dx_nm
    return H, x

def solve_se_bound_states(V_profile_eV, dx_nm, m_eff, n_states=10):
    H, x = build_hamiltonian_1d(V_profile_eV, dx_nm, m_eff)
    # Use sparse eigensolver for lowest eigenvalues
    eigvals, eigvecs = spla.eigsh(H, k=min(n_states, H.shape[0]-2), which='SM')
    # eigvals in Joules -> eV
    eig_eV = joule_to_eV(eigvals)
    return np.sort(eig_eV), x, eigvecs

# ----------------------------
# 3) WKB transmission approximation
# ----------------------------
def twkb_transmission(E_eV, V_func, x1_nm, x2_nm, m_eff):
    """
    WKB approximate transmission through classically forbidden region x1..x2
    V_func: callable V(x_nm) -> eV
    Returns T_wkb
    """
    xs = np.linspace(x1_nm, x2_nm, 1000)
    integrand = []
    for x in xs:
        Vx = V_func(x)
        if Vx > E_eV:
            kappa = sqrt(2.0 * m_eff * m_e * (energy_to_joule(Vx - E_eV))) / hbar
            integrand.append(kappa)
        else:
            integrand.append(0.0)
    integral = np.trapz(integrand, xs * nm_to_m)
    T = np.exp(-2.0 * integral)
    return T

# ----------------------------
# 4) Landauer current integration
# ----------------------------
def landauer_current_density(Vbias_V, T_of_E, E_eV, T_kelvin=default_T):
    """
    Compute current density (A/m^2) using Landauer formula, symmetric bias applied:
      mu_L = +qV/2, mu_R = -qV/2
    T_of_E: array of T(E) same length as E_eV (eV)
    """
    E_j = energy_to_joule(E_eV)
    muL = 0.5 * e * Vbias_V
    muR = -0.5 * e * Vbias_V
    fL = 1.0 / (1.0 + np.exp((E_j - muL) / (k_B * T_kelvin)))
    fR = 1.0 / (1.0 + np.exp((E_j - muR) / (k_B * T_kelvin)))
    integrand = T_of_E * (fL - fR)
    dE_J = (E_eV[1] - E_eV[0]) * eV_to_J
    I = (2.0 * e / h) * np.sum(integrand) * dE_J  # A per unit transverse channel; treat as density per area
    return I

# ----------------------------
# 5) Simple Schrödinger-Poisson loop (1D)
# ----------------------------
def poisson_solve_1d(rho, dx_nm, eps_rel=3.9, phi_left=0.0, phi_right=0.0):
    """
    Solve 1D Poisson: d/dx (epsilon dV/dx) = -rho/epsilon0
    Using uniform epsilon for simplicity. rho in C/m^3, dx in nm.
    Returns potential array phi (V).
    """
    eps0 = 8.8541878128e-12
    dx = dx_nm * nm_to_m
    N = len(rho)
    eps = eps_rel * eps0
    # finite difference matrix for second derivative
    main = np.full(N, -2.0)
    off = np.full(N-1, 1.0)
    A = sp.diags([off, main, off], offsets=[-1, 0, 1], format='csr') / (dx*dx)
    # apply Dirichlet BCs: modify first/last row
    A = A.tolil()
    A[0,:] = 0; A[0,0] = 1
    A[-1,:] = 0; A[-1,-1] = 1
    b = -rho / eps
    b[0] = phi_left; b[-1] = phi_right
    phi = spla.spsolve(A.tocsc(), b)
    return phi

def schrodinger_poisson_1d(initial_V_eV, dx_nm, m_eff, doping_Cperm3, mu_left_eV=0.0, mu_right_eV=0.0,
                           max_iter=100, tol_eV=1e-5, T_kelvin=300.0):
    """
    Very simplified self-consistent loop:
      - Solve SE (bound-state eigenproblem) in potential V(x)
      - Estimate electron density n(x) from eigenstates (approx), then update Poisson.
    Note: This is a 1D toy; for full quantum-corrected density and continuum states
    a scattering-state solver or NEGF is required.
    """
    V = np.array(initial_V_eV).copy()
    N = len(V)
    dx = dx_nm * nm_to_m
    for k in range(max_iter):
        # Solve bound states
        eigvals_eV, x_nm, eigvecs = solve_se_bound_states(V, dx_nm, m_eff, n_states=20)
        # compute approximate n(x) from low-lying eigenstates with Fermi occupation
        # assume chemical potential mu = average of contacts
        mu = 0.5*(mu_left_eV + mu_right_eV)
        n = np.zeros(N)
        for idx, E in enumerate(eigvals_eV):
            occ = 1.0 / (1.0 + np.exp((energy_to_joule(E) - energy_to_joule(mu)) / (k_B * T_kelvin)))
            psi = np.abs(eigvecs[:, idx])**2
            # normalize psi: eigenvectors from sparse solver are mass-scaled; approximate normalize by integral
            psi_norm = psi / (np.sum(psi) * dx)
            n += occ * psi_norm  # electrons per m
        # convert n (1D per unit length) to volumetric density estimate by dividing by a nominal cross-section (1e-18 m^2)
        nominal_area = 1e-18
        rho = -e * (n / nominal_area) + doping_Cperm3  # C/m^3
        # solve Poisson for phi (V)
        phi = poisson_solve_1d(rho, dx_nm, eps_rel=3.9, phi_left=mu_left_eV, phi_right=mu_right_eV)
        V_new = -phi  # potential energy in eV (approx)
        # mixing for stability
        mix = 0.3
        V = (1-mix)*V + mix*V_new
        diff = np.max(np.abs(V - V_new))
        if diff < tol_eV:
            break
    return V, phi, k+1

# ----------------------------
# 6) Monte Carlo sampling & stats
# ----------------------------
def monte_carlo_barrier_transmission(E_target_eV, V0_eV, t0_nm, sigma_nm, N_samples, m_eff=0.25, trunc=3.0):
    rng = np.random.default_rng(123456)
    lower = -trunc * sigma_nm
    upper = trunc * sigma_nm
    samples = []
    while len(samples) < N_samples:
        draw = rng.normal(0.0, sigma_nm)
        if lower <= draw <= upper:
            samples.append(draw)
    samples = np.array(samples)
    thicknesses = t0_nm + samples
    Tvals = np.array([tmm_transmission_rectangular(E_target_eV, V0_eV, t, m_eff) for t in thicknesses])
    meanT = np.mean(Tvals)
    stdT = np.std(Tvals, ddof=1)
    skew = ((np.mean((Tvals-meanT)**3)) / (stdT**3)) if stdT>0 else 0.0
    ci95 = 1.96 * stdT / np.sqrt(N_samples)
    stats = {'mean': meanT, 'std': stdT, 'skewness': skew, 'ci95': ci95, 'N': N_samples}
    return thicknesses, Tvals, stats

# ----------------------------
# 7) Phenomenological Lorentzian broadening
# ----------------------------
def lorentzian_broaden_T(E_eV, T_of_E, Gamma_eV):
    """
    Convolve T(E) with Lorentzian kernel of width Gamma_eV to approximate inelastic broadening.
    """
    # construct Lorentzian kernel on same E grid
    dE = E_eV[1] - E_eV[0]
    ecenter = E_eV
    half_range = ecenter[-1] - ecenter[0]
    kernel_x = np.linspace(-half_range, half_range, len(E_eV))
    gamma = Gamma_eV
    L = (1.0/np.pi) * (gamma / (kernel_x**2 + gamma**2))
    L /= np.sum(L)  # normalize
    T_conv = fftconvolve(T_of_E, L, mode='same')
    return T_conv

# ----------------------------
# 8) Utilities for mapping code -> paper figures
# ----------------------------
def produce_T_vs_E_TMM(V0_eV, t_nm, params: DeviceParams):
    E = np.arange(params.E_min, params.E_max + params.dE, params.dE)
    T = np.array([tmm_transmission_rectangular(e, V0_eV, t_nm, params.m_eff) for e in E])
    return E, T

def produce_I_V_from_T(E_eV, T_of_E, Vbias_array, params: DeviceParams):
    Ivals = np.array([landauer_current_density(Vb, T_of_E, E_eV, params.T_kelvin) for Vb in Vbias_array])
    return Ivals

# ----------------------------
# 9) Demo main: reproduce core workflows and save results
# ----------------------------
def demo_run_and_save(outdir='results_demo'):
    os.makedirs(outdir, exist_ok=True)
    print("Running demo verification suite. Output ->", outdir)
    # parameters chosen to match manuscript examples
    params = DeviceParams(V0_eV=0.5, t0_nm=1.8, m_eff=0.067,
                          T_kelvin=300.0, dE=0.001, E_min=0.0, E_max=1.5, dx_nm=0.005)
    # (A) T(E) for RTD-like double barrier using multilayer TMM
    E_grid = np.arange(params.E_min, params.E_max + params.dE, params.dE)
    # build regions for double-barrier
    def regions_for_RTD(Gamma_eV=0.0):
        Vbar = params.V0_eV + 1j * (Gamma_eV/2.0)
        regions = [
            (0.0+0j, 0.0),
            (Vbar, params.t0_nm),
            (0.0+0j, 4.0),
            (Vbar, params.t0_nm),
            (0.0+0j, 0.0)
        ]
        return regions
    Gamma_vals = [0.0, 0.01, 0.03, 0.07]
    Tcurves = {}
    for G in Gamma_vals:
        regs = regions_for_RTD(G)
        Tcurves[G] = np.array([tmm_multilayer(E, regs, params.m_eff) for E in E_grid])
    # save T curves as .npz
    np.savez(os.path.join(outdir, 'Tcurves_RTD.npz'), E=E_grid, **{f'G{int(1000*G)}': Tcurves[G] for G in Gamma_vals})
    print("Saved RTD T(E) curves.")
    # (B) Monte Carlo histogram at target energy (matching Fig A2 e.g., E=0.70 eV)
    E_target = 0.70
    thicknesses, Tvals, stats = monte_carlo_barrier_transmission(E_target, V0_eV=0.9, t0_nm=1.0,
                                                                 sigma_nm=0.02, N_samples=5000, m_eff=0.25)
    np.savetxt(os.path.join(outdir,'mc_thicknesses.txt'), thicknesses)
    np.savetxt(os.path.join(outdir,'mc_Tvals.txt'), Tvals)
    with open(os.path.join(outdir, 'mc_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print("Saved Monte Carlo data and stats:", stats)
    # (C) Generate nominal T(E) for single barrier and I-V curve (example)
    params_single = DeviceParams(V0_eV=0.9, t0_nm=1.0, m_eff=0.25, T_kelvin=300.0, dE=0.001, E_min=-0.5, E_max=1.5)
    E_single, T_single = produce_T_vs_E_TMM(params_single.V0_eV, params_single.t0_nm, params_single)
    Vbias = np.linspace(0.0, 0.5, 51)
    Ivals = produce_I_V_from_T(E_single, T_single, Vbias, params_single)
    np.savetxt(os.path.join(outdir, 'T_single_barrier.txt'), np.column_stack((E_single, T_single)), header='E(eV)  T(E)')
    np.savetxt(os.path.join(outdir, 'IV_single_barrier.txt'), np.column_stack((Vbias, Ivals)), header='V(V)  I(A/m^2)')
    print("Saved nominal single-barrier T(E) and I-V.")
    # (D) Optional: Lorentzian broaden example
    T_lor = lorentzian_broaden_T(E_single, T_single, Gamma_eV=0.02)
    np.savetxt(os.path.join(outdir, 'T_single_broadened_gamma20meV.txt'), np.column_stack((E_single, T_lor)), header='E(eV)  T_broadened')
    print("Saved Lorentzian broadened T(E).")
    # (E) Minimal Schrödinger-Poisson run (toy)
    # Prepare initial flat potential array (in eV)
    Ngrid = int(10.0 / params_single.dx_nm)  # 10 nm device for demo
    V_init = np.zeros(Ngrid) + params_single.V0_eV * 0.5
    doping = np.zeros(Ngrid)  # neutral for demo
    V_sc, phi, iters = schrodinger_poisson_1d(V_init, params_single.dx_nm, params_single.m_eff,
                                             doping_Cperm3=np.zeros(Ngrid), mu_left_eV=0.0, mu_right_eV=0.0,
                                             max_iter=20, tol_eV=params_single.convergence_tol_eV)
    np.savetxt(os.path.join(outdir, 'V_sc.txt'), V_sc)
    print("Ran toy Schrödinger-Poisson loop (iters):", iters)
    print("Demo run complete. Files in:", outdir)

# ----------------------------
# If run as script, execute demo
# ----------------------------
if __name__ == "__main__":
    demo_run_and_save()
    print("All demo outputs saved. See README and use functions in this module to reproduce specific figures/tables.")
