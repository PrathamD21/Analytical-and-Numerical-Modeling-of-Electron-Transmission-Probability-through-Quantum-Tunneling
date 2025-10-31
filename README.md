# Analytical-and-Numerical-Modeling-of-Electron-Transmission-Probability-through-Quantum-Tunneling
QuantumTransportLab is a Python-based simulation framework for modeling quantum tunneling in nanoscale devices. This framework reproduces key plots, numerical values, and validation workflows used in the GaAs/AlGaAs Resonant Tunneling Diode (RTD) study — including transmission spectra, I–V characteristics, inelastic broadening, and Monte Carlo statistics.

⚙️ Features and Capabilities

✅ Transfer Matrix Method (TMM)

Handles both single and multilayer (RTD-like) quantum structures.

Implements complex potential extension for inelastic broadening (Γ model).

Produces accurate T(E) transmission spectra across 0–1.5 eV.

✅ Finite-Difference Schrödinger Equation Solver

Computes bound state eigenenergies and wavefunctions for arbitrary potential profiles.

Uses SciPy’s sparse eigenvalue solver for efficiency.

✅ WKB Approximation

Implements semi-analytical tunneling transmission for smoothly varying barriers.

Useful for quick validation of numerical TMM results.

✅ Self-Consistent Schrödinger–Poisson Solver (Toy Model)

Iterative coupling between charge density and potential.

Solves 1D Poisson equation via finite differences with Dirichlet BCs.

✅ Landauer Formalism for Current–Voltage (I–V)

Computes tunneling current density using integrated transmission spectra.

Includes temperature dependence and Fermi–Dirac occupations.

✅ Monte Carlo Statistical Variation

Models fabrication-induced barrier width fluctuations (Gaussian sampling, N = 5000).

Outputs mean, standard deviation, skewness, and 95 % confidence intervals for transmission.

✅ Phenomenological Inelastic Scattering (Lorentzian Broadening)

Adds imaginary potential iΓ/2 or Lorentzian convolution to simulate phonon-assisted dephasing.

Replicates resonance broadening and coherence loss in experimental RTDs.

✅ Automated Verification Demo

Reproduces paper’s key graphs and outputs all results to a local /results_demo directory.

✅ Convergence and Validation Tests

Benchmarks numerical solvers against analytical results.
Includes automatic mesh refinement and energy-grid convergence checks.

📊 Implemented Models

| Model / Equation      | Method Used                  | Description                           |
| --------------------- | ---------------------------- | ------------------------------------- |
| Schrödinger Equation  | Finite Difference (FD)       | Eigenvalue solver for confined states |
| Transmission (T(E))   | Transfer Matrix Method (TMM) | Multi-barrier RTD tunneling           |
| WKB Approximation     | Semi-Analytical Integration  | Analytical validation                 |
| Poisson Equation      | Finite-Difference Solver     | Electrostatic potential solution      |
| Current Density (I–V) | Landauer–Büttiker Formalism  | Quantum current integration           |
| Statistical Variation | Monte Carlo Sampling         | Randomized barrier width analysis     |
| Inelastic Scattering  | Lorentzian Convolution / iΓ  | Dephasing and resonance broadening    |


🧩 Directory Structure

QuantumTunnelingVerification/
│
├── tunneling_verification_suite.py     # Main verification and simulation script
├── README.md                           # Project documentation (this file)
│
├── results_demo/                       # Auto-generated demo output directory
│   ├── T_RTD_nominal.png               # Transmission Spectrum – RTD (V0=0.5 eV, b=1.8 nm, w=4 nm)
│   ├── IV_RTD_illustrative.png         # Illustrative I–V curve (Landauer integration)
│   ├── T_single_barrier.txt            # T(E) data for single-barrier structure
│   ├── IV_single_barrier.txt           # I–V data (single barrier)
│   ├── mc_thicknesses.txt              # Monte Carlo thickness samples
│   ├── mc_Tvals.txt                    # Monte Carlo transmission samples
│   ├── mc_stats.json                   # Monte Carlo statistical summary
│   ├── T_RTD_G10meV.txt ... G70meV.txt # T(E) for Γ=10–70 meV broadenings
│   ├── V_sc.txt                        # Toy Schrödinger–Poisson convergence potential
│   └── T_single_broadened_gamma20meV.txt
│
└── data/ (optional)                    # Placeholder for user-provided parameter sets or validation datasets


🧠 Theoretical Foundations

The framework integrates multiple established quantum transport theories:

Transfer Matrix Method (TMM) – for exact coherent transmission across layered barriers.

WKB Approximation – for analytical benchmarking and asymptotic scaling.

Landauer–Büttiker Formalism – for bias-dependent tunneling current integration.

Schrödinger–Poisson Coupling – for potential self-consistency (toy demonstration).

Complex-Potential Dephasing – following Datta’s phenomenological model for inelastic scattering.


🚀 How to Run

1️⃣ Requirements
Install Python and dependencies:
pip install numpy scipy matplotlib

2️⃣ Execute the main simulation
python quantum_transport_lab.py

3️⃣ Outputs
The demo automatically generates and saves:

Transmission Spectrum:
T_RTD_nominal.png – for GaAs/AlGaAs RTD (V₀ = 0.5 eV, barrier = 1.8 nm, well = 4.0 nm).

I–V Characteristic:
IV_RTD_illustrative.png – using Landauer formalism with bias shift.

Monte Carlo Statistics:
Mean ± 95 % CI of T(E) under barrier-width fluctuation.

Inelastic Broadening Variants:
Lorentzian-convolved spectra for Γ = 10–70 meV.

Toy Schrödinger–Poisson Profile:
Converged potential stored in V_sc.txt.



🧑‍💻 Citation

If you use this code, please cite the associated IJISRT paper:
Pratham Dungarani, “Analytical and Numerical Modeling of Electron Transmission Probability through Quantum Tunneling Barriers in Nanoscale Diodes,” IJISRT, Vol. X, Issue Y, 2025.
