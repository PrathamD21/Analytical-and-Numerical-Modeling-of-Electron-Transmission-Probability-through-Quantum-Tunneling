# Analytical-and-Numerical-Modeling-of-Electron-Transmission-Probability-through-Quantum-Tunneling
QuantumTransportLab is a Python-based simulation framework for modeling quantum tunneling in nanoscale devices. Implements Schrödinger–Poisson, Transfer Matrix, Landauer, and Monte Carlo methods to analyze MIM, RTD, and TFET structures with inelastic scattering and statistical variability.

⚙️ Features and Capabilities

✅ 1D Schrödinger Equation Solver (Finite Difference)

Computes eigenenergies and wavefunctions for arbitrary potential profiles.
Supports multiple boundary conditions and spatial discretizations.

✅ Transfer Matrix Method (TMM)

Handles arbitrary multilayer and heterostructure systems.
Includes effective mass discontinuity matching at material interfaces.
Produces energy-resolved transmission spectra T(E).

✅ WKB Approximation

Implements analytical tunneling solutions for smooth or slowly varying potentials.
Provides fast validation and benchmarking against numerical solvers.

✅ Self-Consistent Schrödinger–Poisson Solver

Iteratively solves charge density and electrostatic potential profiles.
Uses finite-difference Poisson solver with SciPy.spsolve (LU decomposition).

✅ Landauer Formalism for Current Evaluation

Calculates tunneling current density using energy-integrated transmission.
Supports temperature dependence via Fermi–Dirac statistics.

✅ Monte Carlo Statistical Variation

Models fabrication-induced barrier width fluctuations.
Performs random sampling (Gaussian N=5000) and outputs statistical spread in transmission and current.

✅ Phenomenological Inelastic Scattering (Γ Model)

Adds imaginary potential iΓ/2 to emulate phonon-assisted dephasing.
Reproduces resonance broadening and loss of coherence.

✅ Convergence and Validation Tests

Benchmarks numerical solvers against analytical results.
Includes automatic mesh refinement and energy-grid convergence checks.

📊 Implemented Models

Model	Method	Description

Schrödinger Eq.

TMM	

WKB	

Poisson Eq.	

Landauer

Monte Carlo

Γ Model

🧩 Directory Structure

QuantumTransportLab/

│

├── quantum_transport_lab.py # Main simulation framework

├── README.md # Project documentation

├── results/

│ ├── RTD_T_E.png # RTD Transmission vs Energy (with Γ)

│ ├── poisson_profile.png # Example Poisson potential solution

│ ├── montecarlo_histogram.png # Monte Carlo T(E) distribution

│ └── validation_curves.png # Analytical vs TMM validation

└── data/ # Optional data files or parameter sets

🧠 Theoretical Foundations

The framework integrates multiple established quantum transport theories:

Transfer Matrix Method (TMM) for multilayer tunneling structures.

Wentzel–Kramers–Brillouin (WKB) approximation for analytical scaling.

Landauer–Büttiker formalism for current–voltage (I–V) calculations.

Self-consistent Schrödinger–Poisson coupling for charge–potential feedback.

Phenomenological dephasing model (Datta, 2005) via complex potential broadening.


🚀 How to Run

1️⃣ Requirements
Install Python and dependencies:
pip install numpy scipy matplotlib

2️⃣ Execute the main simulation
python quantum_transport_lab.py

3️⃣ Outputs
Transmission curves (RTD_T_E.png)
Poisson potential profile (poisson_profile.png)
Convergence and validation reports (printed in console)
Monte Carlo statistics and histogram


🧑‍💻 Citation

If you use this code, please cite the associated IJISRT paper:
Pratham Dungarani, “Analytical and Numerical Modeling of Electron Transmission Probability through Quantum Tunneling Barriers in Nanoscale Diodes,” IJISRT, Vol. X, Issue Y, 2025.
