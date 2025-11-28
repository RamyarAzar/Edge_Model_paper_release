# Emergent Dynamic Geometry (EDGE):
## A Geometric Framework for Quantum-like Behavior and Physical Constants

**Author:** Ramyar Azar  
**Last updated:** November 2025  

This repository contains the core simulation and analysis code for the **Emergent Dynamic Geometry (EDGE)** model: a purely geometric framework where quantum-like behavior, gauge fields, and effective physical constants emerge from the curvature dynamics of a time-evolving hyperspherical spacetime. :contentReference[oaicite:0]{index=0}  

The code here mirrors the assets archived on Zenodo (DOI `10.5281/zenodo.15873778`) and is organized for **reproducibility, inspection, and extension**. :contentReference[oaicite:1]{index=1}  

---

## Snapshot

### Field
- Theoretical physics  
- Emergent geometry & quantum foundations  

### Model type
- 4D orbital hypersphere embedded in a curved ambient space  
- Geometry → scalar field → gauge sector → emergent constants  
- Modal / spectral analysis and Schrödinger-like dynamics  

### Stage
- Full numerical prototype  
- High-resolution simulations on a  
  \(400 \times 400 \times 400 \times 4 \times 4\) lattice, 50 time steps,  
  with ~86 Python modules and large tensor archives   

### Main questions

1. Can a **4D hyperspherical spacetime** governed only by a radial field \(R(\chi,\theta,t)\) remain dynamically stable (bounded curvature, no blow-ups)? :contentReference[oaicite:3]{index=3}  
2. Can we generate a scalar field \(w(x,t)\) from this geometry and recover **quantum-like structure** (modes, vortices, modons, effective Hamiltonian)? :contentReference[oaicite:4]{index=4}  
3. Can **gauge-like fields** \(A_\mu = \nabla_\mu \arg w\) and field tensor \(F_{\mu\nu}\) emerge purely from the scalar phase? :contentReference[oaicite:5]{index=5}  
4. Can we extract effective constants \(\hbar_{\text{eff}}(t)\), \(c_{\text{eff}}(t)\), and \(\lambda(t)\) which stabilize in time and support **forward and inverse reconstruction** of the dynamics? :contentReference[oaicite:6]{index=6}  

---

## Conceptual overview

EDGE treats observed spacetime as a **4D hypersphere** \(\Sigma_4\) with coordinates  
\((\chi,\theta,\phi,t)\), embedded in a mildly curved higher-dimensional background. The geometry is encoded in a single radial field \(R(\chi,\theta,t)\) with angular harmonics and Gaussian modulation. :contentReference[oaicite:7]{index=7}  

From this starting point, the pipeline is:

1. **Geometry:** build the induced metric \(g_{\mu\nu}\), Christoffel symbols, Ricci tensor, and extrinsic curvature from the embedding \(X^A(x^\mu)\). :contentReference[oaicite:8]{index=8}  
2. **Scalar:** construct a scalar field \(w(x,t)\) from gradients of \(R\), then promote it to a complex field \(|w|e^{i\phi}\) to expose phase topology (vortices, modons, quantized circulation). :contentReference[oaicite:9]{index=9}  
3. **Modal structure:** decompose \(w(x,t)\) into modes \(c_k(t)\psi_k(x)\) on the hypersphere, define an effective Hamiltonian, and study spectral energetics. :contentReference[oaicite:10]{index=10}  
4. **Gauge sector:** define a gauge-like vector field from phase gradients and compute the field tensor \(F_{\mu\nu}\), energy density, and entropy-like measures. :contentReference[oaicite:11]{index=11}  
5. **Emergent constants:** extract \(\lambda(t)\), \(c_{\text{eff}}(t)\), and \(\hbar_{\text{eff}}(t)\) from geometric and modal time-series, then test whether they stabilize and support Schrödinger-like evolution and inverse scaling. :contentReference[oaicite:12]{index=12}  

---

## Numerical setup

Key features of the simulation:   

- **Grid:**  
  \(400_\chi \times 400_\theta \times 400_\phi \times 4_\xi \times 4_\zeta\) (space)  
  with \(N_t = 50\) time steps.  
- **Resolution:** fixed \(\Delta\chi, \Delta\theta, \Delta t\) tuned for stability.  
- **Discretization:** high-order finite differences for derivatives; log-safe formulas for numerically delicate quantities.  
- **Data:** arrays stored as `.npy` / memmap for scalability; summaries and plots in `.txt` / `.png`.  
- **Codebase:** ~86 Python modules grouped by task (geometry, scalar, gauge, spectral, constants, diagnostics).  

---

## Repository contents

> Adjust names to your actual folder structure if needed. Layout follows the Zenodo archive grouping. :contentReference[oaicite:14]{index=14}  

```text
README.md                      # This file

src/
  geometry/
    build_embedding.py         # X^A(x^μ) and R(χ,θ,t)
    build_metric.py            # g_{μν} construction
    christoffel.py             # Γ^λ_{μν}
    ricci.py                   # R_{μν}
    extrinsic_curvature.py     # K_{μν}
  scalar/
    scalar_from_R.py           # w(x,t) from ∇R, ∂_t R
    scalar_phase.py            # |w|, arg w, vortices/modons
  modal/
    modal_decomposition.py     # ψ_k(x), c_k(t)
    modal_energetics.py        # E_k(t), H_eff(t)
  gauge/
    gauge_from_phase.py        # A_μ(x,t)
    field_tensor.py            # F_{μν} and YM-like density
    gauge_entropy.py           # entropy / order diagnostics
  constants/
    ceff_lambda_extraction.py  # c_eff(t,χ,θ), λ(t)
    hbar_extraction.py         # τ(t), ħ_eff(t)
    inverse_scaling.py         # γ_c, γ_ħ, γ_E tests
  reconstruction/
    forward_reconstruction.py  # reproduce A_μ, w from geometry
    inverse_reconstruction.py  # fixed constants → reverse evolution
  pipeline/
    run_full_pipeline.py       # orchestrates full EDGE run

config/
  params.yaml                  # grid, Δt, model parameters, paths

data/
  metric_output/
  christoffel_output/
  ricci_output/
  k_output/
  w_output/
  phase_analysis_outputs/
  gauge_spatial_output/
  fmnunu_output/
  mode_decomposition_output/
  spectral_analysis_outputs/
  effective_field_output/
  effective_lagrangian_components/
  lagrangian_outputs/
  lagrangian_phase/
  lagrangian_phase_recon/
  lagrangian_final_output/
  hamiltonian_phase315/
  hamiltonian_phase315_nograd/
  ceff_output/
  ceff_chi_analysis/
  t_output/
  t_output_phase43/
  t_components_output/
  t_mask_output/
  qft_phase_analysis_output/
  weak_curvature_analysis/
  topology_analysis/
  diff_output/
  validation_outputs/

plots/
  ricci_time_series.png
  w_mean_timeseries.png
  modal_spectra.png
  gauge_entropy_timeseries.png
  emergent_constants_trends.png
  reconstruction_overlap.png
```
## How to run

These scripts assume that the full EDGE geometry (metric, curvature, scalar field) is generated within this pipeline.  
For the original \(400^3 \times 4 \times 4 \times 50\) runs you will need substantial RAM and disk; for experimentation, use smaller grids in `config/params.yaml`.

### 1. Set up environment

Create and activate a Python environment (e.g. `venv` or `conda`), then install dependencies:

```bash
pip install -r requirements.txt
```
### 2. (Optional) Adjust parameters

Edit `config/params.yaml` to change, for example:

- grid size and number of angular modes,
- number of time steps,
- parameters of the radial field \(R(\chi,\theta,t)\),
- paths for input/output arrays,
- which sub-pipelines (geometry, scalar, gauge, constants) to run.

### 3. Run full pipeline

```bash
python src/pipeline/run_full_pipeline.py
```
## Relation to the EDGE paper

This repository implements the simulations reported in:

> **“Emergent Dynamic Geometry: A Geometric Framework for Quantum-like Behavior and Physical Constants”** (submitted).

The main article presents:

- the geometric foundation (orbital hypersphere, embedding, metric, curvature),
- scalar field extraction and modal structure,
- gauge fields from phase gradients,
- emergent constants and Schrödinger-like dynamics,
- numerical tests of stability, reconstruction, and observational implications.

The companion appendix details how the code and outputs are structured for reproducibility and long-term reuse.

---

## Zenodo archive

All simulation assets—code snapshot, large `.npy` arrays, and figures—are also archived on Zenodo:

- **Simulation & code archive (Zenodo):** DOI `10.5281/zenodo.15873778`  
  (restricted license; reuse only by explicit agreement with the author).

You can treat this GitHub repository as the **live, developer-friendly view**, and the Zenodo record as the **frozen, citable snapshot** associated with the manuscript.

---

## License & usage

This repository is provided as a **research demonstration and case study**.

- Use is limited to **personal, academic, and non-commercial** purposes.
- Do **not** integrate this code into clinical, diagnostic, or commercial products.
- Redistribution of modified or unmodified versions is **not permitted** without explicit written permission from the author.

If you are interested in collaboration, extended access, or reuse under a specific agreement (e.g. NDA or joint project), please contact the author.
