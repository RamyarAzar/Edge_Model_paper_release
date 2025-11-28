import numpy as np
import os
from scipy.special import sph_harm
import matplotlib.pyplot as plt

L_MAX = 6
n_theta = 400
n_phi = 400
chi_idx = 200
q_folder = "q_parts"
output_folder = "alm_output"
os.makedirs(output_folder, exist_ok=True)

critical_timesteps = [1, 2, 3, 9, 10, 11, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2 * np.pi, n_phi)
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
dtheta = theta[1] - theta[0]
dphi = phi[1] - phi[0]
sin_theta = np.sin(theta_grid)

def extract_alm_from_q(q00_slice):
    alm = np.zeros((L_MAX + 1, 2 * L_MAX + 1), dtype=np.complex64)
    for l in range(L_MAX + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi_grid, theta_grid)
            integrand = q00_slice * np.conj(Y_lm) * sin_theta
            integral = np.sum(integrand) * dtheta * dphi
            alm[l, m + L_MAX] = integral
    return np.real(alm).astype(np.float32)

for t in critical_timesteps:
    qfile = os.path.join(q_folder, f"q_t{t}.npy")
    Q = np.load(qfile).astype(np.float32)
    q00 = Q[0, 0, chi_idx, :, :]
    alm_real = extract_alm_from_q(q00)
    np.save(os.path.join(output_folder, f"alm_t{t}.npy"), alm_real)

    plt.figure(figsize=(8, 5))
    plt.imshow(alm_real, cmap='coolwarm', origin='lower',
               extent=[-L_MAX, L_MAX, 0, L_MAX])
    plt.xlabel('m')
    plt.ylabel('ℓ')
    plt.title(f"a_{{ℓm}} coefficients from Q00 at t={t}, χ={chi_idx}")
    plt.colorbar(label="Re[a_{ℓm}]")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"alm_t{t}_heatmap.png"))
    plt.close()