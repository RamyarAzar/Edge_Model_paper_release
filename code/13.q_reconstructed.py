import numpy as np
import os
from scipy.special import sph_harm_y
import time

# پارامترها
critical_timesteps = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
l_max = 20

# شبکه زاویه‌ای
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2*np.pi, n_phi)
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')  # shape (n_theta, n_phi)

# مسیر فایل‌ها
alm_dir = "spectral_alm_output"
output_dir = "q_reconstructed"
os.makedirs(output_dir, exist_ok=True)

# ایجاد پایه‌ی Ylmها فقط یک بار (برای s=0)
print("⏳ Precomputing spin-0 spherical harmonics...")
Ylm_dict = {}
for l in range(l_max + 1):
    for m in range(-l, l + 1):
        Ylm_dict[(l, m)] = sph_harm_y(m, l, phi_grid, theta_grid)  # shape: (n_theta, n_phi)

# بازسازی برای همه زمان‌ها
for t in critical_timesteps:
    print(f"\n🔁 Reconstructing Q for t={t}")
    Q_recon = np.memmap(
        os.path.join(output_dir, f"Q_recon_t{t}.npy"),
        dtype=np.complex128, mode='w+',
        shape=(n_coords, n_coords, n_chi, n_theta, n_phi)
    )

    for mu in range(n_coords):
        for nu in range(n_coords):
            fname = f"alm_q_t{t}_s0_mu{mu}_nu{nu}.npz"
            fpath = os.path.join(alm_dir, fname)

            if not os.path.exists(fpath):
                print(f"⚠️ File not found: {fname}")
                continue

            data = np.load(fpath)
            alm = data['alm']  # shape: (n_chi, n_modes)
            l_list = data['l']
            m_list = data['m']
            n_modes = len(l_list)

            for chi in range(n_chi):
                f_recon = np.zeros((n_theta, n_phi), dtype=np.complex128)
                for idx in range(n_modes):
                    l = l_list[idx]
                    m = m_list[idx]
                    f_recon += alm[chi, idx] * Ylm_dict[(l, m)]
                Q_recon[mu, nu, chi] = f_recon

    Q_recon.flush()
    print(f"✅ Saved: Q_recon_t{t}.npy")

print("\n🏁 Reconstruction finished for all timesteps.")