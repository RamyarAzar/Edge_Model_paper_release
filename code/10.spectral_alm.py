import os
import numpy as np
from scipy.special import sph_harm_y
from tqdm import tqdm
import time

# پارامترها
critical_timesteps = [33, 36, 39, 42, 45, 47]
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
block_size = 10
l_max = 20
spin_weights = [0, 1, -1, 2, -2]

# شبکه زاویه‌ای
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2*np.pi, n_phi)
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

# مسیر فایل‌ها
q_dir = "q_output"
output_dir = "spectral_alm_output"
os.makedirs(output_dir, exist_ok=True)

# ساخت پایه هارمونیک‌ها با وزن اسپین s
def generate_spin_harmonics_basis(l_max, s):
    basis = []
    for l in range(abs(s), l_max + 1):
        for m in range(-l, l + 1):
            Y_slm = sph_harm_y(m, l, s, phi_grid, theta_grid)
            basis.append((l, m, Y_slm))  # complex-valued
    return basis

for t in critical_timesteps:
    print(f"\n🔁 تحلیل طیفی Q برای t={t}")
    Q = np.memmap(os.path.join(q_dir, f"Q_t{t}.npy"), dtype=np.float64, mode='r',
                  shape=(n_coords, n_coords, n_chi, n_theta, n_phi))

    for s in spin_weights:
        print(f"  ↪ اسپین وزنی s={s}")
        basis = generate_spin_harmonics_basis(l_max, s)
        n_modes = len(basis)
        l_list = np.array([l for (l, m, _) in basis])
        m_list = np.array([m for (l, m, _) in basis])

        for mu in range(n_coords):
            for nu in range(n_coords):
                alm_all = np.zeros((n_chi, n_modes), dtype=np.complex128)

                for chi in tqdm(range(n_chi), desc=f"    μ={mu}, ν={nu}", leave=False):
                    f = Q[mu, nu, chi]  # f(θ, φ)

                    for idx, (l, m, Yslm) in enumerate(basis):
                        integrand = f * np.conj(Yslm) * np.sin(theta_grid)
                        integral = np.sum(integrand) * (np.pi / n_theta) * (2*np.pi / n_phi)
                        alm_all[chi, idx] = integral

                # ذخیره به صورت npz (فشرده و شامل ℓ, m)
                filename = os.path.join(output_dir, f"alm_q_t{t}_s{s}_mu{mu}_nu{nu}.npz")
                np.savez_compressed(filename, alm=alm_all, l=l_list, m=m_list)
                print(f"    ✔ ذخیره شد → {filename}")
