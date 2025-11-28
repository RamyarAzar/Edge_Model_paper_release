import numpy as np
from scipy.special import sph_harm_y
import os

# پارامترها
L_MAX = 6
R_0 = 1.0
sigma = 1.0
n_chi = 400
n_theta = 400
n_phi = 400
critical_timesteps = [1, 2, 3, 9, 10, 11, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]

# مسیرها
alm_folder = "alm_output"
output_folder = "metric_output"
os.makedirs(output_folder, exist_ok=True)

# شبکه‌ها
chi = np.linspace(0, np.pi, n_chi)
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2 * np.pi, n_phi)
chi_grid, theta_grid, phi_grid = np.meshgrid(chi, theta, phi, indexing='ij')
dchi = chi[1] - chi[0]
sin_theta_grid = np.sin(theta_grid)

# حلقه روی زمان‌ها
for t in critical_timesteps:
    print(f"⏳ Processing t = {t}")
    alm_path = os.path.join(alm_folder, f"alm_t{t}_interp.npy")
    if not os.path.exists(alm_path):
        print(f"⚠️ Missing {alm_path}")
        continue

    alm = np.load(alm_path)
    R_field = np.zeros_like(chi_grid, dtype=np.float32)

    for l in range(L_MAX + 1):
        for m in range(-l, l + 1):
            a_lm = alm[l, m + L_MAX]
            Y_lm = sph_harm_y(m, l, phi_grid, theta_grid)
            modulated = np.sin(chi_grid) * np.exp(-chi_grid**2 / sigma**2)
            R_field += a_lm * np.real(Y_lm) * modulated

    R_field = R_0 * (1 + R_field).astype(np.float32)
    np.save(os.path.join(output_folder, f"R_t{t}.npy"), R_field)

    # ∂R/∂χ عددی
    dR_dchi = np.gradient(R_field, dchi, axis=0)

    # ساخت g
    g = np.zeros((4, 4, n_chi, n_theta, n_phi), dtype=np.float32)
    g[0, 0, ...] = -1
    g[1, 1, ...] = dR_dchi**2
    g[2, 2, ...] = R_field**2
    g[3, 3, ...] = R_field**2 * sin_theta_grid**2

    np.save(os.path.join(output_folder, f"g_t{t}.npy"), g)
    print(f"✅ Saved R_t{t}.npy and g_t{t}.npy")

print("🎯 تمام زمان‌های بحرانی محاسبه شدند.")