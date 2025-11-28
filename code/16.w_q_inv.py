import numpy as np
import os
from scipy.special import sph_harm_y
import time

# پارامترها
critical_timesteps = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
l_max = 30
s = 0  # فقط اسپین صفر برای بازسازی فعلی
mode_filter = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]  # مثال از مودهای مجاز

# شبکه زاویه‌ای
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2 * np.pi, n_phi)
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

# مسیرها
alm_dir = "spectral_alm_output"
metric_dir = "metric_output"
qinv_dir = "q_inv_output"
winv_dir = "w_inv_output"
os.makedirs(qinv_dir, exist_ok=True)
os.makedirs(winv_dir, exist_ok=True)

# آماده‌سازی پایه‌ Y_{lm}
Y_lm_basis = []
for l in range(0, l_max + 1):
    for m in range(-l, l + 1):
        if (l, abs(m)) not in mode_filter:
            continue
        Ylm = sph_harm_y(m, l, phi_grid, theta_grid)  # complex
        Y_lm_basis.append((l, m, Ylm))

n_modes = len(Y_lm_basis)

for t in critical_timesteps:
    print(f"\n⏳ بازسازی Q_inv و w_inv برای t={t}...")
    t0 = time.time()

    # ایجاد خروجی‌های حافظه‌نگاشت
    Q_inv = np.memmap(os.path.join(qinv_dir, f"Q_inv_t{t}.npy"), dtype=np.float64, mode="w+",
                      shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    w_inv = np.memmap(os.path.join(winv_dir, f"w_inv_t{t}.npy"), dtype=np.float64, mode="w+",
                      shape=(n_chi, n_theta, n_phi))

    # بارگذاری متریک عددی
    g = np.memmap(os.path.join(metric_dir, f"g_t{t}.npy"), dtype=np.float32, mode="r",
                  shape=(n_coords, n_coords, n_chi, n_theta, n_phi))

    for mu in range(n_coords):
        for nu in range(n_coords):
            # بارگذاری alm
            alm_path = os.path.join(alm_dir, f"alm_q_t{t}_s{s}_mu{mu}_nu{nu}.npz")
            data = np.load(alm_path)
            alm = data['alm']  # (chi, n_modes)

            # بازسازی Q^{(inv)}_{μν}
            for chi in range(n_chi):
                q_recon = np.zeros((n_theta, n_phi), dtype=np.complex128)
                for idx in range(len(Y_lm_basis)):
                    coeff = alm[chi, idx]
                    q_recon += coeff * Y_lm_basis[idx][2]  # (θ, φ)

                Q_inv[mu, nu, chi] = q_recon.real

    Q_inv.flush()

    # محاسبه w_inv = g^{μν} Q^{(inv)}_{μν}
    block_size = 10
    for i_start in range(0, n_chi, block_size):
        i_end = min(i_start + block_size, n_chi)

        g_blk = g[:, :, i_start:i_end].astype(np.float64)
        Q_blk = Q_inv[:, :, i_start:i_end]

        ginv_blk = np.zeros_like(g_blk)
        for i in range(i_end - i_start):
            for j in range(n_theta):
                for k in range(n_phi):
                    try:
                        ginv_blk[:, :, i, j, k] = np.linalg.inv(g_blk[:, :, i, j, k])
                    except np.linalg.LinAlgError:
                        ginv_blk[:, :, i, j, k] = 0.0

        w_blk = np.einsum("abijk,abijk->ijk", ginv_blk, Q_blk)
        w_blk = np.nan_to_num(w_blk, nan=0.0, posinf=0.0, neginf=0.0)  # ایمنی عددی
        w_inv[i_start:i_end] = w_blk

    w_inv.flush()
    print(f"✅ Q_inv و w_inv برای t={t} ذخیره شدند ({time.time() - t0:.2f}s)")