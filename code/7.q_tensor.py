import numpy as np
import os
import time

# تنظیمات کلی
critical_timesteps = [1, 2, 3, 9, 10, 11, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
block_size = 10

# مسیر فایل‌ها
g_dir = "metric_output"
Ricci_dir = "ricci_output"
K_dir = "k_tensor_output"
output_dir = "q_output"
os.makedirs(output_dir, exist_ok=True)

for t in critical_timesteps:
    print(f"\n⏳ Computing Q_μν for t={t}...")
    t0 = time.time()

    # بارگذاری داده‌ها به صورت حافظه‌نگاشت
    g = np.memmap(os.path.join(g_dir, f"g_t{t}.npy"), dtype=np.float32, mode='r', shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    Ricci = np.memmap(os.path.join(Ricci_dir, f"Ricci_t{t}.npy"), dtype=np.float32, mode='r', shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    Rscalar = np.memmap(os.path.join(Ricci_dir, f"Rscalar_t{t}.npy"), dtype=np.float32, mode='r', shape=(n_chi, n_theta, n_phi))
    K = np.memmap(os.path.join(K_dir, f"K_t{t}.npy"), dtype=np.float32, mode='r', shape=(n_coords, n_coords, n_chi, n_theta, n_phi))

    # ایجاد فایل خروجی Q با float64
    Q = np.memmap(os.path.join(output_dir, f"Q_t{t}.npy"), dtype=np.float64, mode='w+',
                  shape=(n_coords, n_coords, n_chi, n_theta, n_phi))

    # محاسبه بلوکی
    for i_start in range(0, n_chi, block_size):
        i_end = min(i_start + block_size, n_chi)

        # استخراج بلوک‌ها
        g_blk = g[:, :, i_start:i_end].astype(np.float64)
        Ricci_blk = Ricci[:, :, i_start:i_end].astype(np.float64)
        Rscalar_blk = Rscalar[i_start:i_end].astype(np.float64)
        K_blk = K[:, :, i_start:i_end].astype(np.float64)

        # معکوس متریک
        ginv_blk = np.zeros_like(g_blk, dtype=np.float64)
        for i in range(i_end - i_start):
            for j in range(n_theta):
                for k in range(n_phi):
                    try:
                        ginv_blk[:, :, i, j, k] = np.linalg.inv(g_blk[:, :, i, j, k])
                    except np.linalg.LinAlgError:
                        ginv_blk[:, :, i, j, k] = 0.0

        # تریس‌ها
        K_trace = np.einsum("abijk,abijk->ijk", ginv_blk, K_blk)
        KK_full = np.einsum("abijk,abijk->ijk", K_blk, K_blk)

        Q_blk = np.zeros_like(K_blk)
        for mu in range(n_coords):
            for nu in range(n_coords):
                KK_mu_nu = np.einsum("aijk,aijk->ijk", K_blk[mu], K_blk[:, nu])
                term0 = Ricci_blk[mu, nu]
                term1 = -0.5 * g_blk[mu, nu] * Rscalar_blk
                term2 = KK_mu_nu
                term3 = -K_trace * K_blk[mu, nu]
                term4 = -0.5 * g_blk[mu, nu] * (KK_full - K_trace**2)

                Q_blk[mu, nu] = term0 + term1 + term2 + term3 + term4

        Q[:, :, i_start:i_end] = Q_blk

    Q.flush()
    print(f"✅ Q_t{t}.npy saved in {time.time() - t0:.2f} s.")