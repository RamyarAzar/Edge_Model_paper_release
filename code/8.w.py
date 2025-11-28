import numpy as np
import os
import time

# تنظیمات
critical_timesteps = [1, 2, 3, 9, 10, 11, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
block_size = 10

# مسیر فایل‌ها
g_dir = "metric_output"
Q_dir = "q_output"
output_dir = "w_output"
os.makedirs(output_dir, exist_ok=True)

for t in critical_timesteps:
    print(f"\n⏳ Computing w at t={t}...")
    t0 = time.time()

    g = np.memmap(os.path.join(g_dir, f"g_t{t}.npy"), dtype=np.float32, mode='r',
                  shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    Q = np.memmap(os.path.join(Q_dir, f"Q_t{t}.npy"), dtype=np.float64, mode='r',
                  shape=(n_coords, n_coords, n_chi, n_theta, n_phi))

    # ایجاد فایل خروجی برای w
    w = np.memmap(os.path.join(output_dir, f"w_t{t}.npy"), dtype=np.float64, mode='w+',
                  shape=(n_chi, n_theta, n_phi))

    for i_start in range(0, n_chi, block_size):
        i_end = min(i_start + block_size, n_chi)

        g_blk = g[:, :, i_start:i_end].astype(np.float64)
        Q_blk = Q[:, :, i_start:i_end]

        ginv_blk = np.zeros_like(g_blk)
        for i in range(i_end - i_start):
            for j in range(n_theta):
                for k in range(n_phi):
                    try:
                        ginv_blk[:, :, i, j, k] = np.linalg.inv(g_blk[:, :, i, j, k])
                    except np.linalg.LinAlgError:
                        ginv_blk[:, :, i, j, k] = 0.0

        # محاسبه w = g^{μν} Q_{μν}
        w_block = np.einsum("abijk,abijk->ijk", ginv_blk, Q_blk)
        w[i_start:i_end] = w_block

    w.flush()
    # ذخیره نسخه txt برای بررسی
    np.savetxt(os.path.join(output_dir, f"w_t{t}.txt"), w[:, n_theta//2, n_phi//2], fmt="%.8f")

    print(f"✅ w_t{t} saved in {time.time() - t0:.2f} s.")