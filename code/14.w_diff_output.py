import numpy as np
import os
import time

# تنظیمات
critical_timesteps = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
block_size = 10

# مسیر فایل‌ها
g_dir = "metric_output"
q_num_dir = "q_output"
q_rec_dir = "q_reconstructed"
output_dir = "w_diff_output"
os.makedirs(output_dir, exist_ok=True)

for t in critical_timesteps:
    print(f"\n🔎 Computing w_diff at t={t}...")
    t0 = time.time()

    g = np.memmap(os.path.join(g_dir, f"g_t{t}.npy"), dtype=np.float32, mode='r',
                  shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    Q_num = np.memmap(os.path.join(q_num_dir, f"Q_t{t}.npy"), dtype=np.float64, mode='r',
                      shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    Q_rec = np.memmap(os.path.join(q_rec_dir, f"Q_recon_t{t}.npy"), dtype=np.complex128, mode='r',
                      shape=(n_coords, n_coords, n_chi, n_theta, n_phi))

    w_diff = np.memmap(os.path.join(output_dir, f"w_diff_t{t}.npy"), dtype=np.float64, mode='w+',
                       shape=(n_chi, n_theta, n_phi))

    for i_start in range(0, n_chi, block_size):
        i_end = min(i_start + block_size, n_chi)

        g_blk = g[:, :, i_start:i_end].astype(np.float64)
        Qnum_blk = Q_num[:, :, i_start:i_end]
        Qrec_blk = Q_rec[:, :, i_start:i_end].real  # فقط بخش حقیقی برای بازسازی s=0

        # معکوس‌گیری از g
        ginv_blk = np.zeros_like(g_blk)
        for i in range(i_end - i_start):
            for j in range(n_theta):
                for k in range(n_phi):
                    try:
                        ginv_blk[:, :, i, j, k] = np.linalg.inv(g_blk[:, :, i, j, k])
                    except np.linalg.LinAlgError:
                        ginv_blk[:, :, i, j, k] = 0.0

        # محاسبه ∆Q = Q_rec - Q_num
        deltaQ_blk = Qrec_blk - Qnum_blk

        # محاسبه w_diff = g^{μν} ∆Q_{μν}
        wdiff_blk = np.einsum("abijk,abijk->ijk", ginv_blk, deltaQ_blk)

        # 🚨 پایداری عددی (جلوگیری از NaN و Inf)
        wdiff_blk = np.nan_to_num(wdiff_blk, nan=0.0, posinf=0.0, neginf=0.0)

        w_diff[i_start:i_end] = wdiff_blk

    w_diff.flush()

    # ذخیره برش مرکزی (θ=π/2, φ=π) برای بررسی بصری
    center_line = w_diff[:, n_theta//2, n_phi//2]
    np.savetxt(os.path.join(output_dir, f"w_diff_t{t}.txt"), center_line, fmt="%.8e")

    print(f"✅ Done t={t} in {time.time() - t0:.2f} s — saved .npy + .txt")