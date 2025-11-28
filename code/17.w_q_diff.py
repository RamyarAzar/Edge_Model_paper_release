import numpy as np
import os
import time
import matplotlib.pyplot as plt

# تنظیمات
critical_timesteps = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
block_size = 10

# مسیر فایل‌ها
qnum_dir = "q_output"
qinverse_dir = "q_inv_output"
wnum_dir = "w_output"
winverse_dir = "w_inv_output"
g_dir = "metric_output"
diff_dir = "diff_output"
plot_dir = "diff_plots"
os.makedirs(diff_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

for t in critical_timesteps:
    print(f"\n⏳ Computing Q_diff and w_diff for t={t}...")
    t0 = time.time()

    Q_num = np.memmap(os.path.join(qnum_dir, f"Q_t{t}.npy"), dtype=np.float64, mode='r',
                      shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    Q_inv = np.memmap(os.path.join(qinverse_dir, f"Q_inv_t{t}.npy"), dtype=np.float64, mode='r',
                      shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    g = np.memmap(os.path.join(g_dir, f"g_t{t}.npy"), dtype=np.float32, mode='r',
                  shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    w_inv = np.memmap(os.path.join(winverse_dir, f"w_inv_t{t}.npy"), dtype=np.float64, mode='r',
                      shape=(n_chi, n_theta, n_phi))
    w_num = np.memmap(os.path.join(wnum_dir, f"w_t{t}.npy"), dtype=np.float64, mode='r',
                      shape=(n_chi, n_theta, n_phi))

    # ایجاد فایل خروجی
    Q_diff = np.memmap(os.path.join(diff_dir, f"Q_diff_t{t}.npy"), dtype=np.float64, mode='w+',
                       shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    w_diff = np.memmap(os.path.join(diff_dir, f"w_diff_t{t}.npy"), dtype=np.float64, mode='w+',
                       shape=(n_chi, n_theta, n_phi))

    for i_start in range(0, n_chi, block_size):
        i_end = min(i_start + block_size, n_chi)

        Qnum_blk = Q_num[:, :, i_start:i_end]
        Qinv_blk = Q_inv[:, :, i_start:i_end]
        g_blk = g[:, :, i_start:i_end].astype(np.float64)

        Qd_blk = Qinv_blk - Qnum_blk
        Q_diff[:, :, i_start:i_end] = Qd_blk

        ginv_blk = np.zeros_like(g_blk)
        for i in range(i_end - i_start):
            for j in range(n_theta):
                for k in range(n_phi):
                    try:
                        ginv_blk[:, :, i, j, k] = np.linalg.inv(g_blk[:, :, i, j, k])
                    except np.linalg.LinAlgError:
                        ginv_blk[:, :, i, j, k] = 0.0

        w_blk = np.einsum("abijk,abijk->ijk", ginv_blk, Qd_blk)
        w_blk = np.nan_to_num(w_blk, nan=0.0, posinf=0.0, neginf=0.0)
        w_diff[i_start:i_end] = w_blk

    Q_diff.flush()
    w_diff.flush()

    # ذخیره txt مقطع مرکزی
    centerline = w_diff[:, n_theta // 2, n_phi // 2]
    np.savetxt(os.path.join(diff_dir, f"w_diff_center_t{t}.txt"), centerline, fmt="%.8f")

    # ترسیم نمودار 1: مقطع مرکزی χ برای θ=π/2، φ=0
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(n_chi), centerline)
    plt.xlabel("χ index")
    plt.ylabel("w_diff(t={}, θ=π/2, φ=0)".format(t))
    plt.title(f"Profile of w_diff at θ=π/2, φ=0 for t={t}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"w_diff_centerline_t{t}.png"))
    plt.close()

    # ترسیم نمودار 2: میانگین w_diff روی کل فضا در هر χ
    mean_profile = np.mean(np.abs(w_diff), axis=(1, 2))  # shape: (χ,)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(n_chi), mean_profile)
    plt.xlabel("χ index")
    plt.ylabel("mean |w_diff(t={})|".format(t))
    plt.title(f"Mean |w_diff| over θ, φ for each χ — t={t}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"w_diff_mean_t{t}.png"))
    plt.close()

    print(f"✅ Done in {time.time() - t0:.2f} s — Saved Q_diff & w_diff & plots for t={t}")