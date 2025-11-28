import numpy as np
import os
import matplotlib.pyplot as plt

# پارامترها
timesteps = list(range(33, 42))
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
chi_block_size = 10
κ = 8.9875517873681764e-18  # قابل تنظیم

# مسیر فایل‌ها
q_dir = "q_output"
k_dir = "k_output"
t_block_dir = "tmunu_final"
output_dir = "weak_curvature_analysis"
os.makedirs(output_dir, exist_ok=True)

K_threshold = 1e-12

for t in timesteps:
    print(f"🔍 Processing weak curvature analysis at t={t}...")

    # بارگذاری Q (کامل) و K (float32)
    Q = np.memmap(os.path.join(q_dir, f"Q_t{t}.npy"), dtype=np.float64, mode='r',
                  shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    K = np.memmap(os.path.join(k_dir, f"K_t{t}.npy"), dtype=np.float32, mode='r',
                  shape=(n_coords, n_coords, n_chi, n_theta, n_phi))

    # نُرم K در هر نقطه
    K_norm = np.sqrt(np.sum(K.astype(np.float64)**2, axis=(0, 1)))
    mask_lowK = (K_norm < K_threshold)

    if np.sum(mask_lowK) == 0:
        print(f"⚠️ No low-K regions detected at t={t}, skipping...")
        continue

    # بازسازی T از بلوک‌ها
    T = np.zeros((n_coords, n_coords, n_chi, n_theta, n_phi), dtype=np.float64)
    for chi_start in range(0, n_chi, chi_block_size):
        chi_end = min(chi_start + chi_block_size, n_chi)
        t_block_path = os.path.join(t_block_dir, f"T_block_t{t}_chi{chi_start}.npy")
        if not os.path.exists(t_block_path):
            print(f"⚠️ Missing block: {t_block_path}")
            continue
        T_block = np.load(t_block_path)
        T[:, :, chi_start:chi_end, :, :] = T_block

    # محاسبه طرف راست معادله اینشتین: Q + κT
    diff = Q + κ * T

    # اعمال ماسک به نقاط با K≈0
    diff_masked = diff[:, :, mask_lowK]
    diff_norm = np.sqrt(np.sum(diff_masked**2, axis=(0, 1)))

    mean_dev = np.mean(diff_norm)
    max_dev = np.max(diff_norm)

    # ذخیره آماری
    with open(os.path.join(output_dir, f"weak_limit_t{t}.txt"), "w") as f:
        f.write(f"t = {t}\n")
        f.write(f"Mean deviation (low K): {mean_dev:.5e}\n")
        f.write(f"Max deviation  (low K): {max_dev:.5e}\n")
        f.write(f"Number of low-K points: {np.sum(mask_lowK)}\n")

    # ذخیره .npy و نمودار
    np.save(os.path.join(output_dir, f"diff_masked_t{t}.npy"), diff_masked)

    plt.figure(figsize=(6, 4))
    plt.hist(diff_norm, bins=120, log=True, color='purple', alpha=0.7)
    plt.title(f"Deviation from GR (low-K) at t={t}")
    plt.xlabel("Deviation magnitude")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"weak_curvature_hist_t{t}.png"))
    plt.close()

print("✅ تحلیل Weak Curvature کامل شد. لطفاً آفلاین اجرا شود.")