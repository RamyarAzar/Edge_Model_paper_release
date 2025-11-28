import numpy as np
import os

# 📁 مسیرهای ورودی/خروجی
w_dir = "w_output"
modes_dir = "mode_decomposition_output"
output_dir = "hybrid_structure_output"
os.makedirs(output_dir, exist_ok=True)

# ⚙️ تنظیمات پایه
critical_timesteps = list(range(33, 42))
n_chi, n_theta, n_phi = 400, 400, 400
block_size = 50

# جفت‌های مد قوی‌کوپل‌شده (index از 1 شروع می‌شود)
coupled_pairs = [(1, 6), (1, 10), (4, 10), (6, 10), (1, 7)]

# 🎯 حلقه‌ی اصلی زمان
for t in critical_timesteps:
    print(f"\n⏳ Processing hybrid structures at t={t}...")

    # بارگذاری w_t
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

    # بارگذاری w_modes_t
    modes_path = os.path.join(modes_dir, f"w_modes_t{t}.npy")
    modes = np.load(modes_path)  # shape = (n_modes, n_chi, n_theta, n_phi)

    # 🔁 برای هر جفت مد ترکیبی
    for (i, j) in coupled_pairs:
        # آماده‌سازی خروجی‌ها
        psi_plus_sq = np.zeros((n_chi, n_theta, n_phi), dtype=np.float64)
        psi_minus_sq = np.zeros((n_chi, n_theta, n_phi), dtype=np.float64)

        psi_i = modes[i - 1]
        psi_j = modes[j - 1]

        # 🧠 پردازش بلوک به بلوک
        for chi_start in range(0, n_chi, block_size):
            chi_end = min(chi_start + block_size, n_chi)

            w_blk = w[chi_start:chi_end, :, :]
            psi_i_blk = psi_i[chi_start:chi_end, :, :]
            psi_j_blk = psi_j[chi_start:chi_end, :, :]

            # ترکیب خطی مدها (بدون نرمال‌سازی نهایی)
            psi_plus_blk = (psi_i_blk + psi_j_blk) / np.sqrt(2)
            psi_minus_blk = (psi_i_blk - psi_j_blk) / np.sqrt(2)

            w_sq_blk = w_blk**2 + 1e-100  # برای جلوگیری از تقسیم بر صفر

            # محاسبه انرژی نرمال‌شده بر حسب w
            psi_plus_sq[chi_start:chi_end] = (np.abs(psi_plus_blk)**2) / w_sq_blk
            psi_minus_sq[chi_start:chi_end] = (np.abs(psi_minus_blk)**2) / w_sq_blk

        # ذخیره‌ی فایل خروجی
        np.save(os.path.join(output_dir, f"psi_plus_sq_t{t}_modes{i}_{j}.npy"), psi_plus_sq)
        np.save(os.path.join(output_dir, f"psi_minus_sq_t{t}_modes{i}_{j}.npy"), psi_minus_sq)

        print(f"✅ Done: ψ₊, ψ₋ for modes ({i}, {j}) at t={t}")