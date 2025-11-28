import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات
timesteps = list(range(33, 43))  # t=33 to t=42
chi_blocks = list(range(0, 400, 10))  # chi=0 to 390, step=10
n_theta = 400
n_phi = 400
ceff_dir = "ceff_output"  # مسیر پوشه فایل‌های ceff_t[t]_chi[xx].npy
output_dir = "ceff_anisotropy_output"
os.makedirs(output_dir, exist_ok=True)

summary_lines = []

for t in timesteps:
    variances = []
    means = []
    for chi in chi_blocks:
        # بارگذاری فایل ceff برای این t و chi
        file_path = os.path.join(ceff_dir, f"ceff_t{t}_chi{chi}.npy")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        ceff = np.load(file_path)

        # بررسی داده‌ها
        mean_val = np.mean(ceff)
        var_val = np.var(ceff)
        variances.append(var_val)
        means.append(mean_val)

    # ذخیره خروجی متنی
    summary_lines.append(f"\nTime step = {t}")
    for i, chi in enumerate(chi_blocks[:len(means)]):
        summary_lines.append(f"chi={chi:03d}-{chi+10:03d} -> mean={means[i]:.4e}  var={variances[i]:.4e}")

    # رسم نمودار تغییرات میانگین و واریانس
    plt.figure(figsize=(10, 5))
    plt.plot(chi_blocks[:len(means)], means, marker='o', label='Mean ceff')
    plt.plot(chi_blocks[:len(variances)], variances, marker='x', label='Variance of ceff')
    plt.xlabel('Chi Block Start')
    plt.ylabel('Value')
    plt.title(f"c_eff angular block analysis – t={t}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ceff_anisotropy_t{t}.png"))
    plt.close()

# ذخیره فایل متنی خلاصه بدون کاراکترهای Unicode
with open(os.path.join(output_dir, "ceff_anisotropy_summary.txt"), "w", encoding='ascii', errors='ignore') as f:
    f.write("\n".join(summary_lines))