import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات مسیر
r_dir = "r_output"  # مسیر فایل‌های R(t, χ, θ)
output_dir = "ceff_output"
os.makedirs(output_dir, exist_ok=True)

# پارامترهای مش
n_chi, n_theta = 400, 400
chi_block_size = 10
t_steps = list(range(33, 43))  # زمان‌های 33 تا 42

# λ(t) به‌صورت دیکشنری از داده‌های شما
lambda_dict = {
    33: 13208793.62, 34: 13239207.29, 35: 12455078.46,
    36: 21449203.81, 37: 25107541.81, 38: 25369562.2,
    39: 24098134.35, 40: 27036612.69, 41: 27040064.16, 42: 27040064.16
}

# تابع log-safe برای تقسیم
def log_safe_divide(a, b):
    return np.exp(np.log(np.abs(a) + 1e-300) - np.log(np.abs(b) + 1e-300))

# پردازش گام به گام زمانی
for t in t_steps:
    print(f"\n🌀 Processing c_eff at t = {t}...")
    lambda_t = lambda_dict[t]
    
    # آماده‌سازی خروجی میانگین
    ceff_mean = 0
    count_blocks = 0

    for chi_start in range(0, n_chi, chi_block_size):
        chi_end = min(chi_start + chi_block_size, n_chi)
        print(f"  ⏳ Chi block: {chi_start}-{chi_end}")

        # خواندن بلوکی R(t, χ, θ)
        R_block = np.memmap(os.path.join(r_dir, f"R_t{t}.npy"),
                            dtype=np.float32, mode='r',
                            shape=(n_chi, n_theta))[chi_start:chi_end, :]

        # محاسبه‌ی c_eff = R / lambda
        ceff_block = log_safe_divide(R_block, lambda_t)

        # ذخیره فایل npy
        np.save(os.path.join(output_dir, f"ceff_t{t}_chi{chi_start}.npy"), ceff_block)

        # ذخیره txt
        with open(os.path.join(output_dir, f"ceff_t{t}_chi{chi_start}.txt"), "w") as ftxt:
            for row in ceff_block:
                ftxt.write(" ".join([f"{val:.6e}" for val in row]) + "\n")

        # رسم هیستوگرام
        plt.hist(ceff_block.flatten(), bins=100, log=True)
        plt.title(f"Histogram of c_eff at t={t}, chi {chi_start}-{chi_end}")
        plt.xlabel("c_eff")
        plt.ylabel("Count (log)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"ceff_hist_t{t}_chi{chi_start}.png"))
        plt.clf()

        # آماره‌ی میانگین
        ceff_mean += np.mean(ceff_block)
        count_blocks += 1

    # چاپ میانگین برای کل زمان t
    print(f"✅ Mean c_eff at t={t} = {ceff_mean / count_blocks:.6e}")