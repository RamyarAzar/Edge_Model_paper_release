import numpy as np
import os

# مسیر پوشه‌های ورودی و خروجی
comp_dir = "effective_lagrangian_components"
veff_dir = "effective_field_output"
output_dir = "lagrangian_final_outputs"
os.makedirs(output_dir, exist_ok=True)

# بازه زمانی مورد نظر
time_indices = range(33, 42)

# محاسبه و ذخیره L_eff برای هر زمان
for t in time_indices:
    print(f"Processing L_eff at t={t}...")

    # بارگذاری مؤلفه‌ها
    grad_amp_sq = np.load(os.path.join(comp_dir, f"grad_amp_sq_t{t}.npy"))
    grad_phase_sq = np.load(os.path.join(comp_dir, f"grad_phase_sq_t{t}.npy"))
    phase_term = np.load(os.path.join(comp_dir, f"phase_term_t{t}.npy"))
    veff = np.load(os.path.join(veff_dir, f"veff_t{t}.npy"))

    # ساخت چگالی لاگرانژی مؤثر
    L_eff = 0.5 * grad_amp_sq + 0.5 * phase_term - veff

    # ذخیره فایل numpy
    np.save(os.path.join(output_dir, f"L_eff_t{t}.npy"), L_eff)

    # ذخیره آمار ساده برای بررسی سریع
    L_mean = np.mean(L_eff)
    L_std = np.std(L_eff)
    with open(os.path.join(output_dir, f"L_eff_t{t}_stats.txt"), 'w') as f:
        f.write(f"Mean: {L_mean}\n")
        f.write(f"Std: {L_std}\n")

    print(f"✅ Done t={t}")

print("🎯 L_eff محاسبه و ذخیره شد برای تمام زمان‌های t = 33 تا 41.")