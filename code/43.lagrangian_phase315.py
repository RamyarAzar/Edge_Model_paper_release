import os
import numpy as np

# تنظیم مسیرها
w_dir = 'w_output'
field_dir = 'effective_field_output'
out_dir = 'lagrangian_phase315'
os.makedirs(out_dir, exist_ok=True)

# گام‌های زمانی مورد نظر
time_steps = list(range(33, 42))

# بارگذاری λ(t)
lambda_values = np.load('lambda_normalization.npy')

# مشخصات شبکه
shape = (400, 400, 400)

for i, t in enumerate(time_steps):
    print(f"گام زمانی {t} در حال پردازش است...")

    # λ(t)
    lam = lambda_values[i]

    # بارگذاری w_t و ساخت w_norm
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=shape)
    w_norm = lam * w
    w_norm_path = os.path.join(out_dir, f"w_norm_t{t}.npy")
    np.save(w_norm_path, w_norm)

    # بارگذاری فاز و پتانسیل
    phase_path = os.path.join(field_dir, f"phase_kinetic_t{t}.npy")
    veff_path = os.path.join(field_dir, f"veff_t{t}.npy")

    phase_kinetic = np.memmap(phase_path, dtype='float64', mode='r', shape=shape)
    veff = np.memmap(veff_path, dtype='float64', mode='r', shape=shape)

    # محاسبه چگالی لاگرانژی
    lagrangian_density = phase_kinetic - veff
    lagrangian_path = os.path.join(out_dir, f"lagrangian_density_t{t}.npy")
    np.save(lagrangian_path, lagrangian_density)

    print(f"ذخیره w_norm_t{t} و lagrangian_density_t{t} کامل شد.")

print("✅ گام ۱ از فاز ۳.۱۵ تکمیل شد.")