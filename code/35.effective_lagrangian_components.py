import numpy as np
import os

# 📁 مسیرها
amp_dir = "phase_analysis_outputs"
phase_dir = "phase_analysis_outputs"
output_dir = "effective_lagrangian_components"
os.makedirs(output_dir, exist_ok=True)

# ⚙️ تنظیمات شبکه
Nx, Ny, Nz = 400, 400, 400
time_indices = range(33, 42)  # ⏱ فقط t = 33 تا 41
dx = dy = dz = 1.0  # اگر شبکه واقعی متفاوت بود اینها را تنظیم کن
dt = 1.0

# 🧰 مشتق‌گیر مرکزی مرتبه دوم
def grad(f, axis, d):
    return np.gradient(f, d, axis=axis, edge_order=2)

# 🚀 حلقه روی بازه زمانی
for t in time_indices:
    print(f"Processing t={t}...")

    amp_path = os.path.join(amp_dir, f"amp_t{t}.npy")
    phase_path = os.path.join(phase_dir, f"phase_t{t}.npy")

    amp = np.load(amp_path)
    phase = np.load(phase_path)

    # ░▒▓ محاسبه گرادیان‌ها ▓▒░
    grad_ax = grad(amp, axis=0, d=dx)
    grad_ay = grad(amp, axis=1, d=dy)
    grad_az = grad(amp, axis=2, d=dz)
    grad_amp_sq = grad_ax**2 + grad_ay**2 + grad_az**2

    grad_phix = grad(phase, axis=0, d=dx)
    grad_phiy = grad(phase, axis=1, d=dy)
    grad_phiz = grad(phase, axis=2, d=dz)
    grad_phase_sq = grad_phix**2 + grad_phiy**2 + grad_phiz**2

    phase_term = amp**2 * grad_phase_sq

    # 💾 ذخیره مؤلفه‌های لاگرانژی مؤثر
    np.save(os.path.join(output_dir, f"grad_amp_sq_t{t}.npy"), grad_amp_sq)
    np.save(os.path.join(output_dir, f"grad_phase_sq_t{t}.npy"), grad_phase_sq)
    np.save(os.path.join(output_dir, f"phase_term_t{t}.npy"), phase_term)

    print(f"✅ Done t={t}")

print("🎯 بازه t=33 تا t=41 با موفقیت پردازش و ذخیره شد.")