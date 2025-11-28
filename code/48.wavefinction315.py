import numpy as np
import os
import matplotlib.pyplot as plt

# مسیر ورودی و خروجی
lagrangian_path = 'lagrangian_phase315/lagrangian_total.npy'  # L(t) فیزیکی
hbar_path = 'lambda_analysis_phase315/hbar_eff_t.npy'
output_dir = 'wavefunction_phase315'
os.makedirs(output_dir, exist_ok=True)

# بارگذاری داده‌ها
L_arr = np.load(lagrangian_path)        # shape (9,) → L(t) فیزیکی
hbar_eff_arr = np.load(hbar_path)       # shape (9,)
time_steps = list(range(33, 42))
Δt = 1.0

# محاسبه اکشن تجمعی مؤثر
S_eff = np.cumsum(L_arr * Δt)

# تابع موج: ψ(t) = exp(-i * S(t)/ℏ_eff(t))
psi_t = np.exp(-1j * S_eff / hbar_eff_arr)

# استخراج دامنه و فاز
amplitude = np.abs(psi_t)
phase = np.angle(psi_t)

# ذخیره خروجی‌ها
np.save(os.path.join(output_dir, 'S_eff_t.npy'), S_eff)
np.save(os.path.join(output_dir, 'psi_t_complex.npy'), psi_t)
np.save(os.path.join(output_dir, 'psi_amplitude.npy'), amplitude)
np.save(os.path.join(output_dir, 'psi_phase.npy'), phase)

# رسم نمودار دامنه و فاز تابع موج
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(time_steps, amplitude, 'b-o')
plt.title('Amplitude |ψ(t)|')
plt.xlabel('Time Step')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time_steps, phase, 'r-o')
plt.title('Phase arg(ψ(t))')
plt.xlabel('Time Step')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'psi_wavefunction_plots.png'))
plt.show()