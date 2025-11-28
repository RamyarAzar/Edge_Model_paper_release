import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# مسیر فایل‌های اصلی
lagr_dir = 'lagrangian_phase315'
ham_dir = 'hamiltonian_phase315_filtered'

# بارگذاری مقادیر خام
L_raw = np.load(os.path.join(lagr_dir, 'lagrangian_total.npy'))
H_raw = np.load(os.path.join(ham_dir, 'hamiltonian_total.npy'))

# محاسبه log نسبی نسبت به بیشینه
epsilon = 1e-12
L_max = np.max(np.abs(L_raw))
H_max = np.max(np.abs(H_raw))

L_lognorm = np.log10(np.abs(L_raw) / L_max + epsilon)
H_lognorm = np.log10(np.abs(H_raw) / H_max + epsilon)

# ذخیره خروجی‌ها
np.save(os.path.join(lagr_dir, 'lagrangian_lognorm.npy'), L_lognorm)
np.save(os.path.join(ham_dir, 'hamiltonian_lognorm.npy'), H_lognorm)

# ذخیره CSV برای بررسی سریع
time_steps = list(range(33, 42))
df = pd.DataFrame({
    'time_step': time_steps,
    'L_raw': L_raw,
    'H_raw': H_raw,
    'L_log_norm': L_lognorm,
    'H_log_norm': H_lognorm
})
df.to_csv('energy_log_normalized_summary.csv', index=False)

# رسم نمودار مقیاس‌گذاری نسبی
plt.figure(figsize=(10, 5))
plt.plot(time_steps, L_lognorm, marker='o', label='log10(|L(t)| / max)', color='blue')
plt.plot(time_steps, H_lognorm, marker='s', label='log10(|H(t)| / max)', color='orange', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Log-Scaled (Normalized) Energy')
plt.title('Relative Log-Scaled Energy in DMH Phase 3.15')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()