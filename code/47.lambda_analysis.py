import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# مسیر فایل‌ها و پوشه خروجی
lambda_path = 'lambda_normalization.npy'
q_path = 'q_quantum_invariant.npy'
output_dir = 'lambda_analysis_phase315'
os.makedirs(output_dir, exist_ok=True)

# بارگذاری داده‌ها
lambda_arr = np.load(lambda_path)  # بردار λ(t) با طول 9
Q_arr = np.load(q_path)            # بردار Q(t) با طول 9
time_steps = list(range(33, 42))   # t = 33 تا 41

# محاسبه τ(t) و ℏ_eff(t)
tau_arr = 1.0 / lambda_arr
hbar_eff_arr = Q_arr / lambda_arr

# ذخیره خروجی‌ها
np.save(os.path.join(output_dir, 'tau_t.npy'), tau_arr)
np.save(os.path.join(output_dir, 'hbar_eff_t.npy'), hbar_eff_arr)

# ذخیره به صورت CSV
df = pd.DataFrame({
    'time_step': time_steps,
    'lambda': lambda_arr,
    'tau': tau_arr,
    'Q': Q_arr,
    'hbar_eff': hbar_eff_arr
})
df.to_csv(os.path.join(output_dir, 'lambda_tau_hbar_summary.csv'), index=False)

# رسم نمودار
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(time_steps, lambda_arr, 'bo-', label='λ(t)')
plt.title('λ(t) - Normalization Scale')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(time_steps, tau_arr, 'go-', label='τ(t) = 1/λ(t)')
plt.title('τ(t) - Oscillation Time Scale')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(time_steps, hbar_eff_arr, 'ro-', label='ℏ_eff(t)')
plt.title('ℏ_eff(t) = Q(t)/λ(t)')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lambda_tau_hbar_plots.png'))
plt.show()