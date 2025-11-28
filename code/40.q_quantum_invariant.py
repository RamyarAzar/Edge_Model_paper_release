import os
import numpy as np
import matplotlib.pyplot as plt

# تنظیمات مسیرها و گام‌های زمانی
phase_dir = 'effective_field_output'
q_path = 'q_quantum_invariant.npy'
time_steps = list(range(33, 42))

# بارگذاری Q(t)
q_values = np.load(q_path)

# لیست خروجی‌ها
tau_values = []
h_eff_values = []
omega_values = []

for idx, t in enumerate(time_steps):
    print(f"در حال پردازش phase_kinetic_t{t}...")

    # بارگذاری داده انرژی جنبشی فاز
    pk_path = os.path.join(phase_dir, f"phase_kinetic_t{t}.npy")
    phase_kinetic = np.memmap(pk_path, dtype='float64', mode='r', shape=(400, 400, 400))

    # میانگین انرژی جنبشی فاز
    avg_kin = np.mean(phase_kinetic)

    # محاسبه فرکانس و زمان مشخصه
    omega = np.sqrt(avg_kin)
    tau = 1.0 / omega

    # محاسبه h_eff = Q × τ
    h_eff = q_values[idx] * tau

    # ذخیره در لیست‌ها
    tau_values.append(tau)
    h_eff_values.append(h_eff)
    omega_values.append(omega)

# تبدیل به آرایه numpy و ذخیره‌سازی
np.save("tau_values.npy", np.array(tau_values))
np.save("h_eff_values.npy", np.array(h_eff_values))
np.save("omega_values.npy", np.array(omega_values))

# چاپ مقادیر
for t, tau, omega, h_eff in zip(time_steps, tau_values, omega_values, h_eff_values):
    print(f"t={t} | τ = {tau:.3e} | ω = {omega:.3e} | h_eff = {h_eff:.3e}")

# رسم نمودار h_eff و ω_eff
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(time_steps, h_eff_values, marker='o')
plt.xlabel("Time Step")
plt.ylabel("h_eff = Q × τ")
plt.title("Effective Quantum Constant (h_eff)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time_steps, omega_values, marker='o', color='orange')
plt.xlabel("Time Step")
plt.ylabel("ω_eff = √⟨phase_kinetic⟩")
plt.title("Effective Frequency from Phase Dynamics")
plt.grid(True)

plt.tight_layout()
plt.show()