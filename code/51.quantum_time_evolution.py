import numpy as np
import os
import matplotlib.pyplot as plt

# مسیر خروجی
output_dir = "quantum_time_evolution_scalar"
os.makedirs(output_dir, exist_ok=True)

# بارگذاری داده‌ها
psi_t = np.load("data/psi_t_complex.npy")          # (N,)
hbar_eff_t = np.load("data/hbar_eff_t.npy")        # (N,)
eig_H = np.load("data/eig_H.npy")                  # H_eff مقادیر انرژی (N,)
eig_T = np.load("data/eig_T.npy")                  # زمان‌های گسسته

# تنظیمات
delta_t = 1.0
N = len(psi_t)
psi_evolved = []

# تحول زمانی با U(t) اسکالر: ψ(t+1) = exp(-i * H(t) * dt / ħ_eff(t)) * ψ(t)
for i in range(N - 1):
    H = eig_H[i]
    ħ = hbar_eff_t[i]
    U = np.exp(-1j * H * delta_t / ħ)
    psi_next = U * psi_t[i]
    psi_evolved.append(psi_next)

# اضافه کردن ψ[0] برای تطابق زمانی
psi_evolved = np.array(psi_evolved)
psi_evolved = np.insert(psi_evolved, 0, psi_t[0])

# ذخیره خروجی
np.save(os.path.join(output_dir, "psi_t_evolved_scalar.npy"), psi_evolved)

# رسم فاز و دامنه
amplitudes = np.abs(psi_evolved)
phases = np.angle(psi_evolved)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(eig_T, amplitudes, 'b', marker='o')
plt.title("Amplitude |ψ(t)| — Scalar U(t)")
plt.xlabel("Time Step")
plt.ylabel("|ψ(t)|")

plt.subplot(1, 2, 2)
plt.plot(eig_T, phases, 'r', marker='o')
plt.title("Phase arg(ψ(t)) — Scalar U(t)")
plt.xlabel("Time Step")
plt.ylabel("arg(ψ(t))")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "psi_evolution_scalar.png"))