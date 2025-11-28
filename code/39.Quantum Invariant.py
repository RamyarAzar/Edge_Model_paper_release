import os
import numpy as np
import matplotlib.pyplot as plt

# تنظیمات پایه شبکه
w_dir = 'w_output'
n_chi, n_theta, n_phi = 400, 400, 400

# در صورت استفاده از متریک خمیده:
# اگر |g| (دترمینان متریک) موجود باشد:
#     |g| باید به صورت آرایه 3D با ابعاد (n_chi, n_theta, n_phi) خوانده شود
#     و عنصر حجم = np.sqrt(np.abs(g)) * dχ * dθ * dφ
# در غیر این صورت فرض تخت:
delta_chi = delta_theta = delta_phi = 1.0  # حجم واحد در فضای تخت

# گام‌های زمانی مورد نظر
time_steps = list(range(33, 42))
q_values = []

for t in time_steps:
    print(f"در حال پردازش w_t{t}...")

    # بارگذاری با memmap
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

    # محاسبه Q(t) = ∫ |w|² dV بدون نرمال‌سازی
    volume_element = delta_chi * delta_theta * delta_phi  # حجم شبکه تخت
    q_t = np.sum(w ** 2) * volume_element
    q_values.append(q_t)

# ذخیره مقادیر Q(t) برای استفاده در مراحل بعدی
np.save("q_quantum_invariant.npy", np.array(q_values))

# نمایش عددی
for t, q in zip(time_steps, q_values):
    print(f"Q(t={t}) = {q:.6e}")

# رسم نمودار لگاریتمی نیمه‌لگ
plt.figure(figsize=(8, 5))
plt.semilogy(time_steps, q_values, marker='o')
plt.xlabel('Time Step')
plt.ylabel('Q(t)')
plt.title('Quantum Invariant Q(t) in Semilog Scale')
plt.grid(True)
plt.tight_layout()
plt.show()