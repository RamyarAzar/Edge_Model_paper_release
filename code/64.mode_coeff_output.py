import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات مسیرها
w_dir = "w_output"
modes_dir = "mode_decomposition_output"
output_dir = "mode_coeff_output"
os.makedirs(output_dir, exist_ok=True)

# تنظیمات شبکه
n_chi, n_theta, n_phi = 400, 400, 400
dv = 1.0  # حجم سلولی (در صورت نیاز می‌توان مقدار دقیق‌تر وارد کرد)

# تنظیمات زمان
critical_timesteps = list(range(33, 42))
n_modes = 10  # تعداد مدها (مطابق تحلیل مرحله قبل)

# خروجی‌ها
c_kt_all = []  # شکل: (n_t, n_modes)

for t in critical_timesteps:
    print(f"⏳ Processing mode coefficients at t={t}...")

    # بارگذاری w(x,t) با memmap
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

    # بارگذاری مدها ψ_k(x,t)
    modes_path = os.path.join(modes_dir, f"w_modes_t{t}.npy")
    modes = np.load(modes_path)  # shape = (n_modes, n_chi, n_theta, n_phi)

    # محاسبه ضرایب c_k(t)
    c_kt = []
    for k in range(n_modes):
        mode_k = modes[k]
        integrand = w * mode_k  # نقطه به نقطه
        ck = np.sum(integrand) * dv  # تقریب انتگرال
        c_kt.append(ck)

    c_kt_all.append(c_kt)

# تبدیل به آرایه numpy و ذخیره
c_kt_all = np.array(c_kt_all)  # shape: (n_t, n_modes)
np.save(os.path.join(output_dir, "c_kt.npy"), c_kt_all)

# ذخیره خلاصه متنی
with open(os.path.join(output_dir, "mode_coefficients_summary.txt"), "w", encoding="utf-8") as f:
    f.write("📌 Quantum Mode Coefficients c_k(t)\n\n")
    for i, t in enumerate(critical_timesteps):
        f.write(f"t={t}:\n")
        for k, ck in enumerate(c_kt_all[i]):
            f.write(f"  Mode {k+1} = {ck:.6e}   |c_k|² = {np.abs(ck)**2:.6e}\n")
        f.write("\n")

# رسم نمودار انرژی مدها |c_k(t)|²
plt.figure(figsize=(8, 6))
for k in range(n_modes):
    energy_k = np.abs(c_kt_all[:, k])**2
    plt.plot(critical_timesteps, energy_k, marker='o', label=f"Mode {k+1}")

plt.xlabel("Time t")
plt.ylabel("|c_k(t)|²")
plt.title("Quantum Mode Energies Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mode_coefficients_energy_plot.png"))
plt.close()