import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات مسیرها
w_dir = "w_output"
phase_dir = "effective_field_output"
veff_dir = "veff_output"
output_dir = "gauge_outputs_full"
os.makedirs(output_dir, exist_ok=True)

# تنظیمات عددی
critical_timesteps = [33, 34, 35, 36, 37, 38, 39, 40, 41]
n_chi, n_theta, n_phi = 400, 400, 400
dx3 = 1.0  # در صورت نیاز به تغییر، واحد حجم

# لیست‌های خروجی
s_eff_list = []
hbar_eff_list = []
a0_list = []

for t in critical_timesteps:
    print(f"⏳ Processing t={t}...")

    # بارگذاری میدان‌ها
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

    pk_path = os.path.join(phase_dir, f"phase_kinetic_t{t}.npy")
    phase_kinetic = np.load(pk_path)

    veff_path = os.path.join(veff_dir, f"veff_t{t}.npy")
    veff = np.load(veff_path)

    # محاسبه S_eff(t) = ∫ w²(x) * phase_kinetic(x) dV
    integrand_s = w**2 * phase_kinetic
    S_eff_t = np.sum(integrand_s) * dx3

    # محاسبه ħ_eff(t) = ∫ w²(x) * veff(x) dV
    integrand_h = w**2 * veff
    hbar_eff_t = np.sum(integrand_h) * dx3 + 1e-12  # جلوگیری از صفر شدن مخرج

    # محاسبه A₀(t)
    A0_t = S_eff_t / hbar_eff_t

    # ذخیره‌سازی
    s_eff_list.append(S_eff_t)
    hbar_eff_list.append(hbar_eff_t)
    a0_list.append(A0_t)

# ذخیره فایل‌های عددی
np.save(os.path.join(output_dir, "s_eff_t.npy"), np.array(s_eff_list))
np.save(os.path.join(output_dir, "hbar_eff_t.npy"), np.array(hbar_eff_list))
np.save(os.path.join(output_dir, "A_mu_t.npy"), np.array(a0_list))

# ذخیره گزارش متنی
with open(os.path.join(output_dir, "gauge_field_summary.txt"), "w", encoding="utf-8") as f:
    f.write("🔬 Gauge Field A₀(t) Summary:\n\n")
    for i, t in enumerate(critical_timesteps):
        f.write(f"t = {t} | S_eff = {s_eff_list[i]:.6e} | ħ_eff = {hbar_eff_list[i]:.6e} | A₀ = {a0_list[i]:.6e}\n")

# رسم نمودار
plt.figure(figsize=(8, 5))
plt.plot(critical_timesteps, a0_list, marker='o')
plt.title("Gauge Field A₀(t) from Geometric Effective Action")
plt.xlabel("Critical Time t")
plt.ylabel("A₀(t)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gauge_field_A0_plot.png"))
plt.close()