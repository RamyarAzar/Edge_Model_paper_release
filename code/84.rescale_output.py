import numpy as np
import os
import matplotlib.pyplot as plt

# مسیرهای خروجی
output_dir = "rescale_output"
os.makedirs(output_dir, exist_ok=True)

# ⚙️ مقادیر تجربی مرجع (در SI units)
c_phys = 2.99792458e8        # m/s
hbar_phys = 1.054571817e-34  # J.s

# داده‌های مدل برای t = 33 تا 42
time_steps = list(range(33, 43))
lambda_values = [13208793.62, 13239207.29, 12455078.46, 21449203.81, 25107541.81,
                 25369562.2, 24098134.35, 27036612.69, 27040064.16, 27040064.16]
tau_values = [7.57e-8, 7.55e-8, 8.03e-8, 4.66e-8, 3.98e-8, 3.94e-8, 4.15e-8, 3.70e-8, 3.70e-8, 3.70e-8]
Q_values = [1.38e75, 1.37e75, 1.55e75, 5.23e74, 3.82e74, 3.74e74, 4.15e74, 3.29e74, 3.29e74, 3.29e74]
hbar_eff_values = [1.04e68, 1.04e68, 1.25e68, 2.44e67, 1.52e67, 1.47e67, 1.72e67, 1.22e67, 1.22e67, 1.22e67]
ceff_values = [4.534754e19, 4.524336e19, 4.809173e19, 2.792580e19, 2.385683e19,
               2.361043e19, 2.485613e19, 2.215463e19, 2.215181e19, 2.215181e19]

# 📦 آرایه‌های خروجی
gamma_c = []
gamma_hbar = []
gamma_L = []
gamma_T = []
gamma_E = []

# 🔁 محاسبه ضرایب برای هر t
for i, t in enumerate(time_steps):
    ceff = ceff_values[i]
    hbar_eff = hbar_eff_values[i]

    # ضریب تطبیق سرعت نور
    gamma_c_t = c_phys / ceff
    gamma_c.append(gamma_c_t)

    # ضریب تطبیق پلانک
    gamma_hbar_t = hbar_phys / hbar_eff
    gamma_hbar.append(gamma_hbar_t)

    # استخراج ضرایب مقیاس طول، زمان، انرژی
    gamma_T_t = np.sqrt(gamma_hbar_t / gamma_c_t**2)
    gamma_L_t = gamma_c_t * gamma_T_t
    gamma_E_t = gamma_hbar_t / gamma_T_t

    gamma_T.append(gamma_T_t)
    gamma_L.append(gamma_L_t)
    gamma_E.append(gamma_E_t)

# ذخیره در npy
np.save(os.path.join(output_dir, "gamma_c.npy"), gamma_c)
np.save(os.path.join(output_dir, "gamma_hbar.npy"), gamma_hbar)
np.save(os.path.join(output_dir, "gamma_T.npy"), gamma_T)
np.save(os.path.join(output_dir, "gamma_L.npy"), gamma_L)
np.save(os.path.join(output_dir, "gamma_E.npy"), gamma_E)

# ذخیره در فایل متنی خلاصه برای ارسال
with open(os.path.join(output_dir, "rescale_summary.txt"), "w") as f:
    f.write("t\tgamma_c\tgamma_hbar\tgamma_T\tgamma_L\tgamma_E\n")
    for i, t in enumerate(time_steps):
        f.write(f"{t}\t{gamma_c[i]:.3e}\t{gamma_hbar[i]:.3e}\t{gamma_T[i]:.3e}\t{gamma_L[i]:.3e}\t{gamma_E[i]:.3e}\n")

# 📈 نمودارهای نمایشی ضرایب تطبیق
def plot_gamma(values, label):
    plt.figure()
    plt.plot(time_steps, values, marker='o')
    plt.xlabel("Time step")
    plt.ylabel(label)
    plt.title(f"{label} across time")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{label}.png"))
    plt.close()

plot_gamma(gamma_c, "gamma_c")
plot_gamma(gamma_hbar, "gamma_hbar")
plot_gamma(gamma_T, "gamma_T")
plot_gamma(gamma_L, "gamma_L")
plot_gamma(gamma_E, "gamma_E")