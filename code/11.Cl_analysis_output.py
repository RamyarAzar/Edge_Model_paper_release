import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# پارامترها
critical_timesteps = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]
spin_weights = [0, 1, -1, 2, -2]
n_coords = 4

# مسیر فایل‌ها
input_dir = "spectral_alm_output"
output_dir = "Cl_analysis_output"
os.makedirs(output_dir, exist_ok=True)

for t in critical_timesteps:
    print(f"\n📊 تحلیل طیفی برای t={t}...")

    for s in spin_weights:
        print(f"  ↪ اسپین وزنی s={s}")
        Cl_dict = defaultdict(list)

        for mu in range(n_coords):
            for nu in range(n_coords):
                filename = f"alm_q_t{t}_s{s}_mu{mu}_nu{nu}.npz"
                path = os.path.join(input_dir, filename)
                if not os.path.exists(path):
                    print(f"    ⚠️ فایل پیدا نشد: {filename}")
                    continue

                data = np.load(path)
                alm = data["alm"]   # [chi, mode]
                l_vals = data["l"]  # [mode]

                for chi in range(alm.shape[0]):
                    for l in np.unique(l_vals):
                        idx = np.where(l_vals == l)[0]
                        alm_lm = alm[chi, idx]
                        Cl = np.mean(np.abs(alm_lm)**2) / (2 * l + 1)
                        Cl_dict[l].append(Cl)

        # میانگین‌گیری بر روی χ، μ, ν
        ls = sorted(Cl_dict.keys())
        Cl_avg = [np.mean(Cl_dict[l]) for l in ls]

        # رسم نمودار
        plt.figure(figsize=(7, 5))
        plt.plot(ls, Cl_avg, marker='o', linewidth=2)
        plt.title(f"Spectral Power $C_\\ell$ for Q — t={t}, s={s}")
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$C_\ell$")
        plt.grid(True)
        plt.tight_layout()

        # ذخیره تصویر
        plot_path = os.path.join(output_dir, f"Cl_q_t{t}_s{s}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"    ✅ نمودار ذخیره شد: {plot_path}")
