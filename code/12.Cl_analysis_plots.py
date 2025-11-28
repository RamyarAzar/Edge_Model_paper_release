import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات
timesteps = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]
spins = [0, 1, -1, 2, -2]
n_coords = 4
input_dir = "spectral_alm_output"
output_dir = "Cl_analysis_plots"
os.makedirs(output_dir, exist_ok=True)

# شروع تحلیل
for s in spins:
    plt.figure(figsize=(10, 6))
    for t in timesteps:
        # جمع توان طیفی برای تمام ترکیب‌های μν
        Cl_sum = {}
        Cl_count = {}

        for mu in range(n_coords):
            for nu in range(n_coords):
                fname = f"alm_q_t{t}_s{s}_mu{mu}_nu{nu}.npz"
                path = os.path.join(input_dir, fname)
                if not os.path.exists(path):
                    continue
                data = np.load(path)
                alm = data["alm"]  # shape: [chi, modes]
                l = data["l"]
                m = data["m"]

                for idx, l_val in enumerate(l):
                    abs2 = np.abs(alm[:, idx])**2  # sum over chi
                    if l_val not in Cl_sum:
                        Cl_sum[l_val] = 0.0
                        Cl_count[l_val] = 0
                    Cl_sum[l_val] += np.mean(abs2)
                    Cl_count[l_val] += 1

        if len(Cl_sum) == 0:
            continue

        ells = sorted(Cl_sum.keys())
        Cl_vals = [Cl_sum[l] / Cl_count[l] for l in ells]
        plt.plot(ells, Cl_vals, label=f"t={t}")

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell$")
    plt.title(f"Specral evolution for spin {s}")
    plt.yscale("log")
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"Cl_vs_t_spin{s}.png"))
    plt.close()

# همچنین مقایسه بین اسپین‌ها برای زمان‌های خاص:
for t in [2, 45]:
    plt.figure(figsize=(10, 6))
    for s in spins:
        Cl_sum = {}
        Cl_count = {}

        for mu in range(n_coords):
            for nu in range(n_coords):
                fname = f"alm_q_t{t}_s{s}_mu{mu}_nu{nu}.npz"
                path = os.path.join(input_dir, fname)
                if not os.path.exists(path):
                    continue
                data = np.load(path)
                alm = data["alm"]
                l = data["l"]
                m = data["m"]

                for idx, l_val in enumerate(l):
                    abs2 = np.abs(alm[:, idx])**2
                    if l_val not in Cl_sum:
                        Cl_sum[l_val] = 0.0
                        Cl_count[l_val] = 0
                    Cl_sum[l_val] += np.mean(abs2)
                    Cl_count[l_val] += 1

        if len(Cl_sum) == 0:
            continue

        ells = sorted(Cl_sum.keys())
        Cl_vals = [Cl_sum[l] / Cl_count[l] for l in ells]
        plt.plot(ells, Cl_vals, label=f"s={s}")

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell$")
    plt.title(f"Spin comparison at t={t}")
    plt.yscale("log")
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"Cl_vs_s_t{t}.png"))
    plt.close()