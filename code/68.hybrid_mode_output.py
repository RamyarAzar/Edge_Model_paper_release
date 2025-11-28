import numpy as np
import os
import matplotlib.pyplot as plt

# مسیرها
w_dir = "w_output"
output_dir = "hybrid_mode_output_corrected"
os.makedirs(output_dir, exist_ok=True)

# تنظیمات
critical_timesteps = list(range(33, 42))
n_chi, n_theta, n_phi = 400, 400, 400
n_modes = 10

# جفت‌های کوپل‌شده قوی
strongly_coupled_pairs = [(1, 6), (1, 10), (4, 10), (6, 10), (1, 7)]

# آماده‌سازی دیکشنری خروجی
hybrid_energy = {f"{i}_{j}": [] for i, j in strongly_coupled_pairs}

# حلقه روی زمان‌ها
for t in critical_timesteps:
    print(f"⏳ Processing hybrid modes at t={t}")
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

    χ = np.linspace(0, np.pi, n_chi)

    for (i, j) in strongly_coupled_pairs:
        k_i = i
        k_j = j

        basis_i = np.sin(k_i * χ)[:, None, None]  # حالت پایه i
        basis_j = np.sin(k_j * χ)[:, None, None]  # حالت پایه j

        ψ_i = basis_i * w
        ψ_j = basis_j * w

        ψ_plus = (ψ_i + ψ_j) / np.sqrt(2)
        ψ_minus = (ψ_i - ψ_j) / np.sqrt(2)

        E_plus = np.sum(np.abs(ψ_plus)**2)
        E_minus = np.sum(np.abs(ψ_minus)**2)

        hybrid_energy[f"{i}_{j}"].append((E_plus, E_minus))

# ذخیره فایل‌های خروجی
np.save(os.path.join(output_dir, "hybrid_energy_corrected.npy"), hybrid_energy)

with open(os.path.join(output_dir, "hybrid_energy_summary_corrected.txt"), "w", encoding="utf-8") as f:
    f.write("⚛ Hybrid Mode Energies E = ∫|ψ|² dx (based on w(x,t))\n\n")
    for key, values in hybrid_energy.items():
        i, j = key.split("_")
        f.write(f"Hybrid Modes ψ_{i} ± ψ_{j}:\n")
        for idx, t in enumerate(critical_timesteps):
            e_plus, e_minus = values[idx]
            f.write(f" t={t}:   E₊ = {e_plus:.4e},   E₋ = {e_minus:.4e}\n")
        f.write("\n")

# رسم نمودار انرژی مدهای ترکیبی
plt.figure(figsize=(10, 6))
for key, values in hybrid_energy.items():
    E_plus = [v[0] for v in values]
    E_minus = [v[1] for v in values]
    plt.plot(critical_timesteps, E_plus, label=f"E₊: ψ_{key.replace('_',' + ')}")
    plt.plot(critical_timesteps, E_minus, linestyle='--', label=f"E₋: ψ_{key.replace('_',' - ')}")

plt.xlabel("Time t")
plt.ylabel("Hybrid Mode Energy")
plt.title("Energy of Hybrid Quantum Modes (from w)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "hybrid_mode_energy_plot_corrected.png"))
plt.close()