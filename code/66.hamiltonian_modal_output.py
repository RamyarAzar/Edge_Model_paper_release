import numpy as np
import os
import matplotlib.pyplot as plt

# مسیر فایل‌های ورودی
c_kt_path = "mode_coeff_output/c_kt.npy"
output_dir = "hamiltonian_modal_output"
os.makedirs(output_dir, exist_ok=True)

# بارگذاری ضرایب مدها
c_kt_all = np.load(c_kt_path)  # shape: (n_t, n_modes)

n_t, n_modes = c_kt_all.shape
H_modal_all = []

for i in range(n_t):
    c_kt = c_kt_all[i]
    E_k = np.abs(c_kt)**2
    H_modal = np.diag(E_k)
    H_modal_all.append(H_modal)

# ذخیره ساختار هامیلتونی برای هر زمان
H_modal_all = np.array(H_modal_all)  # shape: (n_t, n_modes, n_modes)
np.save(os.path.join(output_dir, "H_modal_t.npy"), H_modal_all)

# ذخیره بردار حالت برای هر زمان
np.save(os.path.join(output_dir, "psi_state_t.npy"), c_kt_all)

# خلاصه‌سازی خروجی
with open(os.path.join(output_dir, "hamiltonian_modal_summary.txt"), "w", encoding='utf-8') as f:
    f.write("🧠 Modal Hamiltonian and Quantum States Summary\n\n")
    for i in range(n_t):
        f.write(f"t={33+i}:\n")
        for k in range(n_modes):
            ck = c_kt_all[i, k]
            ek = np.abs(ck)**2
            f.write(f"  Mode {k+1}: c_k = {ck:.4e} | E_k = {ek:.4e}\n")
        f.write("\n")

# رسم نمودار انرژی هر مد
for k in range(n_modes):
    plt.plot(range(33, 33+n_t), np.abs(c_kt_all[:, k])**2, label=f"Mode {k+1}")

plt.xlabel("Time t")
plt.ylabel("Energy E_k = |c_k|²")
plt.title("Modal Hamiltonian Energies Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "modal_hamiltonian_energy_plot.png"))
plt.close()