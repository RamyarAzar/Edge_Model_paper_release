import numpy as np
import os
from numpy import gradient
import matplotlib.pyplot as plt

# مسیرها
w_dir = "w_output"
phase_dir = "effective_field_output"
output_dir = "gauge_spatial_output"
os.makedirs(output_dir, exist_ok=True)

# تنظیمات
critical_timesteps = [33, 34, 35, 36, 37, 38, 39, 40, 41]
n_chi, n_theta, n_phi = 400, 400, 400
epsilon = 1e-10

# خروجی نهایی
mean_Ai_list = []

for t in critical_timesteps:
    print(f"🔧 Processing spatial gauge field at t={t}...")

    # بارگذاری w(x,t)
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

    # بارگذاری مشتق فازی φ̇(x,t)
    phase_path = os.path.join(phase_dir, f"phase_kinetic_t{t}.npy")
    phase_kinetic = np.load(phase_path)

    # محاسبه S_eff(x,t)
    s_eff = w * phase_kinetic

    # محاسبه گرادیان فاز ∇φ
    grad_phi = np.gradient(phase_kinetic)
    grad_phi_sq = sum([g**2 for g in grad_phi])
    hbar_eff = np.sqrt(grad_phi_sq) + epsilon

    # محاسبه گرادیان S_eff
    grad_seff = np.gradient(s_eff)

    # محاسبه میدان گیج فضایی A_i(x,t)
    A_spatial = np.zeros((3, n_chi, n_theta, n_phi))
    for i in range(3):
        A_spatial[i] = grad_seff[i] / hbar_eff

    # ذخیره فایل میدان گیج فضایی
    np.save(os.path.join(output_dir, f"A_spatial_t{t}.npy"), A_spatial)

    # محاسبه میانگین برای خلاصه
    mean_A = [np.mean(A_spatial[i]) for i in range(3)]
    mean_Ai_list.append(mean_A)

# ذخیره خلاصه میانگین‌ها
mean_Ai_array = np.array(mean_Ai_list)  # shape: (len(t), 3)
np.save(os.path.join(output_dir, "A_spatial_means.npy"), mean_Ai_array)

# ذخیره txt
with open(os.path.join(output_dir, "gauge_field_spatial_summary.txt"), "w", encoding="utf-8") as f:
    f.write("📡 Spatial Gauge Field A_i(x,t) Summary\n\n")
    for i, t in enumerate(critical_timesteps):
        A1, A2, A3 = mean_Ai_array[i]
        f.write(f"t={t:2d} | ⟨A_χ⟩={A1:.4e}, ⟨A_θ⟩={A2:.4e}, ⟨A_φ⟩={A3:.4e}\n")

# نمودار میانگین‌ها
labels = ['⟨A_χ⟩', '⟨A_θ⟩', '⟨A_φ⟩']
for i in range(3):
    plt.plot(critical_timesteps, mean_Ai_array[:, i], marker='o', label=labels[i])
plt.title("Mean Spatial Gauge Field Components ⟨A_i(t)⟩")
plt.xlabel("Critical Time t")
plt.ylabel("⟨A_i(t)⟩")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gauge_spatial_means_plot.png"))
plt.close()