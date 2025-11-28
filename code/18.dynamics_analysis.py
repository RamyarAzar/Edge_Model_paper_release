import numpy as np
import os
import matplotlib.pyplot as plt

# پارامترها
critical_timesteps = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]
n_chi, n_theta, n_phi = 400, 400, 400
sigma = 0.2
w_dm = 0.2
w_dir = "w_output"
results_dir = "dynamics_analysis"
os.makedirs(results_dir, exist_ok=True)

# توابع کمکی
def compute_fractions(w, sigma=0.2, w_dm=0.2):
    f_m = np.exp(- (w)**2 / (2 * sigma**2))
    f_dm = np.exp(- (w - w_dm)**2 / (2 * sigma**2))
    f_de = np.exp(- (w + 1)**2 / (2 * sigma**2))
    norm = f_m + f_dm + f_de
    norm = np.where(norm == 0, 1e-8, norm)
    return f_m / norm, f_dm / norm, f_de / norm

# فضای حجم (وزن‌دار بر حسب θ)
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2 * np.pi, n_phi)
dtheta = np.pi / n_theta
dphi = 2 * np.pi / n_phi
sin_theta = np.sin(theta)
volume_element = sin_theta[:, None] * dtheta * dphi

# نتایج نهایی
E_m_all, E_dm_all, E_de_all = [], [], []
mean_w_all = []

for t in critical_timesteps:
    print(f"\n⏳ Analyzing energy components for t={t}...")

    # بارگذاری w
    w = np.memmap(os.path.join(w_dir, f"w_t{t}.npy"), dtype=np.float64, mode='r',
                  shape=(n_chi, n_theta, n_phi))

    # محاسبه ضرایب
    f_m, f_dm, f_de = compute_fractions(w, sigma, w_dm)

    # اعمال وزن حجمی
    f_m_weighted = f_m * w * volume_element
    f_dm_weighted = f_dm * w * volume_element
    f_de_weighted = f_de * w * volume_element

    # انتگرال‌گیری روی کل فضا
    E_m = np.sum(f_m_weighted)
    E_dm = np.sum(f_dm_weighted)
    E_de = np.sum(f_de_weighted)

    E_m_all.append(E_m)
    E_dm_all.append(E_dm)
    E_de_all.append(E_de)
    mean_w_all.append(np.mean(w))

    print(f"✅ t={t}: E_m={E_m:.3e}, E_dm={E_dm:.3e}, E_de={E_de:.3e}")

# ذخیره داده‌ها
np.savez(os.path.join(results_dir, "energy_components.npz"),
         t=critical_timesteps,
         E_m=np.array(E_m_all),
         E_dm=np.array(E_dm_all),
         E_de=np.array(E_de_all),
         mean_w=np.array(mean_w_all))

# 📊 رسم نمودار انرژی‌ها
plt.figure(figsize=(10, 6))
plt.plot(critical_timesteps, E_m_all, label="E_m (matter)")
plt.plot(critical_timesteps, E_dm_all, label="E_dm (dark matter)")
plt.plot(critical_timesteps, E_de_all, label="E_de (dark energy)")
plt.xlabel("t (critical)")
plt.ylabel("Energy Integral")
plt.title("Energy Components vs Time")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, "energy_vs_time.png"))

# 📊 رسم نمودار میانگین w
plt.figure(figsize=(8, 5))
plt.plot(critical_timesteps, mean_w_all, marker='o')
plt.xlabel("t")
plt.ylabel("⟨w⟩")
plt.title("Mean w vs Time")
plt.grid(True)
plt.savefig(os.path.join(results_dir, "mean_w_vs_time.png"))

print("\n🎯 All results saved in 'dynamics_analysis/'")