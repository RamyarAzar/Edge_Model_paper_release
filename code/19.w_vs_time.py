import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات
w_dir = "w_output"
critical_timesteps = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]
n_chi, n_theta, n_phi = 400, 400, 400

# بارگذاری wها
w_all = []
for t in critical_timesteps:
    w = np.memmap(os.path.join(w_dir, f"w_t{t}.npy"), dtype=np.float64, mode='r', shape=(n_chi, n_theta, n_phi))
    w_all.append(w)
w_all = np.stack(w_all, axis=0)  # shape: (n_t, n_chi, n_theta, n_phi)

# محاسبه مشتقات زمانی
dt = 1  # فرض گام زمانی یکنواخت
dw_dt = np.gradient(w_all, dt, axis=0)
d2w_dt2 = np.gradient(dw_dt, dt, axis=0)

# استخراج آماره‌ها
mean_w = np.mean(w_all, axis=(1,2,3))
mean_dw_dt = np.mean(dw_dt, axis=(1,2,3))
mean_d2w_dt2 = np.mean(d2w_dt2, axis=(1,2,3))

# رسم نمودارها
plt.figure()
plt.plot(critical_timesteps, mean_w, marker='o')
plt.title("⟨w⟩ vs Time")
plt.xlabel("t (critical)"); plt.ylabel("Mean w")
plt.grid(True)
plt.savefig("mean_w_vs_time.png")

plt.figure()
plt.plot(critical_timesteps, mean_dw_dt, marker='o', label="∂w/∂t")
plt.plot(critical_timesteps, mean_d2w_dt2, marker='s', label="∂²w/∂t²")
plt.title("Time Derivatives of w")
plt.xlabel("t (critical)"); plt.ylabel("Mean derivatives")
plt.legend(); plt.grid(True)
plt.savefig("w_time_derivatives.png")

# استخراج انرژی دینامیکی مؤثر
E_dyn = 0.5 * mean_dw_dt**2
plt.figure()
plt.plot(critical_timesteps, E_dyn, marker='^', color='orange')
plt.title("Effective Dynamic Energy (0.5 * (∂w/∂t)²)")
plt.xlabel("t (critical)"); plt.ylabel("E_dyn")
plt.grid(True)
plt.savefig("E_dyn_vs_time.png")

print("✅ تحلیل دینامیکی w آماده شد. نمودارها ذخیره شدند.")
