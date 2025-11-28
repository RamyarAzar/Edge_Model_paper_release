import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات
timesteps = list(range(33, 42))
shape = (400, 400, 400)
w_dir = "w_output"
eff_dir = "effective_field_output"
output_dir = "lagrangian_phase_recon"
os.makedirs(output_dir, exist_ok=True)

# تعریف R هندسی (برای آزمایش فقط l=2 مود لحاظ شده)
def compute_R_grid(chi, theta, t, R0=1.0, sigma=0.3, a2=0.1):
    return R0 * (1 + a2 * np.sin(chi) * np.exp(-chi**2 / sigma**2) * np.polynomial.legendre.legval(np.cos(theta), [0, 0, 1]))

# مشتق مرکزی فضای سه‌بعدی
def spatial_grad_sq(w, dx):
    grad_sq = np.zeros_like(w)
    for axis in range(3):
        grad = np.gradient(w, dx, axis=axis)
        grad_sq += grad**2
    return grad_sq

# چگالی لاگرانژی
def compute_lagrangian_density(w_t2, grad_sq, veff, R, chi, theta):
    term1 = 0.5 * w_t2
    term2 = 0.5 * grad_sq / (R**2)
    return term1 - term2 - veff

# لیست برای لاگرانژی کل
L_list = []

# حلقه زمانی
for t in timesteps:
    print(f"⏳ Processing Lagrangian at t={t}...")

    # بارگذاری
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=shape)
    w_t2 = np.load(os.path.join(eff_dir, f"phase_kinetic_t{t}.npy"))
    veff = np.load(os.path.join(eff_dir, f"veff_t{t}.npy"))

    dx = 1.0 / 400
    grad_sq = spatial_grad_sq(w, dx)

    # مش زاویه‌ای (برای 400×400 کافیست)
    chi_vals = np.linspace(0, np.pi, shape[0])
    theta_vals = np.linspace(0, np.pi, shape[1])
    chi_grid, theta_grid = np.meshgrid(chi_vals, theta_vals, indexing='ij')

    # R بازسازی‌شده
    R_grid = compute_R_grid(chi_grid, theta_grid, t)
    R_3D = R_grid[:, :, None]  # تبدیل به 3D برای ضرب تانسوری

    # وزن حجم کروی
    measure = R_3D**3 * np.sin(chi_grid[:, :, None])**2 * np.sin(theta_grid[:, :, None])

    # چگالی لاگرانژی و انتگرال‌گیری
    lag_density = compute_lagrangian_density(w_t2, grad_sq, veff, R_3D, chi_grid[:, :, None], theta_grid[:, :, None])
    L_total = np.sum(lag_density * measure)

    # ذخیره‌سازی
    np.save(os.path.join(output_dir, f"lagrangian_density_t{t}.npy"), lag_density)
    np.save(os.path.join(output_dir, f"lagrangian_total_t{t}.npy"), L_total)
    L_list.append((t, L_total))

# 📊 ذخیره داده‌ها
L_array = np.array(L_list)
np.savetxt(os.path.join(output_dir, "lagrangian_summary.txt"), L_array, header="t\tL(t)")

# 📈 رسم نمودار
plt.figure(figsize=(8, 5))
plt.plot(L_array[:, 0], L_array[:, 1], marker='o', lw=2)
plt.title("Total Lagrangian L(t) over Time")
plt.xlabel("Time step t")
plt.ylabel("L(t)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "lagrangian_summary_plot.png"))
plt.close()