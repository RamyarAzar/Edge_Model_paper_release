import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# مسیر داده‌ها
w_dir = "w_output"  # مسیر فایل‌های w با np.memmap
output_dir = "quantum_outputs"
os.makedirs(output_dir, exist_ok=True)

# پارامترهای مدل
hbar = 1.0
m_eff = 1.0
dx = dtheta = dphi = 1.0  # فرض اولیه، در صورت نیاز دقیق‌تر تنظیم شود

# زمان‌های بحرانی موجود
t_vals = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]

for i, t in enumerate(t_vals):
    print(f"🔁 Computing quantum structure at t={t}...")

    # بارگذاری w(t)
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r')
    w = w.reshape((400, 400, 400))  # فرض ساختار شبکه

    # حذف مقادیر غیرمعتبر
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    # تابع موج مؤثر
    psi = np.sqrt(np.abs(w))
    psi[psi == 0] = 1e-12  # جلوگیری از تقسیم بر صفر

    # گرادیان w
    dw_dx, dw_dtheta, dw_dphi = np.gradient(w, dx, dtheta, dphi)

    # گرادیان Ψ برای لاپلاسین
    dpsi_dx, dpsi_dtheta, dpsi_dphi = np.gradient(psi, dx, dtheta, dphi)
    d2psi_dx2 = np.gradient(dpsi_dx, dx, axis=0)
    d2psi_dtheta2 = np.gradient(dpsi_dtheta, dtheta, axis=1)
    d2psi_dphi2 = np.gradient(dpsi_dphi, dphi, axis=2)
    laplacian_psi = d2psi_dx2 + d2psi_dtheta2 + d2psi_dphi2

    # تکانه مؤثر
    p_eff_x = dw_dx / (2 * w + 1e-12)
    p_eff_theta = dw_dtheta / (2 * w + 1e-12)
    p_eff_phi = dw_dphi / (2 * w + 1e-12)
    p_eff_mag = np.sqrt(p_eff_x**2 + p_eff_theta**2 + p_eff_phi**2)

    # انرژی مؤثر (فقط اگر t±1 موجود باشند)
    if i > 0 and i < len(t_vals) - 1:
        t_prev = t_vals[i - 1]
        t_next = t_vals[i + 1]
        w_prev = np.memmap(os.path.join(w_dir, f"w_t{t_prev}.npy"), dtype='float64', mode='r').reshape((400, 400, 400))
        w_next = np.memmap(os.path.join(w_dir, f"w_t{t_next}.npy"), dtype='float64', mode='r').reshape((400, 400, 400))
        dt = (t_next - t_prev)
        dw_dt = (w_next - w_prev) / dt
        E_eff = dw_dt / (2 * w + 1e-12)
    else:
        E_eff = np.zeros_like(w)

    # پتانسیل مؤثر
    V_eff = E_eff + (hbar**2 / (2 * m_eff)) * laplacian_psi / (psi + 1e-12)

    # ذخیره‌سازی داده‌ها
    np.save(os.path.join(output_dir, f"psi_t{t}.npy"), psi.astype(np.float64))
    np.save(os.path.join(output_dir, f"p_eff_t{t}.npy"), p_eff_mag.astype(np.float64))
    np.save(os.path.join(output_dir, f"E_eff_t{t}.npy"), E_eff.astype(np.float64))
    np.save(os.path.join(output_dir, f"V_eff_t{t}.npy"), V_eff.astype(np.float64))

    # ترسیم مقطع χ=200
    chi_slice = 200
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(p_eff_mag[chi_slice], cmap='plasma')
    plt.title(f"|p_eff| at t={t}")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(E_eff[chi_slice], cmap='viridis')
    plt.title(f"E_eff at t={t}")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(V_eff[chi_slice], cmap='inferno')
    plt.title(f"V_eff at t={t}")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"quantum_slices_t{t}.png"), dpi=150)
    plt.close()
    print(f"✅ Saved quantum outputs for t={t}")