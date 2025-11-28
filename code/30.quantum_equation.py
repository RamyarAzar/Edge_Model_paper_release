import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import os

# پارامترهای شبکه
n_chi, n_theta, n_phi = 400, 400, 400

# مسیرها
w_dir = 'w_output'
vrecons_dir = 'vrecons_outputs_v3'
out_dir = 'quantum_equation_outputs'
os.makedirs(out_dir, exist_ok=True)

# زمان‌های در دسترس
available_w_times = {1,2,3,9,10,11,24,25,26,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48}
available_v_times = {2,25,33,34,35,36,37,38,39,40,41,42,43,44}  # به‌روز شده طبق بازسازی مرحله 3.3

for t in sorted(available_v_times):
    if not ({t-1, t+1} <= available_w_times):
        print(f"⚠️ Skipping t={t}: missing neighbors for central difference.")
        continue

    try:
        print(f"🔁 Processing quantum field equation at t={t}...")

        # بازسازی پتانسیل کوانتومی
        v_file = os.path.join(vrecons_dir, f'Vw_data_t{t}.npy')
        w_dense, V_dense = np.load(v_file)
        V_func = InterpolatedUnivariateSpline(w_dense, V_dense, k=3, ext='zeros')

        # بارگذاری w(t−1), w(t), w(t+1)
        w_tm1 = np.memmap(os.path.join(w_dir, f"w_t{t-1}.npy"), dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))
        w_t   = np.memmap(os.path.join(w_dir, f"w_t{t}.npy"),   dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))
        w_tp1 = np.memmap(os.path.join(w_dir, f"w_t{t+1}.npy"), dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

        # مشتق زمانی دوم: ∂²w/∂t² ≈ w(t+1) − 2w(t) + w(t−1)
        w_tt = w_tp1 - 2 * w_t + w_tm1

        # مشتق عددی V(w)
        dw = 1e-5
        V_plus = V_func(w_t + dw)
        V_minus = V_func(w_t - dw)
        dVdw = (V_plus - V_minus) / (2 * dw)

        # باقی‌مانده معادله مؤثر: ∂²w/∂t² + dV/dw
        residual = w_tt + dVdw

        # حذف نقاط نان یا بینهایت
        residual[np.isnan(residual)] = 0
        residual[np.isinf(residual)] = 0

        # ذخیره باقی‌مانده
        np.save(os.path.join(out_dir, f"quantum_rhs_t{t}.npy"), residual)

        # ذخیره تصویری از مقطع χ میانی
        residual_slice = residual[n_chi // 2, :, :]
        plt.figure(figsize=(6, 5))
        plt.imshow(residual_slice, cmap='RdBu', origin='lower', extent=[0, n_phi, 0, n_theta])
        plt.colorbar(label='Residual (∂²w/∂t² + dV/dw)')
        plt.title(f'Quantum Equation Residual at t={t}')
        plt.xlabel('φ')
        plt.ylabel('θ')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"quantum_residual_t{t}.png"))
        plt.close()

        print(f"✅ Done: t={t}")

    except Exception as e:
        print(f"⛔ Error at t={t}: {e}")