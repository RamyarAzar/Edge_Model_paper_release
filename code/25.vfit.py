import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

# تنظیمات مسیرها و پارامترها
w_dir = 'w_output'
vprime_dir = 'vprime_outputs'
output_dir = 'vfit_outputs'
os.makedirs(output_dir, exist_ok=True)

n_chi, n_theta, n_phi = 400, 400, 400
critical_times = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]

for t in critical_times:
    print(f"🔁 Processing t={t}...")

    try:
        # خواندن w با memmap
        w_file = os.path.join(w_dir, f"w_t{t}.npy")
        w_data = np.memmap(w_file, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

        # خواندن dV/dw
        v_file = os.path.join(vprime_dir, f"dVdw_t{t}.npy")
        v_data = np.load(v_file)

        # انتخاب مقطع ۲ بعدی χ=200
        w_slice = w_data[n_chi // 2, :, :]

        # تطبیق شکل‌ها
        if v_data.shape != w_slice.shape:
            raise ValueError("⛔ Shape mismatch between w_slice and dV/dw")

        # مسطح‌سازی و فیلتراسیون اولیه
        w_flat = w_slice.flatten()
        v_flat = v_data.flatten()
        mask = np.isfinite(w_flat) & np.isfinite(v_flat)

        w_valid = w_flat[mask]
        v_valid = v_flat[mask]

        # حذف نقاط پرت (برای جلوگیری از انفجار عددی)
        w_mean = np.mean(w_valid)
        w_std = np.std(w_valid)
        std_clip = 5  # حداکثر ۵ انحراف معیار
        mask_clip = np.abs(w_valid - w_mean) < std_clip * w_std

        w_valid = w_valid[mask_clip]
        v_valid = v_valid[mask_clip]

        # نرمال‌سازی اولیه w
        w_norm = (w_valid - w_valid.mean()) / w_valid.std()

        # مرتب‌سازی برای انتگرال‌گیری
        idx_sort = np.argsort(w_norm)
        w_sorted = w_norm[idx_sort]
        v_sorted = v_valid[idx_sort]

        # بازسازی V(w) با انتگرال‌گیری عددی
        dw = np.diff(w_sorted)
        v_avg = 0.5 * (v_sorted[:-1] + v_sorted[1:])
        V_rec = -np.concatenate([[0], np.cumsum(v_avg * dw)])

        # فیت اسپلاین روی V(w)
        spline = UnivariateSpline(w_sorted, V_rec, s=1e-3 * len(V_rec))
        w_dense = np.linspace(w_sorted.min(), w_sorted.max(), 1000)
        V_dense = spline(w_dense)

        # نمودار پراکندگی اولیه برای sanity check
        plt.figure(figsize=(6, 4))
        plt.scatter(w_sorted, V_rec, s=2, alpha=0.5, label='Raw')
        plt.plot(w_dense, V_dense, 'r', label='Fitted V(w)')
        plt.xlabel('Normalized w')
        plt.ylabel('V(w)')
        plt.title(f'Quantum Potential Fit at t={t}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'Vw_fit_t{t}.png'))
        plt.close()

        # ذخیره داده‌های نهایی
        np.save(os.path.join(output_dir, f'Vw_dense_t{t}.npy'), np.vstack([w_dense, V_dense]))

        print(f"✅ Done: t={t}, data points = {len(w_valid)}")

    except Exception as e:
        print(f"⛔ Error at t={t}: {e}")