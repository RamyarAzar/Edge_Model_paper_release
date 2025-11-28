import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

# تنظیم مسیرها و پارامترها
w_dir = 'w_output'
vprime_dir = 'vprime_outputs'
output_dir = 'vfit_density_overlay'
os.makedirs(output_dir, exist_ok=True)

n_chi, n_theta, n_phi = 400, 400, 400
critical_times = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]

for t in critical_times:
    print(f"⏳ Preparing overlay at t={t}...")

    try:
        # ---------- مرحله ۱: بازخوانی و برش ----------
        w_data = np.memmap(os.path.join(w_dir, f"w_t{t}.npy"), dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))
        w_slice = w_data[n_chi // 2, :, :]

        dVdw_data = np.load(os.path.join(vprime_dir, f"dVdw_t{t}.npy"))
        if dVdw_data.shape != w_slice.shape:
            raise ValueError("⛔ Shape mismatch")

        # ---------- مرحله ۲: آماده‌سازی داده‌ها ----------
        w_flat = w_slice.flatten()
        dVdw_flat = dVdw_data.flatten()
        mask = np.isfinite(w_flat) & np.isfinite(dVdw_flat)
        w_valid = w_flat[mask]
        dVdw_valid = dVdw_flat[mask]

        # حذف نقاط پرت (۵ انحراف معیار)
        w_mean, w_std = np.mean(w_valid), np.std(w_valid)
        mask_clip = np.abs(w_valid - w_mean) < 5 * w_std
        w_clipped = w_valid[mask_clip]
        dVdw_clipped = dVdw_valid[mask_clip]

        # مرتب‌سازی برای انتگرال‌گیری
        idx_sort = np.argsort(w_clipped)
        w_sorted = w_clipped[idx_sort]
        dVdw_sorted = dVdw_clipped[idx_sort]

        # ---------- مرحله ۳: بازسازی V(w) ----------
        dw = np.diff(w_sorted)
        v_avg = 0.5 * (dVdw_sorted[:-1] + dVdw_sorted[1:])
        V_rec = -np.concatenate([[0], np.cumsum(v_avg * dw)])

        spline = UnivariateSpline(w_sorted, V_rec, s=1e-3 * len(V_rec))
        w_dense = np.linspace(w_sorted.min(), w_sorted.max(), 1000)
        V_dense = spline(w_dense)

        # ---------- مرحله ۴: چگالی گره‌ها از هیستوگرام ----------
        hist_density, bin_edges = np.histogram(w_clipped, bins=100, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # ---------- مرحله ۵: ترسیم نمودار ----------
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(w_dense, V_dense, 'k', label='V(w)')
        ax1.set_xlabel('Normalized w')
        ax1.set_ylabel('V(w)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        ax2 = ax1.twinx()
        ax2.plot(bin_centers, hist_density, 'b--', label='Node Density')
        ax2.set_ylabel('Node Density', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        plt.title(f'Density Overlay at t={t}')
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f'overlay_density_t{t}.png'))
        plt.close()

        print(f"✅ Overlay ready for t={t}")

    except Exception as e:
        print(f"❌ Error at t={t}: {e}")