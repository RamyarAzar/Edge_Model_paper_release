import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import os

# تنظیمات مسیرها
vfit_dir = 'vfit_outputs'
w_dir = 'w_output'
output_dir = 'validation_outputs'
os.makedirs(output_dir, exist_ok=True)

critical_times = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]
n_chi, n_theta, n_phi = 400, 400, 400

for t in critical_times:
    print(f'🔍 Analyzing t={t}...')

    try:
        # بارگذاری V(w) بازسازی‌شده
        vw_file = os.path.join(vfit_dir, f'Vw_dense_t{t}.npy')
        vw_data = np.load(vw_file)
        w_dense, V_dense = vw_data[0], vw_data[1]

        # شناسایی مینیمم‌ها و ماکزیمم‌ها در V(w)
        local_min_idx = argrelextrema(V_dense, np.less)[0]
        local_max_idx = argrelextrema(V_dense, np.greater)[0]
        w_minima = w_dense[local_min_idx]
        w_maxima = w_dense[local_max_idx]

        # بارگذاری فیلد w و استخراج مقطع
        w_file = os.path.join(w_dir, f'w_t{t}.npy')
        w_data = np.memmap(w_file, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))
        w_slice = w_data[n_chi // 2, :, :]
        w_flat = w_slice.flatten()
        w_norm = (w_flat - np.mean(w_flat)) / np.std(w_flat)

        # ساخت هیستوگرام چگالی گره‌ها روی محور w
        hist, bin_edges = np.histogram(w_norm, bins=200, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # نمودار تطبیق چگالی w با V(w)
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(w_dense, V_dense, 'k-', label='V(w)')
        ax1.set_ylabel('V(w)', color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.set_xlabel('Normalized w')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(bin_centers, hist, 'b--', label='w node density')
        ax2.set_ylabel('Node Density', color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        fig.suptitle(f'Density Overlay at t={t}')
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f'overlay_density_t{t}.png'))
        plt.close()

        # ذخیره موقعیت مینیمم/ماکزیمم برای تحلیل بعدی
        np.savez(os.path.join(output_dir, f'extrema_t{t}.npz'), w_minima=w_minima, w_maxima=w_maxima)

        print(f"✅ Done: t={t}, min/max points: {len(w_minima)}/{len(w_maxima)}")

    except Exception as e:
        print(f"⛔ Error at t={t}: {e}")
