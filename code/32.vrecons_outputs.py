import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline

# تنظیم اولیه
os.environ["OMP_NUM_THREADS"] = "4"
n_chi, n_theta, n_phi = 400, 400, 400
slice_range = range(160, 241)  # سطح مقطع‌های گسترده χ
bin_count = 100
min_valid_points = 70
w_range_clip = (-20, 20)
critical_times = [2, 10, 25, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# مسیرهای ورودی و خروجی
w_dir = 'w_output'
L_dir = 'lagrangian_outputs'
out_dir = 'vrecons_outputs_v3'
os.makedirs(out_dir, exist_ok=True)

# اجرای مرحله ۳.۳ برای هر زمان
for t in critical_times:
    print(f"🔁 Processing t={t}...")

    try:
        w_path = os.path.join(w_dir, f"w_t{t}.npy")
        L_path = os.path.join(L_dir, f"L_density_t{t}.npy")
        w_data = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))
        L_data = np.memmap(L_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

        all_w, all_L = [], []

        for chi in slice_range:
            w_slice = w_data[chi].flatten()
            L_slice = L_data[chi].flatten()

            mask = np.isfinite(w_slice) & np.isfinite(L_slice)
            w_valid = w_slice[mask]
            L_valid = L_slice[mask]

            clip_mask = (w_valid >= w_range_clip[0]) & (w_valid <= w_range_clip[1])
            w_valid = w_valid[clip_mask]
            L_valid = L_valid[clip_mask]

            all_w.append(w_valid)
            all_L.append(L_valid)

        w_all = np.concatenate(all_w)
        L_all = np.concatenate(all_L)

        if len(w_all) < min_valid_points:
            print(f"⚠️ Skipping t={t}: Not enough valid points after all slices.")
            continue

        # Bin کردن
        bins = np.linspace(w_range_clip[0], w_range_clip[1], bin_count + 1)
        bin_indices = np.digitize(w_all, bins) - 1
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        V_rec = np.zeros(bin_count)
        counts = np.zeros(bin_count)

        for i in range(bin_count):
            idx = (bin_indices == i)
            if np.any(idx):
                V_rec[i] = -np.mean(L_all[idx])
                counts[i] = np.sum(idx)

        # حذف binهای بدون داده
        valid = counts > 0
        w_binned = bin_centers[valid]
        V_rec = V_rec[valid]

        if len(w_binned) < min_valid_points:
            print(f"⚠️ Skipping t={t}: Not enough unique binned points ({len(w_binned)})")
            continue

        # بازسازی V(w)
        dw = np.diff(w_binned)
        if np.any(dw == 0):
            print(f"⚠️ Skipping t={t}: Zero differences in dw.")
            continue

        v_avg = 0.5 * (V_rec[:-1] + V_rec[1:])
        V_cum = np.concatenate([[0], np.cumsum(v_avg * dw)])

        spline = UnivariateSpline(w_binned, V_cum, s=1e-2 * len(V_cum))
        w_dense = np.linspace(w_binned.min(), w_binned.max(), 1000)
        V_dense = spline(w_dense)

        if np.all(np.isnan(V_dense)) or np.all(V_dense == 0):
            print(f"⚠️ Skipping t={t}: All NaNs or zeros in V(w)")
            continue

        # ذخیره خروجی‌ها
        np.save(os.path.join(out_dir, f'Vw_data_t{t}.npy'), np.vstack([w_dense, V_dense]))
        spline_out = InterpolatedUnivariateSpline(w_dense, V_dense, k=3)
        np.save(os.path.join(out_dir, f'spline_V_t{t}.npy'), spline_out(w_dense))

        plt.figure(figsize=(6, 4))
        plt.plot(w_dense, V_dense, label='V(w)', color='darkblue')
        plt.scatter(w_binned, V_cum, s=5, alpha=0.3, label='Reconstructed')
        plt.xlabel('w')
        plt.ylabel('V(w)')
        plt.title(f'Reconstructed Quantum Potential at t={t}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'Vw_plot_t{t}.png'))
        plt.close()

        print(f"✅ Done: t={t}, valid points: {len(w_binned)}")

    except Exception as e:
        print(f"⛔ Error at t={t}: {e}")