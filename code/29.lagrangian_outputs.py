import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات
w_dir = 'w_output'
output_dir = 'lagrangian_outputs'
os.makedirs(output_dir, exist_ok=True)

n_chi, n_theta, n_phi = 400, 400, 400
dx = 1.0  # گام شبکه مکانی (برای χ, θ, φ)
dt = 1.0  # گام زمانی

# زمان‌هایی که مشتق زمانی قابل محاسبه است
valid_times = [10, 25, 30, 33, 36, 39, 42, 45]
critical_map = {10: (2, 25), 25: (10, 30), 30: (25, 33), 33: (30, 36),
                36: (33, 39), 39: (36, 42), 42: (39, 45), 45: (42, 47)}

for t in valid_times:
    print(f" Processing t={t}...")

    try:
        t_prev, t_next = critical_map[t]

        # خواندن w در t-1, t, t+1
        w_prev = np.memmap(os.path.join(w_dir, f"w_t{t_prev}.npy"), dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))
        w_curr = np.memmap(os.path.join(w_dir, f"w_t{t}.npy"), dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))
        w_next = np.memmap(os.path.join(w_dir, f"w_t{t_next}.npy"), dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

        # مشتق زمانی مرکزی
        dwt = (w_next - w_prev) / (2 * dt)

        # مشتقات مکانی (مرکزی)
        dw_chi = (np.roll(w_curr, -1, axis=0) - np.roll(w_curr, 1, axis=0)) / (2 * dx)
        dw_theta = (np.roll(w_curr, -1, axis=1) - np.roll(w_curr, 1, axis=1)) / (2 * dx)
        dw_phi = (np.roll(w_curr, -1, axis=2) - np.roll(w_curr, 1, axis=2)) / (2 * dx)

        # چگالی لاگرانژی کوانتومی (فرم ساده‌شده‌ی کلاسیک): L = ½ (∂_t w)^2 - ½ (∇w)^2
        grad_sq = dw_chi**2 + dw_theta**2 + dw_phi**2
        L_density = 0.5 * dwt**2 - 0.5 * grad_sq

        # ذخیره لاگرانژی کامل برای هر نقطه
        np.save(os.path.join(output_dir, f"L_density_t{t}.npy"), L_density)

        # نمایش مقطع تصویری برای χ=200 جهت بررسی اثر ساده‌سازی
        slice_L = L_density[n_chi // 2, :, :]

        plt.figure(figsize=(6, 5))
        plt.imshow(slice_L, cmap='inferno', origin='lower', aspect='auto')
        plt.colorbar(label='Lagrangian Density')
        plt.title(f'Lagrangian Density Slice at χ=200, t={t}')
        plt.xlabel('φ')
        plt.ylabel('θ')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'L_density_slice_t{t}.png'))
        plt.close()

        print(f" Done: t={t}, L slice saved.")

    except Exception as e:
        print(f"⛔ Error at t={t}: {e}")