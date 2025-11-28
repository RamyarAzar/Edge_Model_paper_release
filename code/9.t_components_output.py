import numpy as np
import os
import time

# تنظیمات
critical_timesteps = [2, 39, 42, 45, 47]
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
block_size = 10
sigma = 0.2
w_dm = 0.2

# مسیر فایل‌ها
q_dir = "q_output"
w_dir = "w_output"
output_base = "t_components_output"
mask_output = "t_mask_output"
os.makedirs(output_base, exist_ok=True)
os.makedirs(mask_output, exist_ok=True)

for t in critical_timesteps:
    print(f"\n⏳ Processing T components for t={t}...")
    t0 = time.time()

    # بارگذاری داده‌ها با حافظه‌نگاشت
    Q = np.memmap(os.path.join(q_dir, f"Q_t{t}.npy"), dtype=np.float64, mode='r',
                  shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    w = np.memmap(os.path.join(w_dir, f"w_t{t}.npy"), dtype=np.float64, mode='r',
                  shape=(n_chi, n_theta, n_phi))

    # فایل‌های خروجی
    T_m = np.memmap(os.path.join(output_base, f"T_m_t{t}.npy"), dtype=np.float64, mode='w+',
                    shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    T_dm = np.memmap(os.path.join(output_base, f"T_dm_t{t}.npy"), dtype=np.float64, mode='w+',
                     shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    T_de = np.memmap(os.path.join(output_base, f"T_de_t{t}.npy"), dtype=np.float64, mode='w+',
                     shape=(n_coords, n_coords, n_chi, n_theta, n_phi))

    # ماسک کلی اختیاری برای نقاط معتبر
    global_mask = np.ones((n_chi, n_theta, n_phi), dtype=bool)

    # پردازش بلوکی
    for i_start in range(0, n_chi, block_size):
        i_end = min(i_start + block_size, n_chi)

        Q_blk = Q[:, :, i_start:i_end].astype(np.float64)
        w_blk = w[i_start:i_end]  # shape: [blk, θ, φ]

        # محاسبه ضرایب فضایی
        f_m = np.exp(- (w_blk)**2 / (2 * sigma**2))
        f_dm = np.exp(- (w_blk - w_dm)**2 / (2 * sigma**2))
        f_de = np.exp(- (w_blk + 1)**2 / (2 * sigma**2))

        norm = f_m + f_dm + f_de
        norm = np.where(norm == 0, 1e-8, norm)  # محافظت در برابر تقسیم بر صفر

        f_m /= norm
        f_dm /= norm
        f_de /= norm

        # ساخت ماسک معتبر بودن (اختیاری برای ذخیره)
        mask_blk = np.isfinite(w_blk) & (norm > 1e-8)
        global_mask[i_start:i_end] = mask_blk

        # reshape برای broadcast
        f_m = f_m[None, None, :, :, :]
        f_dm = f_dm[None, None, :, :, :]
        f_de = f_de[None, None, :, :, :]

        # ضرب و ذخیره
        T_m[:, :, i_start:i_end] = (f_m * Q_blk)
        T_dm[:, :, i_start:i_end] = (f_dm * Q_blk)
        T_de[:, :, i_start:i_end] = (f_de * Q_blk)

    # ذخیره نتایج
    T_m.flush()
    T_dm.flush()
    T_de.flush()

    # ذخیره ماسک اختیاری
    np.save(os.path.join(mask_output, f"mask_valid_t{t}.npy"), global_mask)

    print(f"✅ Done in {time.time() - t0:.2f} s — Components and mask saved.")