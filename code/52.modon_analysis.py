import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# مسیرهای فایل
w_dir = 'w_output'
eff_dir = 'effective_field'
modon_output = 'modon_analysis_outputs'
os.makedirs(modon_output, exist_ok=True)

# ابعاد شبکه
n_chi, n_theta, n_phi = 400, 400, 400

# بازه زمانی هدف
time_range = list(range(33, 42))

# پارامترهای فیلتر برای حذف نویز کوتاه‌مقیاس و تشخیص ساختارهای همدوس
smooth_sigma = 2
threshold_fraction = 0.2  # آستانه تشخیص modon نسبت به بیشینه چگالی

for t in time_range:
    print(f"🔍 Processing t={t}...")
    try:
        # بارگذاری w با memmap
        w_path = os.path.join(w_dir, f"w_t{t}.npy")
        w = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

        # فیلتر گوسی سه‌بعدی برای صاف‌سازی ساختارها
        w_smooth = gaussian_filter(w, sigma=smooth_sigma)

        # آستانه برای استخراج ساختارهای همدوس (modonها)
        threshold = threshold_fraction * np.max(np.abs(w_smooth))
        modon_mask = np.abs(w_smooth) > threshold

        # شمارش و خلاصه‌سازی modonها
        num_modon_voxels = np.count_nonzero(modon_mask)
        modon_volume_fraction = num_modon_voxels / w.size

        # خروجی تصویری برای مقطع χ = 200
        slice_img = modon_mask[n_chi // 2, :, :].astype(int)

        plt.figure(figsize=(6, 5))
        plt.imshow(slice_img, cmap='gray', origin='lower', aspect='auto')
        plt.title(f'Modon Structures (χ=200) at t={t}')
        plt.xlabel('φ')
        plt.ylabel('θ')
        plt.tight_layout()
        plt.savefig(os.path.join(modon_output, f'modon_slice_t{t}.png'))
        plt.close()

        # ذخیره نتایج آماری
        with open(os.path.join(modon_output, f'modon_stats_t{t}.txt'), 'w') as f:
            f.write(f"Time step: {t}\n")
            f.write(f"Threshold: {threshold:.4e}\n")
            f.write(f"Modon voxels: {num_modon_voxels}\n")
            f.write(f"Volume fraction: {modon_volume_fraction:.6f}\n")

        print(f"✅ Done: t={t}, modons detected and saved.")

    except Exception as e:
        print(f"⛔ Error at t={t}: {e}")
