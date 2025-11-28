import numpy as np
import matplotlib.pyplot as plt
import os

# مسیرهای ورودی/خروجی
w_dir = 'w_output'
out_dir = 'phase_analysis_outputs'
os.makedirs(out_dir, exist_ok=True)

# پارامترهای شبکه
n_chi, n_theta, n_phi = 400, 400, 400
critical_times = [33, 34, 35, 36, 37, 38, 39, 40, 41]

for t in critical_times:
    try:
        print(f"🔁 Processing phase decomposition for t={t}...")

        # بارگذاری میدان w(x,t) – فرض بر این است که داده‌ها واقعی هستند
        w_data = np.memmap(os.path.join(w_dir, f"w_t{t}.npy"), dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

        # ساخت نسخه مختلط میدان:
        # فرض: w حقیقی است و ما فاز مؤثر را با مشتق زمانی استخراج می‌کنیم (ساده‌سازی مرسوم)
        # در حالت کلی، اگر w مختلط باشد: w = A * exp(i*phi)
        # ما از مشتق زمانی به عنوان مشتق فاز استفاده می‌کنیم:
        if not all(os.path.exists(os.path.join(w_dir, f"w_t{tp}.npy")) for tp in [t-1, t+1]):
            print(f"⚠️ Skipping t={t}: missing neighboring frames for phase estimate")
            continue

        w_prev = np.memmap(os.path.join(w_dir, f"w_t{t-1}.npy"), dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))
        w_next = np.memmap(os.path.join(w_dir, f"w_t{t+1}.npy"), dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))
        w_dot = (w_next - w_prev) / 2.0  # مشتق زمانی

        # تعریف میدان مختلط مؤثر
        w_complex = w_data + 1j * w_dot

        # استخراج دامنه و فاز
        amplitude = np.abs(w_complex)
        phase = np.angle(w_complex)

        # ذخیره مقاطع
        amp_slice = amplitude[n_chi // 2, :, :]
        phase_slice = phase[n_chi // 2, :, :]

        plt.figure(figsize=(6, 5))
        plt.imshow(amp_slice, cmap='inferno', origin='lower')
        plt.colorbar(label='|w|')
        plt.title(f'Amplitude |w| slice at t={t}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'amp_slice_t{t}.png'))
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.imshow(phase_slice, cmap='twilight', origin='lower')
        plt.colorbar(label='Arg(w)')
        plt.title(f'Phase Arg(w) slice at t={t}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'phase_slice_t{t}.png'))
        plt.close()

        # ذخیره داده برای مراحل بعد
        np.save(os.path.join(out_dir, f'amp_t{t}.npy'), amplitude)
        np.save(os.path.join(out_dir, f'phase_t{t}.npy'), phase)

        print(f"✅ Done t={t}")

    except Exception as e:
        print(f"⛔ Error at t={t}: {e}")