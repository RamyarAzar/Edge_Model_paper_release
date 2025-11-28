import numpy as np
import os

# مسیر ورودی و خروجی
amp_dir = 'phase_analysis_outputs'
phase_dir = 'phase_analysis_outputs'
out_dir = 'wavefunction_outputs'
os.makedirs(out_dir, exist_ok=True)

# استخراج لیست زمان‌هایی که دادهٔ آمپلیتود و فاز دارند
available_times = sorted(
    list(set(
        int(f.split('_')[1][1:].split('.')[0])
        for f in os.listdir(amp_dir)
        if f.endswith('.npy') and 'amp' in f
    ))
)

for t in available_times:
    try:
        print(f"🔁 Reconstructing Ψ(x,t) at t={t}...")

        # بارگذاری دامنه و فاز
        amp = np.load(os.path.join(amp_dir, f'amp_t{t}.npy')).astype(np.float32)
        phase = np.load(os.path.join(phase_dir, f'phase_t{t}.npy')).astype(np.float32)

        # نرمال‌سازی فاز از [0, 255] به [0, 2π]
        phase_rad = 2 * np.pi * (phase / 255.0)

        # بازسازی تابع موج: Ψ = A * exp(i * φ)
        psi = amp * np.exp(1j * phase_rad)  # dtype: complex64

        # ذخیره به صورت npy (برای فازهای بعدی مثل تحلیل گره‌ها)
        np.save(os.path.join(out_dir, f'psi_t{t}.npy'), psi.astype(np.complex64))

        print(f"✅ Saved Ψ(x,t) for t={t}")

    except Exception as e:
        print(f"⛔ Error at t={t}: {e}")