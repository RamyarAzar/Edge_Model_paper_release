import numpy as np
import os
import matplotlib.pyplot as plt

# مسیرها
w_dir = "w_output"
output_dir = "qft_phase_analysis_output"
os.makedirs(output_dir, exist_ok=True)

# تنظیمات
critical_timesteps = list(range(33, 42))
n_chi, n_theta, n_phi = 400, 400, 400

# نقاط نمونه برای بررسی دقیق
probe_points = [
    (200, 200, 200),
    (100, 100, 100),
    (300, 300, 300)
]

for t in critical_timesteps:
    print(f"\n🔍 Analyzing phase structure of w(x,t) at t={t}...")

    # بارگذاری میدان w(x,t)
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

    # محاسبه فاز میدان (فاز موضعی)
    w_phase = np.angle(w)

    # استخراج گرادیان فاز در امتداد χ
    w_grad_chi = np.gradient(w_phase, axis=0)
    mean_grad_chi = np.mean(np.abs(w_grad_chi))

    # ذخیره‌ی تصویری از فاز میدان در θ = 200
    slice_theta = 200
    plt.imshow(w_phase[:, slice_theta, :], cmap='twilight', origin='lower')
    plt.title(f"Arg(w) at t={t}, θ={slice_theta}")
    plt.colorbar(label="Phase [rad]")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"phase_w_t{t}_theta{slice_theta}.png"))
    plt.close()

    # ذخیره داده عددی فاز و گرادیان
    np.save(os.path.join(output_dir, f"arg_w_t{t}.npy"), w_phase)
    np.save(os.path.join(output_dir, f"grad_arg_w_t{t}.npy"), w_grad_chi)

    # تولید فایل خروجی متنی برای تحلیل
    with open(os.path.join(output_dir, f"phase_summary_t{t}.txt"), "w", encoding="utf-8") as f:
        f.write(f"📘 Phase Structure Summary of w(x,t) at t = {t}\n")
        f.write(f"Mean |∂χ arg(w)| = {mean_grad_chi:.4e} [rad/unit]\n\n")

        f.write("📍 Probe Points Phase Values:\n")
        for (chi, theta, phi) in probe_points:
            val = w[chi, theta, phi]
            phase = np.angle(val)
            f.write(f"  (χ,θ,φ)=({chi},{theta},{phi}): w = {val:.4e}, arg(w) = {phase:.3f} rad\n")

    print(f"✅ Done: Phase and gradient extracted for t={t}")