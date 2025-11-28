# Ervin: Quantum DMH — Numerical Scaling to Real Planck Constant (Step 2.8.1)
import numpy as np
import os

# مسیر فایل‌ها
hbar_eff_path = os.path.join("lambda_analysis_phase315", "hbar_eff_t.npy")

# مقدار واقعی پلانک (Joule·second)
hbar_real = 1.0545718e-34

# بارگذاری داده‌های هبار مؤثر
hbar_eff = np.load(hbar_eff_path)  # (length = 10 or more)

# روش ۱: استفاده از میانگین
hbar_eff_avg = np.mean(hbar_eff)
alpha = hbar_real / hbar_eff_avg

# تعریف نسخه عددی از هبار مؤثر
hbar_real_series = alpha * hbar_eff

# ذخیره خروجی‌ها برای تحلیل‌های بعدی
np.save("lambda_analysis_phase315/hbar_real_t.npy", hbar_real_series)

with open("lambda_analysis_phase315/hbar_scaling_summary.txt", "w", encoding="utf-8") as f:
    f.write("📐 Quantum DMH Numerical Scaling to ℏ (Planck Constant)\n")
    f.write(f"Average ℏ_eff(t) from model: {hbar_eff_avg:.6e} [dimensionless]\n")
    f.write(f"Real ℏ value (SI): {hbar_real:.6e} J·s\n")
    f.write(f"Derived scale factor α = ℏ_real / ⟨ℏ_eff⟩ = {alpha:.6e} J·s\n")
    f.write(f"Realistic ℏ(t) series saved to: hbar_real_t.npy\n")