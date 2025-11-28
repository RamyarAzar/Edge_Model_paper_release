import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# 📁 مسیر فایل‌ها
lambda_dir = "lambda_analysis_phase315"
topo_dir = "topology_analysis"
timesteps = np.arange(33, 42)
dt = 1.0  # گام زمانی بین نقاط بحرانی، در صورت نیاز تغییر کن

# 📥 بارگذاری داده‌ی λ(t)
lambda_path = os.path.join(lambda_dir, "lambda_normalization.npy")
lambda_t = np.load(lambda_path)

# ✅ بازسازی زمان t با انتگرال‌گیری معکوس λ(t)
# t_recon[i] = ∫₀^i dt / λ(t)
t_recon = cumulative_trapezoid(1.0 / lambda_t, dx=dt, initial=0.0)

# 💾 ذخیره خروجی
np.save(os.path.join(lambda_dir, "t_reconstructed.npy"), t_recon)

# 📊 رسم نمودار بازسازی زمان
plt.figure(figsize=(8, 4))
plt.plot(timesteps, t_recon, color="purple", marker="o", linewidth=2)
plt.title("Reconstructed Time from λ(t)")
plt.xlabel("Original Time Step")
plt.ylabel("Reconstructed Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(lambda_dir, "time_reconstruction_plot.png"))
plt.show()