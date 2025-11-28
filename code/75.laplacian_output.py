import os
import numpy as np
from scipy.ndimage import laplace
import matplotlib.pyplot as plt

# تنظیمات
timesteps = list(range(33, 42))
n_chi, n_theta, n_phi = 400, 400, 400
input_dir = "w_output"
output_base = "laplacian_output"
os.makedirs(output_base, exist_ok=True)

# مشتق عددی درجه دوم (مرکزی)
def central_diff(f, axis, dx):
    return (np.roll(f, -1, axis=axis) - 2 * f + np.roll(f, 1, axis=axis)) / (dx**2)

# اندازه‌های شبکه (در صورت نیاز قابل تغییر)
delta_chi = delta_theta = delta_phi = 1.0

for t in timesteps:
    print(f"\n⏳ Processing quantum Laplacian for t={t}...")

    try:
        w_path = os.path.join(input_dir, f"w_t{t}.npy")
        w = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

        # محاسبه لاپلاسیان
        lap_chi = central_diff(w, axis=0, dx=delta_chi)
        lap_theta = central_diff(w, axis=1, dx=delta_theta)
        lap_phi = central_diff(w, axis=2, dx=delta_phi)
        laplacian = lap_chi + lap_theta + lap_phi

        # ذخیره به صورت npy
        np.save(os.path.join(output_base, f"laplacian_t{t}.npy"), laplacian)

        # ذخیره خلاصه آماری به صورت txt
        stats = {
            "mean": np.mean(laplacian),
            "std": np.std(laplacian),
            "min": np.min(laplacian),
            "max": np.max(laplacian),
        }
        with open(os.path.join(output_base, f"laplacian_summary_t{t}.txt"), "w") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v:.5e}\n")

        # ذخیره تصویر برش میانی در θ = 200
        plt.figure(figsize=(8, 6))
        plt.imshow(laplacian[:, 200, :], cmap="viridis", origin="lower", aspect="auto")
        plt.colorbar(label="Δ² w")
        plt.title(f"Laplacian Slice at θ=200, t={t}")
        plt.xlabel("ϕ")
        plt.ylabel("χ")
        plt.tight_layout()
        plt.savefig(os.path.join(output_base, f"laplacian_slice_t{t}.png"))
        plt.close()

    except FileNotFoundError:
        print(f"⚠️ w_t{t}.npy not found, skipping...")