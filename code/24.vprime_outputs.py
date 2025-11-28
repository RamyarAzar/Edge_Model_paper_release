# quantum_dv_dw_fixed.py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from tqdm import tqdm

# مسیر فایل‌ها
w_dir = "w_output"
output_dir = "vprime_outputs"
os.makedirs(output_dir, exist_ok=True)

# پارامترهای شبکه
Nx, Ny = 400, 400
dtheta = dphi = np.pi / Nx

# زمان‌های بحرانی (فقط زمان‌هایی که فایلشان موجود است)
critical_times = [t for t in [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]
                  if os.path.exists(os.path.join(w_dir, f"w_t{t}.npy"))]

for t in critical_times:
    print(f"🔁 Processing t={t}...")

    try:
        w_file = os.path.join(w_dir, f"w_t{t}.npy")
        w_data = np.memmap(w_file, dtype='float64', mode='r', shape=(400, 400, 4, 4))
        w_slice = np.copy(w_data[:, :, 2, 2])  # ثابت χ و φ

        box_w = laplace(w_slice, mode='reflect') / (dtheta ** 2)
        dVdw = -box_w

        # ذخیره‌سازی
        np.save(os.path.join(output_dir, f"boxw_t{t}.npy"), box_w.astype(np.float64))
        np.save(os.path.join(output_dir, f"dVdw_t{t}.npy"), dVdw.astype(np.float64))

        # ترسیم
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(box_w, origin='lower', cmap='RdBu_r')
        plt.colorbar()
        plt.title(f"□w at t={t}")

        plt.subplot(1, 2, 2)
        plt.imshow(dVdw, origin='lower', cmap='plasma')
        plt.colorbar()
        plt.title(f"dV/dw estimate at t={t}")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"vprime_map_t{t}.png"), dpi=150)
        plt.close()

        print(f"✅ Saved outputs for t={t}")
   
    except Exception as e:
        print(f"⛔ Error at t={t}: {e}")