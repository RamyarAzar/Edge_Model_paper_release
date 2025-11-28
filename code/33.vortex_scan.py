import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات
phase_dir = 'phase_analysis_outputs'
output_dir = 'vortex_scan_outputs'
os.makedirs(output_dir, exist_ok=True)

n_chi, n_theta, n_phi = 400, 400, 400
slice_range = range(180, 221)  # بازه مقاطع χ (قابل تنظیم)
times = list(range(33, 42))    # زمان‌های بحرانی

def compute_winding_number(grid):
    # حساب تغییر فاز حول هر سلول 2×2
    dphi_x = np.angle(np.exp(1j * (np.roll(grid, -1, axis=1) - grid)))
    dphi_y = np.angle(np.exp(1j * (np.roll(grid, -1, axis=0) - grid)))
    curl = dphi_x[:-1, :-1] + dphi_y[:-1, 1:] - dphi_x[1:, 1:] - dphi_y[1:, :-1]
    winding = np.round(curl / (2 * np.pi))
    return winding

for t in times:
    try:
        phase_file = os.path.join(phase_dir, f'phase_t{t}.npy')
        if not os.path.exists(phase_file):
            print(f"⚠️ Phase data missing for t={t}, skipping.")
            continue

        phi_data = np.load(phase_file)

        for chi in slice_range:
            phi_slice = phi_data[chi, :, :]
            winding_map = compute_winding_number(phi_slice)

            num_vortices = np.count_nonzero(winding_map)
            print(f"✅ t={t}, χ={chi} | Vortices: {num_vortices}")

            # ذخیره تصویر
            plt.figure(figsize=(6, 5))
            plt.imshow(winding_map, cmap='bwr', origin='lower', extent=[0, n_phi, 0, n_theta])
            plt.title(f'Vortex Map at χ={chi}, t={t}')
            plt.colorbar(label='Winding Number')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'vortex_map_t{t}_chi{chi}.png'))
            plt.close()

            # ذخیره داده عددی
            np.save(os.path.join(output_dir, f'winding_map_t{t}_chi{chi}.npy'), winding_map)

    except Exception as e:
        print(f"⛔ Error at t={t}, χ={chi}: {e}")