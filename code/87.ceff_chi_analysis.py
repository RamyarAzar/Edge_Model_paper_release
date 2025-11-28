import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات
t_range = range(33, 43)
n_chi, n_theta, n_phi = 400, 400, 400
output_dir = "ceff_chi_analysis"
r_dir = "r_output"
lambda_table = {
    33: 13208793.62, 34: 13239207.29, 35: 12455078.46, 36: 21449203.81,
    37: 25107541.81, 38: 25369562.2, 39: 24098134.35, 40: 27036612.69,
    41: 27040064.16, 42: 27040064.16,
}
block_size = 10
os.makedirs(output_dir, exist_ok=True)

for t in t_range:
    print(f"Processing t={t}...")
    λ = lambda_table[t]
    R_map = np.memmap(os.path.join(r_dir, f"R_t{t}.npy"), dtype=np.float32,
                      mode='r', shape=(n_chi, n_theta, n_phi))
    
    ceff_chi = np.zeros(n_chi)
    
    for chi_start in range(0, n_chi, block_size):
        chi_end = min(chi_start + block_size, n_chi)
        R_block = R_map[chi_start:chi_end, :, :]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ceff_block = λ / R_block
            ceff_block[np.isnan(ceff_block)] = 0
            ceff_block[np.isinf(ceff_block)] = 0
        
        # میانگین‌گیری روی θ و φ
        ceff_mean = np.mean(ceff_block, axis=(1, 2))
        ceff_chi[chi_start:chi_end] = ceff_mean

    # ذخیره خروجی‌ها
    np.save(os.path.join(output_dir, f"ceff_chi_t{t}.npy"), ceff_chi)
    np.savetxt(os.path.join(output_dir, f"ceff_chi_t{t}.txt"), ceff_chi)
    
    # رسم نمودار
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(n_chi), ceff_chi, label=f"t={t}")
    plt.xlabel("χ index")
    plt.ylabel("⟨c_eff⟩_θφ")
    plt.title(f"Spatial Variation of c_eff across χ (t={t})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ceff_chi_plot_t{t}.png"))
    plt.close()