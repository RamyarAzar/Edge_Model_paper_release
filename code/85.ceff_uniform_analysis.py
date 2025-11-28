import numpy as np
import os
import matplotlib.pyplot as plt

# مسیر فایل‌ها
ceff_dir = "ceff_chi_analysis"
output_dir = "ceff_uniform_analysis"
os.makedirs(output_dir, exist_ok=True)

t_range = range(33, 43)
n_chi = 400
chi_block = 10
threshold_fraction = 0.005  # آستانه سخت‌تر: std < 0.5% mean

for t in t_range:
    print(f"\n🔍 Analyzing ceff uniformity at t={t}...")

    # بارگذاری فایل 1-بعدی
    ceff = np.load(os.path.join(ceff_dir, f"ceff_chi_t{t}.npy"), allow_pickle=False)
    if ceff.ndim != 1 or len(ceff) != n_chi:
        raise ValueError(f"Invalid ceff shape at t={t}: expected (400,), got {ceff.shape}")

    stable_chi = []
    block_means = []
    block_stds = []

    for chi_start in range(0, n_chi, chi_block):
        chi_end = min(chi_start + chi_block, n_chi)
        block = ceff[chi_start:chi_end]
        mu = np.mean(block)
        sigma = np.std(block)
        block_means.append(mu)
        block_stds.append(sigma)

        if mu > 0 and sigma < threshold_fraction * mu:
            stable_chi.extend(range(chi_start, chi_end))

    # ذخیره نمودار
    plt.plot(range(0, n_chi, chi_block), block_means, label="Mean $c_{eff}$")
    plt.plot(range(0, n_chi, chi_block), block_stds, label="Std Dev", linestyle='--')
    plt.title(f"$c_{{eff}}$ Uniformity Across $\\chi$ (t={t})")
    plt.xlabel("Chi Index (start of block)")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ceff_uniform_t{t}.png"))
    plt.clf()

    # ذخیره گزارش متنی
    with open(os.path.join(output_dir, f"uniform_chi_t{t}.txt"), 'w') as f:
        f.write(f"Time step = {t}\n")
        f.write(f"Stable chi slices (std < {threshold_fraction*100:.2f}% of mean):\n")
        f.write(', '.join(str(i) for i in stable_chi) + "\n")
        f.write(f"Total stable chi slices: {len(stable_chi)}\n")

print("\n✅ Uniformity analysis completed.")