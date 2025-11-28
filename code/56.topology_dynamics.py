import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import entropy

# تنظیمات اولیه
topo_dir = "topology_analysis"
output_dir = "topology_dynamics"
os.makedirs(output_dir, exist_ok=True)

timesteps = [33, 34, 35, 36, 37, 38, 39, 40, 41]

# لیست‌ها برای ذخیره خروجی‌ها
entropy_list = []
autocorr_list = []
topo_fields = []

# بارگذاری میدان انرژی توپولوژیک و محاسبه آنتروپی + ذخیره heatmap
for t in timesteps:
    path = os.path.join(topo_dir, f"topo_energy_t{t}.npy")
    topo = np.load(path)
    topo_fields.append(topo)

    # normalize to probability for entropy
    flat = topo.flatten()
    flat = flat - np.min(flat)
    if np.sum(flat) == 0:
        ent = 0
    else:
        prob = flat / np.sum(flat)
        ent = entropy(prob, base=2)
    entropy_list.append(ent)

    # ترسیم برش میانی φ
    if topo.ndim == 3:
        mid = topo.shape[2] // 2
        plt.imshow(topo[:, :, mid], cmap='plasma')
        plt.title(f"Topo Energy Slice (t={t})")
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f"topo_heatmap_t{t}.png"))
        plt.close()

# محاسبه auto-correlation بین فریم‌های متوالی
for i in range(len(topo_fields) - 1):
    a = topo_fields[i].flatten()
    b = topo_fields[i + 1].flatten()
    # normalize
    a = (a - np.mean(a)) / np.std(a)
    b = (b - np.mean(b)) / np.std(b)
    corr = np.corrcoef(a, b)[0, 1]
    autocorr_list.append(corr)

# ذخیره فایل‌های خروجی
np.save(os.path.join(output_dir, "topo_entropy_t.npy"), np.array(entropy_list))
np.save(os.path.join(output_dir, "topo_autocorr_t.npy"), np.array(autocorr_list))

# ذخیره به فایل متنی
with open(os.path.join(output_dir, "topology_temporal_stats.txt"), "w", encoding="utf-8") as f:
    for i, t in enumerate(timesteps):
        f.write(f"t={t}: Entropy = {entropy_list[i]:.5f}\n")
    f.write("\n🔁 Auto-correlation:\n")
    for i in range(len(autocorr_list)):
        f.write(f"t={timesteps[i]} → t={timesteps[i+1]}: Corr = {autocorr_list[i]:.4f}\n")

# نمودار آنتروپی
plt.plot(timesteps, entropy_list, marker='o')
plt.title("Spatial Entropy of Topological Energy")
plt.xlabel("Time step")
plt.ylabel("Entropy (bits)")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "entropy_plot.png"))
plt.close()

# نمودار همبستگی زمانی
plt.plot(timesteps[:-1], autocorr_list, marker='s')
plt.title("Temporal Auto-correlation of Topo Energy")
plt.xlabel("t → t+1")
plt.ylabel("Correlation")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "autocorr_plot.png"))
plt.close()