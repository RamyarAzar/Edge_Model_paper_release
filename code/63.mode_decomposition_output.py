import numpy as np
import os
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# پارامترها
w_dir = "w_output"
output_dir = "mode_decomposition_output"
os.makedirs(output_dir, exist_ok=True)

critical_timesteps = list(range(33, 42))
n_chi, n_theta, n_phi = 400, 400, 400
n_modes = 10  # تعداد مدهای اصلی مورد نظر

# خروجی‌ها
explained_variance_ratios = []

for t in critical_timesteps:
    print(f"⏳ Decomposing w_t{t}.npy ...")

    # بارگذاری داده
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

    # صاف‌سازی عددی
    w_smooth = gaussian_filter(w, sigma=1.0)

    # بازشکل‌دهی برای PCA
    w_flat = w_smooth.reshape((n_chi * n_theta, n_phi))

    # اعمال PCA
    pca = PCA(n_components=n_modes)
    w_pca = pca.fit_transform(w_flat)

    # ذخیره خروجی‌ها
    np.save(os.path.join(output_dir, f"w_modes_t{t}.npy"), pca.components_)  # ماتریس مدها
    np.save(os.path.join(output_dir, f"w_coeffs_t{t}.npy"), w_pca)           # ضرایب هر مد
    explained_variance_ratios.append(pca.explained_variance_ratio_)

# ذخیره واریانس‌ها
np.save(os.path.join(output_dir, "explained_variance_ratios.npy"), np.array(explained_variance_ratios))

# رسم نمودار واریانس توضیح داده‌شده
mean_ratios = np.mean(explained_variance_ratios, axis=0)
plt.figure()
plt.bar(range(1, n_modes + 1), mean_ratios)
plt.xlabel("Mode Index")
plt.ylabel("Mean Explained Variance")
plt.title("Spectral Decomposition of w(x,t)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mode_variance_plot.png"))
plt.close()

# ذخیره به صورت فایل متنی
with open(os.path.join(output_dir, "mode_variance_summary.txt"), "w", encoding="utf-8") as f:
    for i, t in enumerate(critical_timesteps):
        f.write(f"t={t}: " + ", ".join([f"Mode {j+1} = {r:.4f}" for j, r in enumerate(explained_variance_ratios[i])]) + "\n")
