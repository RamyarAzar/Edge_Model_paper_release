import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات
gauge_dir = "gauge_spatial_output"   # مسیر فایل‌های A_spatial_t{t}.npy
output_dir = "fmunu_output"
os.makedirs(output_dir, exist_ok=True)

critical_timesteps = list(range(33, 42))  # زمان‌های بحرانی t
dx = 1.0  # گام مکانی برای ∂iA_j
dt = 1.0  # گام زمانی برای ∂tA_i

# اندیس‌گذاری
coords = ['χ', 'θ', 'φ']
f_labels = []

# خروجی‌های میانگین برای هر مؤلفهٔ غیرتکراری F_{μν}
f_means = {}

# پیمایش زمان‌ها
for t in critical_timesteps[:-1]:  # چون مشتق زمانی نیاز به t+1 دارد
    print(f"⏳ Processing F_μν at t={t}...")

    A_t = np.load(os.path.join(gauge_dir, f"A_spatial_t{t}.npy"))        # shape: (3, nχ, nθ, nφ)
    A_t1 = np.load(os.path.join(gauge_dir, f"A_spatial_t{t+1}.npy"))

    # محاسبه ∂t A_i
    dA_dt = (A_t1 - A_t) / dt  # shape: (3, ...)

    # محاسبه ∂i A_j برای همه i≠j با np.gradient
    dA_dx = [np.gradient(A_t[j], dx, axis=i) for j in range(3) for i in range(3)]

    # ایجاد آرایه 4 بعدی برای F_{μν}: shape = (4, 4, nχ, nθ, nφ)
    shape = A_t.shape[1:]  # (nχ, nθ, nφ)
    F = np.zeros((4, 4) + shape, dtype=np.float64)

    # پر کردن مؤلفه‌های F_{0i} و F_{i0}
    for i in range(3):
        F[0, i+1] = dA_dt[i]
        F[i+1, 0] = -dA_dt[i]
        label = f"F0{coords[i]}"
        val = dA_dt[i]
        f_labels.append(label)
        f_means[label] = f_means.get(label, []) + [np.mean(val)]

    # پر کردن مؤلفه‌های فضایی F_{ij}
    for i in range(3):
        for j in range(i+1, 3):
            dAi_j = dA_dx[j * 3 + i]  # ∂i A_j
            dAj_i = dA_dx[i * 3 + j]  # ∂j A_i
            fij = dAi_j - dAj_i
            F[i+1, j+1] = fij
            F[j+1, i+1] = -fij
            label = f"F{coords[i]}{coords[j]}"
            f_labels.append(label)
            f_means[label] = f_means.get(label, []) + [np.mean(fij)]

    # ذخیره F برای این زمان
    np.save(os.path.join(output_dir, f"f_t{t}.npy"), F)

# ذخیره فایل میانگین‌ها به صورت عددی
np.save(os.path.join(output_dir, "f_means.npy"), f_means)

# ذخیره به صورت متنی
with open(os.path.join(output_dir, "f_summary.txt"), "w", encoding="utf-8") as f:
    f.write("⛲ Gauge Field Strength Tensor F_μν(x,t) (Mean over space)\n\n")
    for label in f_labels:
        values = f_means[label]
        for i, t in enumerate(critical_timesteps[:-1]):
            f.write(f"t={t:2d} | {label} = {values[i]:+.3e}\n")
        f.write("\n")

# رسم نمودار برای هر مؤلفه F_μν
for label in f_labels:
    plt.plot(critical_timesteps[:-1], f_means[label], marker='o', label=label)

plt.xlabel("Critical Time t")
plt.ylabel("⟨F_μν⟩")
plt.title("Mean Field Strength Tensor Components ⟨F_μν(t)⟩")
plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fmunu_plot.png"))
plt.close()