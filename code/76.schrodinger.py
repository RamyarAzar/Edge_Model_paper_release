import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات
time_steps = list(range(33, 43))
n_chi, n_theta, n_phi = 400, 400, 400
d_chi = d_theta = d_phi = 1.0  # یا مقدار صحیح هندسی
m = 1.0  # جرم واحد (بدون ساده‌سازی هنوز قابل تعویض است)
alpha = 2.213958e-102  # ضریب مقیاس از مدل

# مسیرها
w_dir = "w_output"
veff_dir = "veff_output"
hbar_file = "hbar_real_t.npy"
output_dir = "schrodinger_output"
os.makedirs(output_dir, exist_ok=True)

# بارگذاری hbar(t) مقیاس‌یافته
hbar_series = np.load(hbar_file)

for t in time_steps:
    print(f"⏳ Processing Schrödinger equation at t={t}")
    
    # بارگذاری w با memmap
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))
    
    # بارگذاری veff
    veff_path = os.path.join(veff_dir, f"veff_t{t}.npy")
    veff = np.load(veff_path)

    # ℏ واقعی برای این گام زمانی
    hbar = hbar_series[t - 33]

    # محاسبه لاپلاسیان (با فرض تقریب مرکزی)
    laplacian_w = (
        np.gradient(np.gradient(w, d_chi, axis=0), d_chi, axis=0) +
        np.gradient(np.gradient(w, d_theta, axis=1), d_theta, axis=1) +
        np.gradient(np.gradient(w, d_phi, axis=2), d_phi, axis=2)
    )

    # محاسبه طرف راست معادله‌ی شرودینگر
    RHS = -(hbar**2 / (2 * m)) * laplacian_w + veff * w

    # ذخیره خروجی‌ها
    np.save(os.path.join(output_dir, f"schrodinger_rhs_t{t}.npy"), RHS)

    with open(os.path.join(output_dir, f"schrodinger_rhs_t{t}.txt"), 'w') as f:
        stats = {
            "mean": np.mean(RHS),
            "std": np.std(RHS),
            "min": np.min(RHS),
            "max": np.max(RHS)
        }
        for key, val in stats.items():
            f.write(f"{key}: {val:.5e}\n")

    # نمودار مقطع θ=200
    slice_plot = np.abs(RHS[:, 200, :])
    plt.figure(figsize=(6, 5))
    plt.imshow(slice_plot, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label="|RHS|")
    plt.title(f"Schrödinger RHS Slice θ=200, t={t}")
    plt.xlabel("φ index")
    plt.ylabel("χ index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"schrodinger_rhs_slice_t{t}.png"))
    plt.close()

print("✅ Schrödinger RHS computed for all time steps.")