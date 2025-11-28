import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات
w_dir = "w_output"
laplacian_dir = "laplacian_output"
hbar_real_file = "hbar_real_t.npy"
hbar_eff_file = "hbar_eff_t.npy"
output_dir = "meff_output"
os.makedirs(output_dir, exist_ok=True)

timesteps = list(range(33, 42 + 1))
n_chi, n_theta, n_phi = 400, 400, 400
d_chi, d_theta, d_phi = 1.0, 1.0, 1.0  # فرض: فاصله شبکه در هر جهت

# بارگذاری hbarها
hbar_real_t = np.load(hbar_real_file)
hbar_eff_t = np.load(hbar_eff_file)

for t in timesteps:
    print(f"⏳ Processing m_eff at t={t}...")
    t_index = t - 33
    hbar = hbar_real_t[t_index]  # J.s
    hbar_eff = hbar_eff_t[t_index]  # dimensionless

    # خواندن w
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

    # محاسبه گرادیان w
    grad_w_chi = np.gradient(w, d_chi, axis=0)
    grad_w_theta = np.gradient(w, d_theta, axis=1)
    grad_w_phi = np.gradient(w, d_phi, axis=2)
    grad_sq = grad_w_chi**2 + grad_w_theta**2 + grad_w_phi**2  # |∇w|²

    # بارگذاری لاپلاسیان
    laplacian_path = os.path.join(laplacian_dir, f"laplacian_w_t{t}.npy")
    Δw = np.memmap(laplacian_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

    # محاسبه m_eff(x,t)
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = - (hbar**2) * Δw
        denominator = hbar_eff * grad_sq
        meff = np.where(denominator != 0, numerator / denominator, 0)

    # ذخیره خروجی
    np.save(os.path.join(output_dir, f"meff_t{t}.npy"), meff)

    # خلاصه آماری
    mean_val = np.mean(meff)
    std_val = np.std(meff)
    min_val = np.min(meff)
    max_val = np.max(meff)

    with open(os.path.join(output_dir, f"meff_summary_t{t}.txt"), 'w') as f:
        f.write(f"Mean: {mean_val:.5e}\n")
        f.write(f"Std : {std_val:.5e}\n")
        f.write(f"Min : {min_val:.5e}\n")
        f.write(f"Max : {max_val:.5e}\n")

    # رسم و ذخیره تصویر مقطع
    center_slice = meff[:, :, n_phi // 2]
    plt.imshow(center_slice, cmap='inferno', origin='lower')
    plt.colorbar(label='m_eff')
    plt.title(f"m_eff slice at t={t}")
    plt.savefig(os.path.join(output_dir, f"meff_slice_t{t}.png"))
    plt.clf()

print("✅ محاسبه m_eff برای همه زمان‌ها انجام شد.")