import numpy as np
import os
import matplotlib.pyplot as plt

# پارامترها و مسیرها
w_dir = "w_output"
veff_dir = "veff_output"
meff_dir = "meff_output"
r4_dir = "ricci_output"
output_dir = "dyn_eq_with_R_output"
os.makedirs(output_dir, exist_ok=True)

t_range = list(range(33, 42))
n_chi, n_theta, n_phi = 400, 400, 400
chi_block_size = 10
dx = 1.0  # مش‌بندی یکنواخت
xi = 1.0  # کوپلینگ انحنای فضازمان

# توابع عددی
def central_derivative(f_m1, f_0, f_p1):
    return (f_p1 - f_m1) / 2.0

def spatial_laplacian(f):
    lap = sum(np.gradient(np.gradient(f, dx, axis=i), dx, axis=i) for i in range(3))
    return lap

def log_safe_product(a, b):
    return np.exp(np.log(np.abs(a) + 1e-300) + np.log(np.abs(b) + 1e-300)) * np.sign(a) * np.sign(b)

# اجرای کد برای هر t
for t in t_range[1:-1]:  # نیاز به t-1 و t+1
    print(f"\n🚀 Processing dynamic equation with curvature at t={t}...")

    # بارگذاری داده‌ها با memmap
    def load_w(t_idx):
        return np.memmap(os.path.join(w_dir, f"w_t{t_idx}.npy"), dtype='float64', mode='r',
                         shape=(n_chi, n_theta, n_phi))
    
    w_m1 = load_w(t - 1)
    w_0 = load_w(t)
    w_p1 = load_w(t + 1)

    veff = np.memmap(os.path.join(veff_dir, f"veff_t{t}.npy"), dtype='float64', mode='r',
                     shape=(n_chi, n_theta, n_phi))
    
    meff = np.memmap(os.path.join(meff_dir, f"meff_t{t}.npy"), dtype='float64', mode='r',
                     shape=(n_chi, n_theta, n_phi))

    r4 = np.memmap(os.path.join(r4_dir, f"Rscalar_t{t}.npy"), dtype='float32', mode='r',
                   shape=(n_chi, n_theta, n_phi))

    rhs_mean = []

    for chi_start in range(0, n_chi, chi_block_size):
        chi_end = min(chi_start + chi_block_size, n_chi)
        print(f"  📦 Chi block: {chi_start}-{chi_end}")

        # استخراج بلوک
        w_b = w_0[chi_start:chi_end]
        w_m1_b = w_m1[chi_start:chi_end]
        w_p1_b = w_p1[chi_start:chi_end]
        veff_b = veff[chi_start:chi_end]
        meff_b = meff[chi_start:chi_end]
        r4_b = r4[chi_start:chi_end]

        # مشتق‌ها
        d2w_dt2 = (w_p1_b - 2 * w_b + w_m1_b)
        lap = spatial_laplacian(w_b)

        # محاسبه RHS کامل (هر ترم با log-safe)
        rhs = -d2w_dt2 \
              + log_safe_product(meff_b**2, w_b) \
              + log_safe_product(np.ones_like(w_b), np.gradient(veff_b, dx, axis=0)) \
              + log_safe_product(xi * r4_b, w_b) \
              - lap

        rhs_mean.append(np.mean(rhs))

        # ذخیره خروجی‌های این بلوک
        np.save(os.path.join(output_dir, f"dyn_eq_rhs_t{t}_chi{chi_start}.npy"), rhs)

        # ذخیره txt و نمودار هیستوگرام فقط برای بلوک اول
        if chi_start == 0:
            flat = rhs.flatten()
            with open(os.path.join(output_dir, f"rhs_t{t}.txt"), "w") as f:
                for val in flat:
                    f.write(f"{val:.6e}\n")

            plt.hist(flat, bins=100, log=True)
            plt.title(f"RHS Histogram at t={t}")
            plt.xlabel("RHS values")
            plt.ylabel("Count")
            plt.savefig(os.path.join(output_dir, f"rhs_hist_t{t}.png"))
            plt.clf()

    print(f"✅ Mean RHS at t={t}: {np.mean(rhs_mean):.4e}")