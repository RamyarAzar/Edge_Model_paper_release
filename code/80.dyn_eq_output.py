import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات
t_range = list(range(33, 43))
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
chi_block_size = 10
dx = 1.0

# مسیرها
w_dir = "w_output"
veff_dir = "veff_output"
meff_dir = "meff_output"
k_dir = "k_output"
r4_dir = "ricci_output"
output_dir = "dyn_eq_output"
os.makedirs(output_dir, exist_ok=True)

# ابزارهای مشتق‌گیری
def central_derivative(f_m1, f_0, f_p1):
    return (f_p1 - f_m1) / 2.0

def log_safe_square(x):
    return np.exp(2 * np.log(np.abs(x) + 1e-300))

def log_safe_product(a, b):
    return np.exp(np.log(np.abs(a) + 1e-300) + np.log(np.abs(b) + 1e-300))

for t in t_range[1:-1]:
    print(f"\n🔧 Processing dynamics at t={t}...")

    # بارگذاری بلوکی داده‌ها
    w_m1 = np.memmap(os.path.join(w_dir, f"w_t{t-1}.npy"), dtype='float64', mode='r',
                     shape=(n_chi, n_theta, n_phi))
    w_0 = np.memmap(os.path.join(w_dir, f"w_t{t}.npy"), dtype='float64', mode='r',
                    shape=(n_chi, n_theta, n_phi))
    w_p1 = np.memmap(os.path.join(w_dir, f"w_t{t+1}.npy"), dtype='float64', mode='r',
                     shape=(n_chi, n_theta, n_phi))
    veff = np.memmap(os.path.join(veff_dir, f"veff_t{t}.npy"), dtype='float64', mode='r',
                     shape=(n_chi, n_theta, n_phi))
    meff = np.memmap(os.path.join(meff_dir, f"meff_t{t}.npy"), dtype='float64', mode='r',
                     shape=(n_chi, n_theta, n_phi))
    r4 = np.memmap(os.path.join(r4_dir, f"Rscalar_t{t}.npy"), dtype='float32', mode='r',
                   shape=(n_chi, n_theta, n_phi))
    K = np.memmap(os.path.join(k_dir, f"K_t{t}.npy"), dtype='float32', mode='r',
                  shape=(n_coords, n_coords, n_chi, n_theta, n_phi))

    lambda_t = 1.0
    xi_t = 1.0
    beta_t = 1.0

    for chi_start in range(0, n_chi, chi_block_size):
        chi_end = min(chi_start + chi_block_size, n_chi)
        print(f"  ▶ Chi block {chi_start}-{chi_end}")

        w_b = w_0[chi_start:chi_end]
        w_m1_b = w_m1[chi_start:chi_end]
        w_p1_b = w_p1[chi_start:chi_end]
        veff_b = veff[chi_start:chi_end]
        meff_b = meff[chi_start:chi_end]
        r4_b = r4[chi_start:chi_end]
        K_b = K[:, :, chi_start:chi_end]

        # مشتق زمان دوم و لاپلاس
        d2w_dt = (w_p1_b - 2*w_b + w_m1_b) / dx**2
        lap_w = sum(np.gradient(np.gradient(w_b, dx, axis=i), dx, axis=i) for i in range(3))
        box_w = -d2w_dt + lap_w

        # K^{μν} * w
        K_sum = np.zeros_like(w_b)
        for mu in range(n_coords):
            for nu in range(n_coords):
                K_sum += beta_t * K_b[mu, nu]  # ضرب اسکالر در فضای تانسوری

        # طرف چپ و راست معادله
        lhs = (1 - 0.5 * lambda_t**2) * box_w
        rhs = -log_safe_square(meff_b) * w_b \
              - np.gradient(veff_b, dx, axis=0) \
              + log_safe_product(xi_t * r4_b, w_b) \
              + log_safe_product(K_sum, w_b)

        residual = lhs - rhs

        # ذخیره خروجی‌ها
        block_id = f"t{t}_chi{chi_start}"
        np.save(os.path.join(output_dir, f"dyn_eq_res_{block_id}.npy"), residual)

        with open(os.path.join(output_dir, f"dyn_eq_res_{block_id}.txt"), "w") as f:
            for val in residual.flatten():
                f.write(f"{val:.6e}\n")

        plt.hist(residual.flatten(), bins=100, log=True)
        plt.title(f"Dynamics Residual at {block_id}")
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, f"dyn_eq_hist_{block_id}.png"))
        plt.clf()

    print(f"✅ Done with t = {t}")