import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات
w_dir = "w_output"
veff_dir = "veff_output"
output_dir = "tmunu_final"
os.makedirs(output_dir, exist_ok=True)

t_steps = list(range(33, 42))
chi_block_size = 10
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
dx = 1.0  # فرض مش بندی یکنواخت

# توابع مشتق‌گیری و لگاریتمی
def central_derivative(f_m1, f_0, f_p1):
    return (f_p1 - f_m1) / 2.0

def spatial_gradient(f, axis):
    return np.gradient(f, dx, axis=axis)

def spatial_second_derivative(f, axis):
    return np.gradient(np.gradient(f, dx, axis=axis), dx, axis=axis)

def log_safe_product(a, b):
    return np.exp(np.log(np.abs(a) + 1e-300) + np.log(np.abs(b) + 1e-300))

# پردازش هر t
for t in t_steps[1:-1]:  # نیاز به t-1 و t+1
    print(f"\n📦 Processing T_mu_nu at t={t}...")

    # بارگذاری میدان‌های w و V_eff
    def load_w(t_idx):
        return np.memmap(os.path.join(w_dir, f"w_t{t_idx}.npy"), dtype='float64', mode='r',
                         shape=(n_chi, n_theta, n_phi))
    w_m1 = load_w(t - 1)
    w_0 = load_w(t)
    w_p1 = load_w(t + 1)

    veff = np.memmap(os.path.join(veff_dir, f"veff_t{t}.npy"), dtype='float64', mode='r',
                     shape=(n_chi, n_theta, n_phi))

    # آماده‌سازی خروجی
    tmunu_mean = np.zeros((n_coords, n_coords))

    for chi_start in range(0, n_chi, chi_block_size):
        chi_end = min(chi_start + chi_block_size, n_chi)
        print(f"  ⏳ chi {chi_start}-{chi_end}")

        # استخراج بلوک
        w_block = w_0[chi_start:chi_end, :, :]
        w_m1_block = w_m1[chi_start:chi_end, :, :]
        w_p1_block = w_p1[chi_start:chi_end, :, :]
        veff_block = veff[chi_start:chi_end, :, :]

        # مشتق زمانی و گرادیان‌های فضایی
        dw_dt = central_derivative(w_m1_block, w_block, w_p1_block)
        grads = [spatial_gradient(w_block, axis=i) for i in range(3)]
        grad_sq = sum(g**2 for g in grads)

        # محاسبه ترم اول: انرژی اسکارلار
        term_energy = grad_sq + dw_dt**2 + veff_block

        # متریک مینکوفسکی g_{μν}
        g = np.zeros((n_coords, n_coords))
        g[0, 0] = -1.0
        for i in range(1, n_coords):
            g[i, i] = 1.0

        # محاسبه مشتقات دوم w
        second_derivs = np.zeros((n_coords, n_coords, chi_end - chi_start, n_theta, n_phi))
        for mu in range(n_coords):
            for nu in range(n_coords):
                if mu == 0 and nu == 0:
                    d2w = central_derivative(w_m1_block, w_block, w_p1_block)
                    d2w = central_derivative(w_m1_block, w_block, w_p1_block)
                elif mu == 0 or nu == 0:
                    d2w = dw_dt  # تقریب ترکیبی
                else:
                    d2w = spatial_second_derivative(w_block, axis=mu - 1)
                second_derivs[mu, nu] = d2w

        # ساختار نهایی T_{μν}
        T_block = np.zeros((n_coords, n_coords, chi_end - chi_start, n_theta, n_phi))
        for mu in range(n_coords):
            for nu in range(n_coords):
                A = g[mu, nu] * term_energy
                B = 0.5 * log_safe_product(w_block, second_derivs[mu, nu])
                T_block[mu, nu] = A - B
                tmunu_mean[mu, nu] += np.mean(T_block[mu, nu])

                # ذخیره‌ی نمودار و txt برای T_{00}
                if mu == 0 and nu == 0 and chi_start == 0:
                    flat = T_block[mu, nu].flatten()
                    with open(os.path.join(output_dir, f"T00_t{t}.txt"), "w") as f:
                        for val in flat:
                            f.write(f"{val:.6e}\n")
                    plt.hist(flat, bins=100, log=True)
                    plt.title(f"T00 Histogram at t={t}")
                    plt.xlabel("T00 values")
                    plt.ylabel("Count")
                    plt.savefig(os.path.join(output_dir, f"T00_hist_t{t}.png"))
                    plt.clf()

        # ذخیره‌ی npy بلوکی
        np.save(os.path.join(output_dir, f"T_block_t{t}_chi{chi_start}.npy"), T_block)

    print(f"✅ Mean Tμν at t={t}:\n", tmunu_mean / (n_chi / chi_block_size))