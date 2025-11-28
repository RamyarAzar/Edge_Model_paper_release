import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات مسیرها
f_dir = "fmunu_output"         # شامل فایل‌های f_t{t}.npy
g_dir = "metric"        # شامل فایل‌های g_t{t}.npy
output_dir = "energy_output"
os.makedirs(output_dir, exist_ok=True)

# ابعاد شبکه و زمان
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
block_size = 10
critical_timesteps = list(range(33, 41))
eps = 1e-30  # جلوگیری از log(0)

# تابع تقسیم دامنه به بلوک‌ها
def block_ranges(N, block_size):
    for i in range(0, N, block_size):
        yield slice(i, min(i + block_size, N))

# نتایج برای خروجی نمودار
log_energy_means = []

for t in critical_timesteps:
    print(f"⏳ Computing Tr(FμνFμν) at t={t}...")

    # حافظه‌نگاشت برای میدان و متریک
    F = np.load(os.path.join(f_dir, f"f_t{t}.npy"), mmap_mode='r')
    G = np.load(os.path.join(g_dir, f"g_t{t}.npy"), mmap_mode='r')

    # متریک وارون به صورت بلوکی محاسبه می‌شود
    log_energy_total = 0.0
    count = 0

    for chi_range in block_ranges(n_chi, block_size):
        for theta_range in block_ranges(n_theta, block_size):
            for phi_range in block_ranges(n_phi, block_size):
                # استخراج بلوک‌های مکانی
                f_block = F[:, :, chi_range, theta_range, phi_range]
                g_block = G[:, :, chi_range, theta_range, phi_range]

                # وارون متریک برای هر نقطه در بلوک
                shape = f_block.shape[2:]  # spatial block size
                g_inv_block = np.zeros_like(g_block)
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(shape[2]):
                            g_point = g_block[:, :, i, j, k]
                            try:
                                g_inv_block[:, :, i, j, k] = np.linalg.inv(g_point)
                            except np.linalg.LinAlgError:
                                g_inv_block[:, :, i, j, k] = np.zeros((n_coords, n_coords))

                # محاسبه log(term) در هر نقطه
                log_block = np.zeros(shape, dtype=np.float64)

                for mu in range(n_coords):
                    for nu in range(n_coords):
                        for alpha in range(n_coords):
                            for beta in range(n_coords):
                                f1 = f_block[mu, nu, :, :, :]
                                f2 = f_block[alpha, beta, :, :, :]
                                g1 = g_inv_block[mu, alpha, :, :, :]
                                g2 = g_inv_block[nu, beta, :, :, :]

                                log_term = (
                                    np.log(np.abs(f1) + eps) +
                                    np.log(np.abs(f2) + eps) +
                                    np.log(np.abs(g1) + eps) +
                                    np.log(np.abs(g2) + eps)
                                )
                                log_block += log_term

                # میانگین لگاریتمی انرژی در این بلوک
                log_energy_total += np.sum(log_block)
                count += np.prod(shape)

    # میانگین نهایی لگاریتم انرژی در t
    log_energy_mean = log_energy_total / count
    log_energy_means.append(log_energy_mean)

    # ذخیره به صورت npy
    np.save(os.path.join(output_dir, f"log_energy_t{t}.npy"), log_block)

# ذخیره میانگین‌ها
np.save(os.path.join(output_dir, "log_energy_means.npy"), np.array(log_energy_means))

# ذخیره به صورت متنی
with open(os.path.join(output_dir, "log_energy_summary.txt"), "w", encoding="utf-8") as f:
    f.write("⚡ Log-Averaged Energy Tr(FμνFμν) per timestep\n\n")
    for i, t in enumerate(critical_timesteps):
        f.write(f"t={t} | ⟨log_energy⟩ = {log_energy_means[i]:+.5f}\n")

# رسم نمودار
plt.plot(critical_timesteps, log_energy_means, marker='o', label="⟨log Tr(FμνFμν)⟩")
plt.xlabel("Time t")
plt.ylabel("Log-Averaged Energy")
plt.title("Mean Log Energy of Gauge Field over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "log_energy_plot.png"))
plt.close()