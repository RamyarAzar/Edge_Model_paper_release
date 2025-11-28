import numpy as np
import os
import matplotlib.pyplot as plt

# پارامترها و مسیرها
n_chi, n_theta, n_phi = 400, 400, 400
block_size = 20  # برای کاهش مصرف حافظه
w_dir = "w_output"
laplacian_dir = "laplacian_output"
dirac_dir = "dirac_output"
os.makedirs(laplacian_dir, exist_ok=True)
os.makedirs(dirac_dir, exist_ok=True)

# فاصله گام‌ها (در صورت نیاز تغییر دهید)
d_chi, d_theta, d_phi = 1.0, 1.0, 1.0

# ماتریس‌های گامای دیراکی (4×4) بدون ساده‌سازی
gamma = {
    1: np.array([[0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [0, -1, 0, 0],
                 [-1, 0, 0, 0]]),
    
    2: np.array([[0, 0, 0, -1j],
                 [0, 0, 1j, 0],
                 [0, 1j, 0, 0],
                 [-1j, 0, 0, 0]]),
    
    3: np.array([[0, 0, 1, 0],
                 [0, 0, 0, -1],
                 [-1, 0, 0, 0],
                 [0, 1, 0, 0]])
}

# زمان‌های مورد نظر
timesteps = list(range(33, 42))

for t in timesteps:
    print(f"\n🔁 Processing Dirac operator block-wise at t={t}...")

    # حافظه‌نگاشت برای w
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

    # خروجی‌ها
    dirac_magnitude = np.memmap(os.path.join(dirac_dir, f"dirac_magnitude_t{t}.npy"),
                                dtype='float64', mode='w+', shape=(n_chi, n_theta, n_phi))

    laplacian_w = np.memmap(os.path.join(laplacian_dir, f"laplacian_w_t{t}.npy"),
                            dtype='float64', mode='w+', shape=(n_chi, n_theta, n_phi))

    for chi_start in range(1, n_chi - 1, block_size):
        chi_end = min(chi_start + block_size + 2, n_chi)
        chi_slice = slice(chi_start - 1, chi_end)

        w_block = w[chi_slice, :, :]

        # مشتقات اول
        dw_dchi = np.gradient(w_block, d_chi, axis=0)
        dw_dtheta = np.gradient(w_block, d_theta, axis=1)
        dw_dphi = np.gradient(w_block, d_phi, axis=2)

        # لاپلاسیان
        d2w_dchi2 = np.gradient(dw_dchi, d_chi, axis=0)
        d2w_dtheta2 = np.gradient(dw_dtheta, d_theta, axis=1)
        d2w_dphi2 = np.gradient(dw_dphi, d_phi, axis=2)
        lap = d2w_dchi2 + d2w_dtheta2 + d2w_dphi2

        # برش قابل نوشتن
        write_slice = slice(chi_start, min(chi_end - 1, n_chi - 1))

        # ذخیره لاپلاسیان
        laplacian_w[write_slice, :, :] = lap[1:-1, :, :]

        # ساخت ψ
        ψ_block = np.zeros((4,) + w_block.shape, dtype=np.complex128)
        for α in range(4):
            ψ_block[α] = w_block + 0j

        # مشتقات اسپینور
        dψ = {
            1: np.gradient(ψ_block, d_chi, axis=1),
            2: np.gradient(ψ_block, d_theta, axis=2),
            3: np.gradient(ψ_block, d_phi, axis=3),
        }

        # محاسبه اپراتور دیراک
        dirac_block = np.zeros((4,) + w_block.shape, dtype=np.complex128)
        for μ in [1, 2, 3]:
            for α in range(4):
                acc = np.zeros(w_block.shape, dtype=np.complex128)
                for β in range(4):
                    acc += gamma[μ][α, β] * dψ[μ][β]
                dirac_block[α] += acc

        # محاسبه نُرم دیراکی
        dirac_norm = np.sqrt(np.sum(np.abs(dirac_block[:, 1:-1, :, :])**2, axis=0))
        dirac_magnitude[write_slice, :, :] = dirac_norm

    # ذخیره خلاصه‌ها
    with open(os.path.join(dirac_dir, f"dirac_summary_t{t}.txt"), "w") as f:
        f.write(f"t = {t}\n")
        f.write(f"Laplacian w   mean: {laplacian_w.mean():.3e}, std: {laplacian_w.std():.3e}\n")
        f.write(f"Dirac mean: {dirac_magnitude.mean():.3e}, max: {dirac_magnitude.max():.3e}\n")

    # پلات مقطع θ=200
    if t in [33, 35, 37, 40]:
        slice_θ = 200
        plt.figure()
        plt.imshow(laplacian_w[:, slice_θ, :], cmap='plasma')
        plt.title(f"Laplacian Slice θ={slice_θ}, t={t}")
        plt.colorbar(label="Δ²w")
        plt.savefig(os.path.join(laplacian_dir, f"laplacian_slice_t{t}.png"))
        plt.close()

        plt.figure()
        plt.imshow(dirac_magnitude[:, slice_θ, :], cmap='viridis')
        plt.title(f"Dirac Slice θ={slice_θ}, t={t}")
        plt.colorbar(label="‖Dψ‖")
        plt.savefig(os.path.join(dirac_dir, f"dirac_slice_t{t}.png"))
        plt.close()