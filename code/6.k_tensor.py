import numpy as np
import os

# تنظیمات
critical_timesteps = [1, 2, 3, 9, 10, 11, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
chi_block_size = 10

embedding_dir = "embedding_output"
output_dir = "k_tensor_output"
os.makedirs(output_dir, exist_ok=True)

for t in critical_timesteps:
    print(f"⏳ Computing K_μν at t={t}...")

    # بارگذاری بردار نرمال
    n_vec = np.load(os.path.join(embedding_dir, f"normal_n_t{t}.npy")).astype(np.float32)  # [5, χ, θ, φ]

    # ساخت آرایه K به صورت memmap برای نوشتن تدریجی
    K_path = os.path.join(output_dir, f"K_t{t}.npy")
    K = np.memmap(K_path, dtype=np.float32, mode='w+', shape=(n_coords, n_coords, n_chi, n_theta, n_phi))

    # پردازش بلوکی روی χ
    for i_start in range(0, n_chi, chi_block_size):
        i_end = min(i_start + chi_block_size, n_chi)
        block_shape = (5, n_coords, n_coords, i_end - i_start, n_theta, n_phi)

        d2x_filename = os.path.join(embedding_dir, f"D2X_t{t}_block_{i_start}_{i_end}.npy")
        if not os.path.exists(d2x_filename):
            print(f"⚠️ Skipping missing block: {d2x_filename}")
            continue

        D2X_block = np.load(d2x_filename).reshape(block_shape).astype(np.float32)

        # نرمال بلاک
        n_block = n_vec[:, i_start:i_end, :, :]  # [5, χ_blk, θ, φ]

        for mu in range(n_coords):
            for nu in range(n_coords):
                # K_{μν} = - n_A ∂_μ ∂_ν X^A
                dot = np.einsum('aijk,aijk->ijk', n_block, D2X_block[:, mu, nu])
                dot = np.nan_to_num(dot, nan=0.0, posinf=0.0, neginf=0.0)  # پایداری عددی
                K[mu, nu, i_start:i_end] = -dot

    print(f"✅ K_t{t}.npy saved.")