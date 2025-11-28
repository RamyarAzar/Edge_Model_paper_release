import numpy as np
import os

# پارامترها
critical_timesteps = [1, 2, 3, 9, 10, 11, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
n_chi, n_theta, n_phi = 400, 400, 400
dx = [1e-5, np.pi / n_chi, np.pi / n_theta, 2 * np.pi / n_phi]
chi = np.linspace(0, np.pi, n_chi, dtype=np.float32)
theta = np.linspace(0, np.pi, n_theta, dtype=np.float32)
phi = np.linspace(0, 2*np.pi, n_phi, dtype=np.float32)

input_dir = "metric_output"
output_dir = "embedding_output"
os.makedirs(output_dir, exist_ok=True)

# توابع کروی 3-کره
def omega1(chi, theta, phi): return np.sin(chi) * np.sin(theta) * np.cos(phi)
def omega2(chi, theta, phi): return np.sin(chi) * np.sin(theta) * np.sin(phi)
def omega3(chi, theta): return np.sin(chi) * np.cos(theta)
def omega4(chi): return np.cos(chi)

# مشتق عددی اول
def first_deriv(f, h, axis):
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * h)

# مشتق عددی دوم
def second_deriv(f, h, axis):
    return (np.roll(f, -1, axis=axis) - 2*f + np.roll(f, 1, axis=axis)) / h**2

# نرمال‌سازی بردار
def normalize(v):
    norm = np.sqrt(np.sum(v**2, axis=0))
    return v / (norm + 1e-8)

chi_block_size = 10

# اجرای محاسبه
for t in critical_timesteps:
    print(f"🔄 Processing embedding for t={t}")
    R = np.load(os.path.join(input_dir, f"R_t{t}.npy")).astype(np.float32)

    X = np.zeros((5, n_chi, n_theta, n_phi), dtype=np.float32)

    for i in range(n_chi):
        for j in range(n_theta):
            for k in range(n_phi):
                rr = R[i, j, k]
                X[0, i, j, k] = t  # T(t)
                X[1, i, j, k] = rr * omega1(chi[i], theta[j], phi[k])
                X[2, i, j, k] = rr * omega2(chi[i], theta[j], phi[k])
                X[3, i, j, k] = rr * omega3(chi[i], theta[j])
                X[4, i, j, k] = rr * omega4(chi[i])

    np.save(os.path.join(output_dir, f"embedding_X_t{t}.npy"), X)

    # مشتقات اول: ∂_μ X^A
    dX = np.zeros((5, 4, n_chi, n_theta, n_phi), dtype=np.float32)
    for A in range(5):
        for mu in range(3):  # فقط spatial axes
            dX[A, mu] = first_deriv(X[A], dx[mu+1], axis=mu)

    # بردار نرمال با استفاده از گرام-اشمیت عددی (SVD)
    n_vec = np.zeros((5, n_chi, n_theta, n_phi), dtype=np.float32)
    for i in range(n_chi):
        for j in range(n_theta):
            for k in range(n_phi):
                J = np.array([dX[:, mu, i, j, k] for mu in range(3)])  # 3x5
                try:
                    _, _, Vh = np.linalg.svd(J)
                    n = Vh[-1]
                    n_vec[:, i, j, k] = normalize(n)
                except np.linalg.LinAlgError:
                    n_vec[:, i, j, k] = 0  # یا نگه‌دار یا ماسک کن

    np.save(os.path.join(output_dir, f"normal_n_t{t}.npy"), n_vec)

    # مشتقات دوم embedding: بلوکی روی محور χ
    for i_start in range(0, n_chi, chi_block_size):
        i_end = min(i_start + chi_block_size, n_chi)
        D2X_block = np.zeros((5, 4, 4, i_end - i_start, n_theta, n_phi), dtype=np.float32)

        for A in range(5):
            for mu in range(3):
                for nu in range(3):
                    if mu == nu:
                        D2X_block[A, mu, nu] = second_deriv(X[A, i_start:i_end], dx[mu+1], axis=mu)

        np.save(os.path.join(output_dir, f"D2X_t{t}_block_{i_start}_{i_end}.npy"), D2X_block)

    print(f"✅ Saved X, n^A, and D2X blocks for t={t}")