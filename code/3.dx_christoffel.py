import numpy as np
import os
import matplotlib.pyplot as plt

# پارامترها
timesteps = list(range(33, 43))
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
dx = [1.0, np.pi/n_chi, np.pi/n_theta, 2*np.pi/n_phi]  # dt فرضی=1

chi_block = 10
theta_block = 10

# مسیرهای ورودی و خروجی
input_dir = "metric_output"
output_dir = "dg_christoffel_output"
os.makedirs(output_dir, exist_ok=True)

# شبکه
chi = np.linspace(0, np.pi, n_chi, dtype=np.float32)
theta = np.linspace(0, np.pi, n_theta, dtype=np.float32)
phi = np.linspace(0, 2*np.pi, n_phi, dtype=np.float32)

dg_mean_all = []
ddg_mean_all = []
gamma_mean_all = []

def numerical_deriv(f, h, axis):
    return np.gradient(f, h, axis=axis, edge_order=2).astype(np.float32)

for t in timesteps:
    print(f"🔄 Computing for t={t}")
    g = np.load(os.path.join(input_dir, f"g_t{t}.npy"), mmap_mode='r')  # shape (4,4,400,400,400)

    # مشتقات اول
    dg_path = os.path.join(output_dir, f"dg_t{t}.npy")
    dg = np.memmap(dg_path, dtype=np.float32, mode='w+', shape=(4,4,4,n_chi,n_theta,n_phi))

    # مشتقات دوم
    ddg_path = os.path.join(output_dir, f"ddg_t{t}.npy")
    ddg = np.memmap(ddg_path, dtype=np.float32, mode='w+', shape=(4,4,4,n_chi,n_theta,n_phi))

    # گام اول: محاسبه dg و ddg بلوک به بلوک
    for mu in range(n_coords):
        for nu in range(n_coords):
            g_mn = g[mu, nu]
            for lam in range(1, 4):  # ∂_χ, ∂_θ, ∂_φ
                dg[lam, mu, nu] = numerical_deriv(g_mn, dx[lam], axis=lam - 1)
                ddg[lam, mu, nu] = numerical_deriv(dg[lam, mu, nu], dx[lam], axis=lam - 1)

    # گام دوم: محاسبه g^{-1} و Γ
    gamma_path = os.path.join(output_dir, f"Gamma_t{t}.npy")
    Gamma = np.memmap(gamma_path, dtype=np.float32, mode='w+', shape=(4,4,4,n_chi,n_theta,n_phi))

    ginv = np.zeros((4,4,n_chi,n_theta,n_phi), dtype=np.float32)
    inv_mask = np.ones((n_chi,n_theta,n_phi), dtype=bool)

    for i in range(n_chi):
        for j in range(n_theta):
            for k in range(n_phi):
                try:
                    ginv[:,:,i,j,k] = np.linalg.inv(g[:,:,i,j,k].astype(np.float64))
                except np.linalg.LinAlgError:
                    inv_mask[i,j,k] = False
                    ginv[:,:,i,j,k] = np.nan

    np.save(os.path.join(output_dir, f"mask_invfail_t{t}.npy"), inv_mask)

    # گام سوم: ساخت کریستوفل
    for lam in range(n_coords):
        for mu in range(n_coords):
            for nu in range(n_coords):
                for i_start in range(0, n_chi, chi_block):
                    i_end = min(i_start + chi_block, n_chi)
                    for j_start in range(0, n_theta, theta_block):
                        j_end = min(j_start + theta_block, n_theta)
                        temp = np.zeros((i_end-i_start, j_end-j_start, n_phi), dtype=np.float32)
                        for sigma in range(n_coords):
                            try:
                                term1 = dg[mu, sigma, nu, i_start:i_end, j_start:j_end]
                                term2 = dg[nu, sigma, mu, i_start:i_end, j_start:j_end]
                                term3 = dg[sigma, mu, nu, i_start:i_end, j_start:j_end]
                                part = term1 + term2 - term3
                                temp += ginv[lam, sigma, i_start:i_end, j_start:j_end] * part
                            except:
                                continue
                        Gamma[lam, mu, nu, i_start:i_end, j_start:j_end] = 0.5 * temp

    # گام چهارم: ثبت مقادیر میانگین برای نمودارها
    dg_mean_all.append(np.mean(np.abs(dg)))
    ddg_mean_all.append(np.mean(np.abs(ddg)))
    gamma_mean_all.append(np.mean(np.abs(Gamma)))

    del dg, ddg, Gamma
    print(f"✅ t={t} done")

# گام نهایی: رسم نمودار تغییرات
plt.figure(figsize=(10,5))
plt.plot(timesteps, dg_mean_all, marker='o', label='|∂g| mean')
plt.plot(timesteps, ddg_mean_all, marker='s', label='|∂²g| mean')
plt.plot(timesteps, gamma_mean_all, marker='^', label='|Γ| mean')
plt.xlabel("Time step")
plt.ylabel("Mean magnitude")
plt.title("Temporal evolution of derivatives and Christoffel symbols")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "derivatives_christoffel_summary.png"))
plt.show()
