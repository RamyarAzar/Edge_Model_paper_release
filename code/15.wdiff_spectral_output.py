import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y
from tqdm import tqdm

# پارامترها
critical_timesteps = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]
n_chi, n_theta, n_phi = 400, 400, 400
l_max = 30
spin = 0

# شبکه زاویه‌ای
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2*np.pi, n_phi)
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
dΩ = (np.pi / n_theta) * (2 * np.pi / n_phi)

# تولید پایه‌های کروی
Y_basis = []
for l in range(abs(spin), l_max + 1):
    for m in range(-l, l + 1):
        Ylm = sph_harm_y(m, l, phi_grid, theta_grid)
        Y_basis.append((l, m, Ylm))

l_vals = np.array([l for (l, m, _) in Y_basis])
m_vals = np.array([m for (l, m, _) in Y_basis])
n_modes = len(Y_basis)

# مسیرها
wdiff_dir = "w_diff_output"
output_dir = "wdiff_spectral_output"
os.makedirs(output_dir, exist_ok=True)

for t in critical_timesteps:
    print(f"\n📊 Spectral analysis for w_diff at t={t}...")

    # بارگذاری داده
    wdiff = np.memmap(os.path.join(wdiff_dir, f"w_diff_t{t}.npy"), dtype=np.float64, mode='r',
                      shape=(n_chi, n_theta, n_phi))

    Cl_array = np.zeros((n_chi, l_max + 1), dtype=np.float64)

    for chi in tqdm(range(n_chi), desc=f"  Processing χ-layers"):
        f = wdiff[chi]  # f(θ, φ)

        alm = np.zeros(n_modes, dtype=np.complex128)
        for idx, (l, m, Ylm) in enumerate(Y_basis):
            integrand = f * np.conj(Ylm) * np.sin(theta_grid)
            alm[idx] = np.sum(integrand) * dΩ

        for l in range(l_max + 1):
            mask = (l_vals == l)
            Cl_array[chi, l] = np.mean(np.abs(alm[mask])**2)

    # ذخیره فایل فشرده
    npz_path = os.path.join(output_dir, f"Cl_wdiff_t{t}_s{spin}.npz")
    np.savez_compressed(npz_path, Cl=Cl_array, l=np.arange(l_max + 1))
    print(f"✅ Saved: {npz_path}")

    # رسم میانگین توان طیفی
    Cl_mean = np.mean(Cl_array, axis=0)

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(l_max + 1), Cl_mean, marker='o')
    plt.xlabel("ℓ", fontsize=13)
    plt.ylabel(r"$\langle C_\ell \rangle_\chi$", fontsize=13)
    plt.title(f"Mean $C_\\ell$ for $w_{{diff}}$ at $t={t}$, $s=0$", fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"Cl_wdiff_t{t}_s0.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"🖼️ Plot saved: {fig_path}")