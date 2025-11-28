import os
import numpy as np
import matplotlib.pyplot as plt

# تنظیمات
psi_dir = "wavefunction_outputs"
L_eff_dir = "lagrangian_final_outputs"
output_dir = "spectral_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# پارامترهای گرید
Nx, Ny, Nz = 400, 400, 400
dx = dy = dz = 1.0  # در صورت نیاز با مقدار واقعی جایگزین شود

# طیف برداری k برای محورها
kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
kz = np.fft.fftfreq(Nz, d=dz) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)

# تنظیم bin های طیفی
k_max = np.max(k_mag)
k_bins = np.linspace(0, k_max, 100)
dk = k_bins[1] - k_bins[0]
k_indices = np.digitize(k_mag.flat, k_bins) - 1

# تحلیل برای هر t
for t in range(33, 42):
    print(f"🔍 Processing spectral decomposition at t={t}...")

    # بارگذاری ψ و L_eff
    psi = np.load(os.path.join(psi_dir, f"psi_t{t}.npy")).astype(np.complex128)
    L_eff = np.load(os.path.join(L_eff_dir, f"L_eff_t{t}.npy")).astype(np.float64)

    # FFT و چرخش
    psi_fft = np.fft.fftn(psi)
    psi_fft_shifted = np.fft.fftshift(psi_fft)
    fft_mag_sq = np.abs(psi_fft_shifted) ** 2

    # طیف شعاعی برای ψ
    spectrum = np.zeros(len(k_bins), dtype=np.float64)
    for i in range(len(k_bins)):
        spectrum[i] = np.sum(fft_mag_sq.flat[k_indices == i])

    # ذخیره طیف عددی
    np.save(os.path.join(output_dir, f"spectrum_t{t}.npy"), spectrum)

    # ترسیم نمودار طیفی
    plt.figure(figsize=(8, 5))
    plt.plot(k_bins, spectrum, label=f"t={t}")
    plt.xlabel("k (1/unit length)")
    plt.ylabel("Spectral Power |ψ̃(k)|²")
    plt.title(f"Spectral Decomposition of ψ at t={t}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"spectrum_plot_t{t}.png"))
    plt.close()

    print(f"✅ Spectrum computed and saved at t={t}")