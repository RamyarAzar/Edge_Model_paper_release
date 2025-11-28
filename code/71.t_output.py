import numpy as np
import os

# ⚙️ تنظیمات گرید
timesteps = list(range(34, 42))
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
dt = 1.0

# 📁 مسیر پوشه‌ها
w_dir = "w_output"
g_dir = "metric"
veff_dir = "veff_output"
out_dir = "t_output"
os.makedirs(out_dir, exist_ok=True)

for t in timesteps:
    print(f"\n⏳ Computing full T_μν at t={t}...")

    try:
        # 📥 خواندن میدان w در سه زمان
        w_m1 = np.memmap(os.path.join(w_dir, f"w_t{t-1}.npy"), dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))
        w_0  = np.memmap(os.path.join(w_dir, f"w_t{t}.npy"),   dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))
        w_p1 = np.memmap(os.path.join(w_dir, f"w_t{t+1}.npy"), dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

        # 📥 متریک g[μ,ν,χ,θ,φ]
        g_raw = np.load(os.path.join(g_dir, f"g_t{t}.npy")).astype(np.float64)

        # 📥 veff[χ,θ,φ]
        veff = np.load(os.path.join(veff_dir, f"veff_t{t}.npy"))

        # ⛓️ مشتقات ∂_μ w
        dw = [np.gradient(w_0, axis=i) for i in range(3)]  # ∂χ, ∂θ, ∂φ
        dw.append((w_p1 - w_m1) / (2 * dt))                # ∂₀w

        # آماده‌سازی g[χ,θ,φ,μ,ν] به‌صورت یکباره
        g = np.zeros((n_chi, n_theta, n_phi, n_coords, n_coords), dtype=np.float64)
        for mu in range(n_coords):
            for nu in range(n_coords):
                g[..., mu, nu] = g_raw[mu, nu]  # از ترتیب (μ,ν,χ,θ,φ) → (χ,θ,φ,μ,ν)

        # مرحله 1: kinetic = g^{ρσ} ∂_ρ w ∂_σ w
        kinetic = np.zeros((n_chi, n_theta, n_phi), dtype=np.float64)
        for rho in range(n_coords):
            for sigma in range(n_coords):
                kinetic += g[..., rho, sigma] * dw[rho] * dw[sigma]

        # مرحله 2: محاسبه T_{μν}
        T = np.memmap(os.path.join(out_dir, f"T_t{t}.npy"), dtype='float64', mode='w+',
                      shape=(n_chi, n_theta, n_phi, n_coords, n_coords))

        for mu in range(n_coords):
            for nu in range(n_coords):
                T[..., mu, nu] = dw[mu] * dw[nu] - g[..., mu, nu] * (0.5 * kinetic - veff)

        print(f"✅ Done: T_t{t}.npy saved")

    except Exception as e:
        print(f"⛔ Error at t={t}: {e}")