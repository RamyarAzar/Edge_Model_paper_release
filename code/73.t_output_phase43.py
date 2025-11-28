import numpy as np
import os

# تنظیمات پایه
w_dir = "w_output"
veff_dir = "veff_output"
output_dir = "t_output_phase43"
os.makedirs(output_dir, exist_ok=True)

critical_timesteps = list(range(33, 42 + 1))
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400

for t in critical_timesteps:
    print(f"⏳ Processing T_mu_nu at t={t}...")

    try:
        # 🔹 بارگذاری w_t و مشتقات زمانی
        def load_w(ti):
            path = os.path.join(w_dir, f"w_t{ti}.npy")
            return np.memmap(path, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

        if t > critical_timesteps[0] and t < critical_timesteps[-1]:
            w_prev = load_w(t - 1)
            w_curr = load_w(t)
            w_next = load_w(t + 1)
            dw_dt = (w_next - w_prev) / 2
        else:
            w_curr = load_w(t)
            dw_dt = np.zeros_like(w_curr)

        # 🔹 بارگذاری مشتقات مکانی میدان فاز
        dw_dchi   = np.gradient(w_curr, axis=0)
        dw_dtheta = np.gradient(w_curr, axis=1)
        dw_dphi   = np.gradient(w_curr, axis=2)

        # 🔹 بارگذاری veff_t برای تعیین چگالی انرژی پتانسیل
        veff = np.load(os.path.join(veff_dir, f"veff_t{t}.npy"))

        # 🔹 بازسازی Tμν از ساختار گرادیان فاز
        T = np.zeros((4, 4, n_chi, n_theta, n_phi))

        # تعریف بردار گرادیان فاز φ
        grad_phi = [
            dw_dt,
            dw_dchi,
            dw_dtheta,
            dw_dphi
        ]

        # تعریف Tμν = ∂μφ ∂νφ - gμν L_eff
        # در غیاب gμν → تنها بخش ∂μφ ∂νφ ذخیره می‌شود
        for mu in range(4):
            for nu in range(4):
                T[mu, nu] = grad_phi[mu] * grad_phi[nu]

        # افزودن اثر پتانسیل به قطعه T_{00}
        T[0, 0] += veff

        # 🔹 ذخیره‌سازی
        np.save(os.path.join(output_dir, f"T_recovered_t{t}.npy"), T)
        with open(os.path.join(output_dir, f"T_recovered_t{t}.txt"), 'w') as f:
            mean = np.mean(T[0,0])
            std = np.std(T[0,0])
            min_val = np.min(T[0,0])
            max_val = np.max(T[0,0])
            f.write(f"T_00 summary at t={t}:\n")
            f.write(f"Mean: {mean:.4e}\n")
            f.write(f"Std : {std:.4e}\n")
            f.write(f"Min : {min_val:.4e}\n")
            f.write(f"Max : {max_val:.4e}\n")

        print(f"✅ Done: T_mu_nu at t={t}")

    except Exception as e:
        print(f"⛔ Error at t={t}: {e}")