import numpy as np
import os
import matplotlib.pyplot as plt

# ==== تنظیمات اولیه ====
t_range = range(33, 43)
n_chi, n_theta, n_phi = 400, 400, 400
block_size = 10
n_coords = 4

# مسیر پوشه‌ها
w_dir = "w_output"
veff_dir = "veff_output"
r4_dir = "ricci_output"
hbar_eff_path = "hbar_eff_t.npy"
output_dir = "planck_rescaled_output"
os.makedirs(output_dir, exist_ok=True)

# ==== ثابت‌های پلانک ====
hbar_planck = 1.054571817e-34
c = 2.99792458e8
G = 6.67430e-11

l_planck = np.sqrt(hbar_planck * G / c**3)
e_planck = np.sqrt(hbar_planck * c**5 / G)

# ==== بارگذاری ħ_eff ====
hbar_eff_all = np.load(hbar_eff_path)

# ==== محاسبه برای هر t ====
for t in t_range:
    print(f"\n🔁 Processing t={t}...")
    
    # بارگذاری ħ_eff(t) و نرمال‌سازی
    hbar_eff_t = hbar_eff_all[t]
    hbar_eff_norm = hbar_eff_t / hbar_planck
    
    # بارگذاری میدان w و نرمال‌سازی نسبت به مقدار میانگین
    w = np.memmap(os.path.join(w_dir, f"w_t{t}.npy"), dtype=np.float64, mode='r',
                  shape=(n_chi, n_theta, n_phi))
    w_mean = np.mean(np.abs(w))
    w0 = w_mean if w_mean != 0 else 1.0

    # بارگذاری ترم‌های R^{(4)} و V_eff
    r4 = np.memmap(os.path.join(r4_dir, f"Rscalar_t{t}.npy"), dtype=np.float32, mode='r',
                   shape=(n_chi, n_theta, n_phi))
    veff = np.memmap(os.path.join(veff_dir, f"veff_t{t}.npy"), dtype=np.float64, mode='r',
                     shape=(n_chi, n_theta, n_phi))
    
    # خروجی میانگین برای txt
    lagrangian_mean = []

    # پیمایش بلوک‌ها
    for chi_start in range(0, n_chi, block_size):
        chi_end = min(chi_start + block_size, n_chi)

        w_b = w[chi_start:chi_end]
        veff_b = veff[chi_start:chi_end]
        r4_b = r4[chi_start:chi_end]

        # نرمال‌سازی‌ها
        w_tilde = w_b / w0
        veff_tilde = veff_b / e_planck**4
        r4_tilde = l_planck**2 * r4_b

        # ساخت لاگرانژین بدون‌بعد (log-safe)
        grad_term = np.zeros_like(w_tilde)
        lagrangian = np.zeros_like(w_tilde)

        try:
            grad_term += 0.0  # اینجا می‌توان گرادیان‌ها را بعداً اضافه کرد
        except:
            pass

        # استفاده از ضرب لگاریتمی برای جلوگیری از overflow
        def log_safe_product(a, b):
            return np.exp(np.log(np.abs(a) + 1e-300) + np.log(np.abs(b) + 1e-300)) * np.sign(a * b)

        term_kinetic = grad_term  # در این مرحله صفر است
        term_potential = veff_tilde
        term_curvature = 0.5 * (hbar_eff_norm**2) * log_safe_product(r4_tilde, w_tilde**2)

        lagrangian = term_kinetic - term_potential - term_curvature
        lagrangian_mean.append(np.mean(lagrangian))

        # ذخیره خروجی
        block_name = f"t{t}_chi{chi_start}"
        np.save(os.path.join(output_dir, f"lagrangian_block_{block_name}.npy"), lagrangian)

        with open(os.path.join(output_dir, f"lagrangian_block_{block_name}.txt"), "w") as f:
            flat = lagrangian.flatten()
            for val in flat:
                f.write(f"{val:.6e}\n")

        # هیستوگرام بصری از مقادیر لاگرانژی
        plt.hist(lagrangian.flatten(), bins=100, log=True)
        plt.title(f"Lagrangian Histogram (t={t}, chi {chi_start}-{chi_end})")
        plt.xlabel("L dimensionless")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, f"lagrangian_hist_{block_name}.png"))
        plt.clf()

    # میانگین کلی برای t
    with open(os.path.join(output_dir, f"L_mean_t{t}.txt"), "w") as f:
        for val in lagrangian_mean:
            f.write(f"{val:.6e}\n")

print("\n✅ Planck-scale rescaling completed.")