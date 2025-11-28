import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات
timesteps = list(range(33, 42))
shape = (400, 400, 400)
n_coords = 4  # g_{μν} is 4×4

# مسیرها
w_dir = "w_output"
ldir = "effective_field_output"
l_dir = "lagrangian_phase_recon"
g_dir = "metric"
out_dir = "veff_output"
os.makedirs(out_dir, exist_ok=True)

veff_t = []

for t in timesteps:
    print(f"⏳ Computing V_eff for t={t}...")

    # بارگذاری میدان w و مشتق زمانی آن
    w = np.memmap(os.path.join(w_dir, f"w_t{t}.npy"), dtype='float64', mode='r', shape=shape)
    dw_dt = np.load(os.path.join(ldir, f"phase_kinetic_t{t}.npy"))  # ∂ₜw²
    lag = np.load(os.path.join(l_dir, f"lagrangian_density_t{t}.npy"))

    # بارگذاری متریک گام زمانی
    g = np.load(os.path.join(g_dir, f"g_t{t}.npy"))  # شکل (4, 4, 400, 400, 400)
    g00 = g[0, 0]  # فقط g_{00} مورد نیاز است
    sqrt_neg_detg = np.sqrt(np.abs(np.linalg.det(g.transpose(2, 3, 4, 0, 1))))  # محورهای مناسب: (x,y,z,μ,ν)

    # جمله انرژی جنبشی
    T = 0.5 * g00 * dw_dt

    # پتانسیل مؤثر نقطه‌ای
    veff_density = -lag + T

    # ضرب در وزن هندسی √|−g|
    veff_weighted = veff_density * sqrt_neg_detg
    veff_integrated = np.sum(veff_weighted)

    veff_t.append(veff_integrated)

    # ذخیره چگالی
    np.save(os.path.join(out_dir, f"veff_t{t}.npy"), veff_density)

# ذخیره نتایج نهایی
veff_t = np.array(veff_t)
np.save(os.path.join(out_dir, "veff_t.npy"), veff_t)

# نمودار و خروجی متنی
plt.plot(timesteps, veff_t, marker='o')
plt.title("Effective Potential V_eff(t)")
plt.xlabel("Time step t")
plt.ylabel("V_eff(t)")
plt.grid(True)
plt.savefig(os.path.join(out_dir, "veff_summary_plot.png"))

with open(os.path.join(out_dir, "veff_summary.txt"), "w") as f:
    for t, v in zip(timesteps, veff_t):
        f.write(f"t={t}, V_eff={v:.3e}\n")