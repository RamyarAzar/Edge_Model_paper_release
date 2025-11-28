import os
import numpy as np
import pandas as pd

# مسیرها
w_dir = 'lagrangian_phase315'
field_dir = 'effective_field_output'
out_dir = 'hamiltonian_phase315_nograd'
os.makedirs(out_dir, exist_ok=True)

# گام‌های زمانی
time_steps = list(range(33, 42))
shape = (400, 400, 400)

# شبکه
dx = dy = dz = 1.0
volume_element = dx * dy * dz

hamiltonian_total = []

for t in time_steps:
    print(f"در حال محاسبه H(t={t}) بدون گرادیان...")

    # بارگذاری مشتق زمانی و پتانسیل
    pk_path = os.path.join(field_dir, f"phase_kinetic_t{t}.npy")
    veff_path = os.path.join(field_dir, f"veff_t{t}.npy")
    phase_kinetic = np.memmap(pk_path, dtype='float64', mode='r', shape=shape)
    veff = np.memmap(veff_path, dtype='float64', mode='r', shape=shape)

    # چگالی همیلتونی مؤثر بدون گرادیان
    H_density = 0.5 * phase_kinetic + veff

    # ذخیره چگالی
    h_path = os.path.join(out_dir, f"hamiltonian_density_t{t}.npy")
    np.save(h_path, H_density)

    # انتگرال عددی روی شبکه
    H_t = np.sum(H_density) * volume_element
    hamiltonian_total.append(H_t)

    print(f"H(t={t}) = {H_t:.3e}")

# ذخیره خروجی‌ها
np.save(os.path.join(out_dir, "hamiltonian_total.npy"), np.array(hamiltonian_total))

df = pd.DataFrame({
    'time_step': time_steps,
    'H_total': hamiltonian_total
})
df.to_csv(os.path.join(out_dir, "hamiltonian_total.csv"), index=False)

print("✅ محاسبه H(t) بدون گرادیان و ذخیره در hamiltonian_phase315_nograd تکمیل شد.")