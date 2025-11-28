import numpy as np
import os

# مسیر پوشه‌های داده
psi_dir = 'wavefunction_outputs'
phase_dir = 'phase_analysis_outputs'
output_dir = 'effective_field_output'
os.makedirs(output_dir, exist_ok=True)

# تعریف بازه‌های زمانی مورد استفاده
time_steps = [33, 34, 35, 36, 37, 38, 39, 40, 41]

# پارامترهای شبکه
Nx, Ny, Nz = 400, 400, 400
dx = dy = dz = 1.0  # فرض شده یکنواخت است
dt = 1.0            # فاصله زمانی بین اسنپ‌شات‌ها

# حلقه بر روی زمان
for t in time_steps:
    print(f'🔄 Processing t={t}...')

    # بارگذاری داده‌ها
    psi = np.load(os.path.join(psi_dir, f'psi_t{t}.npy'))
    amp = np.load(os.path.join(phase_dir, f'amp_t{t}.npy'))
    phase = np.load(os.path.join(phase_dir, f'phase_t{t}.npy'))

    # مشتق‌گیری عددی گرادیان فاز
    grad_phase = np.gradient(phase, dx, dy, dz, edge_order=2)

    # چگالی انرژی جنبشی فاز: (|∇ϕ|)^2
    grad_squared = sum(g**2 for g in grad_phase)
    phase_kinetic = amp**2 * grad_squared

    # چگالی انرژی پتانسیل مؤثر از تابع موج: v_eff = −(∇²A)/A + (∇ϕ)²
    laplacian_amp = (
        np.gradient(np.gradient(amp, dx, axis=0), dx, axis=0) +
        np.gradient(np.gradient(amp, dy, axis=1), dy, axis=1) +
        np.gradient(np.gradient(amp, dz, axis=2), dz, axis=2)
    )
    v_eff = -laplacian_amp / (amp + 1e-10) + grad_squared

    # ذخیره خروجی‌ها
    np.save(os.path.join(output_dir, f'veff_t{t}.npy'), v_eff)
    np.save(os.path.join(output_dir, f'phase_kinetic_t{t}.npy'), phase_kinetic)

    print(f'✅ t={t} done.')