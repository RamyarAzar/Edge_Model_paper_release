import numpy as np
import os

# تنظیمات
w_dir = 'w_output'
output_dir = 'w_derivatives'
os.makedirs(output_dir, exist_ok=True)

n_chi, n_theta, n_phi = 400, 400, 400
critical_times = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]

# تفاضلات شبکه فرضی (در صورت نداشتن مشخصات هندسی دقیق)
dchi = 1.0
dtheta = 1.0
dphi = 1.0

# اپراتور تفاضل مرکزی مرتبه دوم
def central_diff(f, axis, dx):
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * dx)

def laplacian(f, dx, dy, dz):
    d2x = (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dx**2
    d2y = (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / dy**2
    d2z = (np.roll(f, -1, axis=2) - 2 * f + np.roll(f, 1, axis=2)) / dz**2
    return d2x + d2y + d2z

for t in critical_times:
    try:
        print(f"🔁 Processing t={t}...")
       
        # بارگذاری حافظه‌ای
        w_file = os.path.join(w_dir, f"w_t{t}.npy")
        w_data = np.memmap(w_file, dtype='float64', mode='r', shape=(n_chi, n_theta, n_phi))

        # تبدیل به array معمولی برای پردازش
        w = np.array(w_data)

        # محاسبه گرادیان (∇w)
        grad_chi = central_diff(w, axis=0, dx=dchi)
        grad_theta = central_diff(w, axis=1, dx=dtheta)
        grad_phi = central_diff(w, axis=2, dx=dphi)
        grad = np.stack([grad_chi, grad_theta, grad_phi])  # shape: (3, 400, 400, 400)

        # محاسبه لاپلاسین (∇²w)
        lap = laplacian(w, dchi, dtheta, dphi)

        # تولید ماسک نقاط معتبر عددی
        mask = np.isfinite(w) & np.isfinite(lap)
        mask &= np.all(np.isfinite(grad), axis=0)

        # ذخیره خروجی‌ها
        np.save(os.path.join(output_dir, f"w_grad_t{t}.npy"), grad)
        np.save(os.path.join(output_dir, f"w_box_t{t}.npy"), lap)
        np.save(os.path.join(output_dir, f"w_mask_t{t}.npy"), mask)

        print(f"✅ Done: t={t}")

    except Exception as e:
        print(f"⛔ Error at t={t}: {e}")