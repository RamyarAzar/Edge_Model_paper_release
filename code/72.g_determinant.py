import numpy as np
import os
from numpy.linalg import det

# مسیرهای ورودی/خروجی
g_dir = 'metric'
out_dir = 'g_determinant_output'
os.makedirs(out_dir, exist_ok=True)

# تنظیمات
timesteps = [33, 34, 35, 36, 37, 38, 39, 40, 41]
shape = (400, 400, 400)

for t in timesteps:
    print(f"🔍 Calculating determinant of metric at t={t}...")
    try:
        # بارگذاری متریک: (4, 4, 400, 400, 400)
        g = np.load(os.path.join(g_dir, f"g_t{t}.npy")).astype(np.float64)

        # خروجی دترمینان: (400, 400, 400)
        det_g = np.zeros(shape, dtype=np.float64)
        singular_points = []

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    g_local = g[:, :, i, j, k]
                    try:
                        d = det(g_local)
                        det_g[i, j, k] = d
                        if abs(d) < 1e-10:
                            singular_points.append((i, j, k, d))
                    except:
                        det_g[i, j, k] = 0.0
                        singular_points.append((i, j, k, 0.0))

        # ذخیره npy
        np.save(os.path.join(out_dir, f"det_g_t{t}.npy"), det_g)

        # ذخیره txt: مقادیر بحرانی یا کوچک
        txt_path = os.path.join(out_dir, f"det_g_summary_t{t}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Determinant summary for g_t{t}:\n")
            f.write(f"Shape: {det_g.shape}\n")
            f.write(f"Min: {np.min(det_g)}\n")
            f.write(f"Max: {np.max(det_g)}\n")
            f.write(f"Mean: {np.mean(det_g)}\n")
            f.write(f"Std: {np.std(det_g)}\n")
            f.write(f"\nSingular or near-zero points (up to 1000):\n")
            for idx, (i, j, k, dval) in enumerate(singular_points[:1000]):
                f.write(f"({i},{j},{k}) => det = {dval}\n")

        print(f"✅ Done: det(g) at t={t}, saved to .npy and .txt")

    except Exception as e:
        print(f"⛔ Error at t={t}: {e}")