import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# مسیر فایل ورودی
lagr_dir = 'lagrangian_phase315'
path_dir = 'path_integral_phase315'
os.makedirs(path_dir, exist_ok=True)

# گام‌های زمانی
time_steps = list(range(33, 42))
delta_t = 1.0  # اگر گام زمانی ابعاد دارد، اصلاح شود

# بارگذاری لاگرانژی نرمال‌شده
L_log = np.load(os.path.join(lagr_dir, 'lagrangian_lognorm.npy'))

# اکشن گسسته: مجموع L_log × Δt
S_eff = np.sum(L_log) * delta_t

# ساخت تابع مسیر آماری (احتمال مسیر یا وزن)
P_path = np.exp(-L_log)  # نسخه آماری وزن مسیر

# ذخیره داده‌ها
np.save(os.path.join(path_dir, 'S_eff_total.npy'), np.array([S_eff]))
np.save(os.path.join(path_dir, 'P_path.npy'), P_path)

df = pd.DataFrame({
    'time_step': time_steps,
    'L_log_norm': L_log,
    'P_path': P_path
})
df.to_csv(os.path.join(path_dir, 'path_integral_data.csv'), index=False)

# رسم نمودار وزن مسیر
plt.figure(figsize=(8, 4))
plt.plot(time_steps, P_path, marker='o', color='green')
plt.title('Path Probability Weight based on log-normalized L(t)')
plt.xlabel('Time Step')
plt.ylabel('P ~ exp(-L)')
plt.grid(True)
plt.tight_layout()
plt.show()