import numpy as np
import os
import matplotlib.pyplot as plt

# مسیر داده‌ها
t_steps = np.arange(33, 42)              # 9 زمان گسسته
T_vals = t_steps.astype(float)

# بارگذاری داده‌های انرژی مؤثر (ℏ_eff * ω)
hbar_eff = np.load('hbar_eff_t.npy')     # (9,)
omega = 3.633e+122                       # ثابت فرکانس نوسان
H_vals = hbar_eff * omega                # (9,)

# ساخت فولدر خروجی
output_dir = 'quantum_operators_phase315'
os.makedirs(output_dir, exist_ok=True)

# --- 1. تعریف اپراتورها به صورت ماتریس Hermitian قطری ---
T_op = np.diag(T_vals)
H_op = np.diag(H_vals)

# --- 2. بررسی Hermitian بودن ---
def is_hermitian(matrix):
    return np.allclose(matrix, matrix.conj().T)

print("Is T Hermitian?", is_hermitian(T_op))
print("Is H Hermitian?", is_hermitian(H_op))

# --- 3. محاسبه مقادیر ویژه ---
eig_T = np.linalg.eigvalsh(T_op)
eig_H = np.linalg.eigvalsh(H_op)

# --- 4. ذخیره اپراتورها و مقادیر ویژه ---
np.save(os.path.join(output_dir, 'T_operator.npy'), T_op)
np.save(os.path.join(output_dir, 'H_eff_operator.npy'), H_op)
np.save(os.path.join(output_dir, 'eig_T.npy'), eig_T)
np.save(os.path.join(output_dir, 'eig_H.npy'), eig_H)

# --- 5. رسم اسپکتروم ---
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t_steps, eig_T, 'o-', label='eig(T)')
plt.xlabel('t')
plt.ylabel('Eigenvalue')
plt.title('Spectrum of T Operator')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_steps, eig_H, 'o-', color='red', label='eig(H_eff)')
plt.xlabel('t')
plt.ylabel('Eigenvalue')
plt.title('Spectrum of H_eff Operator')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'quantum_operator_spectra.png'))
plt.show()