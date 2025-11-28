import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.linalg import svd
from sklearn.decomposition import PCA

# مسیر ورودی و خروجی
psi_path = 'wavefunction_phase315/psi_t_complex.npy'
output_dir = 'hilbert_space_phase315'
os.makedirs(output_dir, exist_ok=True)

# بارگذاری تابع موج کمپلکس ψ(t)
psi = np.load(psi_path)  # shape: (9,)
time_steps = list(range(33, 42))

# --- 1. تعریف فضای هیلبرت گسسته C^9 ---
# هر تابع موج، یک بردار 9-بعدی کمپلکس است.
H_eff = psi.copy()

# --- 2. تعریف ضرب داخلی ---
def inner_product(psi1, psi2):
    return np.vdot(psi1, psi2)  # شامل مزدوج کمپلکس ψ1

# مثال تستی: ضرب داخلی خود با خود باید 1 باشد
norm = np.sqrt(inner_product(psi, psi))

# --- 3. ساخت ماتریس حالت برای آنالیز طیفی ---
# اگر فقط یک ψ داریم، برای نمایش PCA چند کپی با نویز می‌سازیم
psi_matrix = np.stack([psi + np.random.normal(0, 0.01, size=9) * 1j for _ in range(50)])

# --- 4. بسط طیفی با PCA ---
pca = PCA(n_components=3)
psi_pca = pca.fit_transform(psi_matrix.real)

# ذخیره خروجی‌های طیفی
np.save(os.path.join(output_dir, 'psi_matrix.npy'), psi_matrix)
np.save(os.path.join(output_dir, 'psi_pca_modes.npy'), pca.components_)

# --- 5. رسم بردارهای پایه طیفی ---
plt.figure(figsize=(8, 4))
for i in range(3):
    plt.plot(time_steps, pca.components_[i], label=f'Mode {i+1}')
plt.title('Spectral Basis Modes (PCA)')
plt.xlabel('Time Step')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spectral_basis_modes.png'))
plt.show()