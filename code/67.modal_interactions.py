import numpy as np
import os
import matplotlib.pyplot as plt

# تنظیمات مسیرها
h_dir = "hamiltonian_modal_output"
output_dir = "modal_interactions_outputs"
os.makedirs(output_dir, exist_ok=True)

# بارگذاری فایل H_modal_t.npy
H = np.load(os.path.join(h_dir, "H_modal_t.npy")).astype(np.float64)  # شکل: (n_t, n_modes, n_modes)
n_t, n_modes, _ = H.shape

# استخراج انرژی مدها از قطر ماتریس H(t)
modal_energies = np.array([np.diag(H[t]) for t in range(n_t)])  # شکل: (n_t, n_modes)

# محاسبه ماتریس همبستگی بین مدها
cor_matrix = np.corrcoef(modal_energies.T)  # شکل: (n_modes, n_modes)

# ذخیره فایل عددی
np.save(os.path.join(output_dir, "modal_correlation_matrix.npy"), cor_matrix)

# ذخیره فایل متنی
with open(os.path.join(output_dir, "modal_interactions_summary.txt"), "w", encoding="utf-8") as f:
    f.write("🔗 Modal Interaction Correlation Matrix (⟨cᵢ cⱼ⟩ normalized)\n\n")
    for i in range(n_modes):
        for j in range(n_modes):
            f.write(f"Corr(c{i+1}, c{j+1}) = {cor_matrix[i, j]:+.4f}\t")
        f.write("\n")

# رسم heatmap از ماتریس همبستگی
plt.figure(figsize=(8, 6))
plt.imshow(cor_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.title("Modal Interaction Correlation Matrix")
plt.xlabel("Mode j")
plt.ylabel("Mode i")
plt.xticks(np.arange(n_modes), labels=[f"{j+1}" for j in range(n_modes)])
plt.yticks(np.arange(n_modes), labels=[f"{i+1}" for i in range(n_modes)])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "modal_interaction_correlation_heatmap.png"))
plt.close()