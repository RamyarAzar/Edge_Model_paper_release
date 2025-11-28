import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import os

# مسیرها
veff_dir = "veff_output"
topo_dir = "topology_analysis"
timesteps = np.arange(33, 42)

# 1. بارگذاری داده‌ها
veff_t = np.array([
    np.mean(np.load(os.path.join(veff_dir, f"veff_t{t}.npy")))
    for t in timesteps
])
nodal_counts = np.load(os.path.join(topo_dir, "nodal_counts.npy"))
avg_topo_energy = np.load(os.path.join(topo_dir, "avg_topo_energy.npy"))

# 2. نرمال‌سازی داده‌ها
X = np.column_stack((nodal_counts, avg_topo_energy))
y = veff_t

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_norm = scaler_X.fit_transform(X)
y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 3. رگرسیون خطی
model = LinearRegression()
model.fit(X_norm, y_norm)

r2_score = model.score(X_norm, y_norm)
coeffs = model.coef_
intercept = model.intercept_

# 4. همبستگی پیرسون
corr_nodes, pval_nodes = pearsonr(nodal_counts, veff_t)
corr_topo, pval_topo = pearsonr(avg_topo_energy, veff_t)

# 5. ذخیره فایل متنی تحلیل
with open("regression_topology_veff.txt", "w") as f:
    f.write(" Linear Regression Results:\n")
    f.write(f"R^2 Score = {r2_score:.4f}\n")
    f.write(f"Coefficients:\n")
    f.write(f"  • Nodal Count = {coeffs[0]:.6f}\n")
    f.write(f"  • Topological Energy = {coeffs[1]:.6f}\n")
    f.write(f"Intercept = {intercept:.6f}\n\n")
    f.write(" Pearson Correlation:\n")
    f.write(f"corr(N_t, V_eff) = {corr_nodes:.4f}  (p = {pval_nodes:.2e})\n")
    f.write(f"corr(E_topo, V_eff) = {corr_topo:.4f}  (p = {pval_topo:.2e})\n")

# 6. رسم نمودار
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(timesteps, nodal_counts, 'r-o')
plt.title("Nodal Count over Time")
plt.xlabel("Time Step t")
plt.ylabel("N_t")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(timesteps, avg_topo_energy, 'b-o')
plt.title("Topological Energy ⟨|∇w|²⟩")
plt.xlabel("Time Step t")
plt.ylabel("Topo Energy")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(timesteps, veff_t, 'g-o')
plt.title("Effective Potential V_eff(t)")
plt.xlabel("Time Step t")
plt.ylabel("V_eff")
plt.grid(True)

plt.tight_layout()
plt.savefig("topology_veff_regression_plot.png", dpi=150)