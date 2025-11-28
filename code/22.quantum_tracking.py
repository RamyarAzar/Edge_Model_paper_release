import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial.distance import cdist

# تنظیمات
w_dir = "w_output"
output_dir = "quantum_tracking"
os.makedirs(output_dir, exist_ok=True)
critical_times = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]
chi_idx = 200
eps = 1e-6
node_threshold = 0.02  # آستانه برای شناسایی گره‌ها

# تابع محاسبه گرادیان‌ها و p_eff
def compute_peff(w):
    dw_dtheta, dw_dphi = np.gradient(w, axis=(0, 1))
    p_theta = -dw_dtheta
    p_phi = -dw_dphi
    p_mag = np.sqrt(p_theta**2 + p_phi**2)
    return p_mag, p_theta, p_phi

# تابع شناسایی گره‌های موضعی
def detect_nodes(p_mag, threshold=node_threshold):
    local_max = maximum_filter(p_mag, size=3) == p_mag
    mask = (p_mag > threshold) & local_max
    coords = np.argwhere(mask)
    return coords  # shape: (n_nodes, 2)

# تابع محاسبه تطبیق بین دو لیست گره‌ها
def match_nodes(coords_prev, coords_next, max_dist=5):
    if len(coords_prev) == 0 or len(coords_next) == 0:
        return [], list(range(len(coords_prev))), list(range(len(coords_next)))

    dists = cdist(coords_prev, coords_next)
    matched = []
    prev_unmatched = set(range(len(coords_prev)))
    next_unmatched = set(range(len(coords_next)))
   
    for i, row in enumerate(dists):
        j = np.argmin(row)
        if row[j] < max_dist:
            matched.append((i, j))
            prev_unmatched.discard(i)
            next_unmatched.discard(j)
   
    return matched, list(prev_unmatched), list(next_unmatched)

# شروع تحلیل زمانی
prev_nodes = None
for i, t in enumerate(critical_times):
    print(f"⏳ Analyzing nodes at t={t}...")
    w_path = os.path.join(w_dir, f"w_t{t}.npy")
    w = np.memmap(w_path, dtype='float64', mode='r', shape=(400, 400, 400))
    w_slice = np.copy(w[:, :, chi_idx])

    p_mag, p_theta, p_phi = compute_peff(w_slice)
    nodes = detect_nodes(p_mag)

    # ذخیره تصویر گره‌ها
    fig, ax = plt.subplots()
    ax.imshow(p_mag.T, origin='lower', cmap='inferno')
    if len(nodes) > 0:
        ax.scatter(nodes[:, 0], nodes[:, 1], s=10, color='cyan', label='Nodes')
    ax.set_title(f"Quantum Nodes at t={t}")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\phi$")
    ax.legend()
    plt.savefig(os.path.join(output_dir, f"quantum_nodes_t{t}.png"))
    plt.close()

    # ردیابی با مرحله قبل
    if prev_nodes is not None:
        matched, deaths, births = match_nodes(prev_nodes, nodes)
        with open(os.path.join(output_dir, f"node_events_t{t}.txt"), "w", encoding="utf-8") as f:
            f.write(f"📌 Node Dynamics at t={t}:\n")
            f.write(f"Total matched: {len(matched)}\n")
            f.write(f"Deaths (vanished nodes): {len(deaths)}\n")
            f.write(f"Births (new nodes): {len(births)}\n")
            if deaths:
                f.write(f"→ Indices of vanished nodes: {deaths}\n")
            if births:
                f.write(f"→ Indices of new nodes: {births}\n")
    prev_nodes = nodes