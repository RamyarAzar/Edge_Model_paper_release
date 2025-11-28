import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max
from scipy.spatial import ConvexHull
import os

# زمان‌های بحرانی
critical_timesteps = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]

# پارامترها
chi_index = 200
shape = (400, 400)
memmap_dtype = np.float64
w_dir = "./w_output"
output_dir = "./quantum_clusters"
os.makedirs(output_dir, exist_ok=True)

def load_w_memmap(filepath, shape):
    return np.memmap(filepath, dtype=memmap_dtype, mode='r', shape=shape)

def compute_gradients(w):
    dw_theta = sobel(w, axis=0)
    dw_phi = sobel(w, axis=1)
    return dw_theta, dw_phi

def compute_peff(w, dw_theta, dw_phi):
    p_theta = w * dw_theta
    p_phi = w * dw_phi
    return p_theta, p_phi

def detect_peaks(p_mag, min_distance=8, threshold_abs=0.1):
    return peak_local_max(p_mag, min_distance=min_distance, threshold_abs=threshold_abs)

def cluster_peaks(peaks, eps=5, min_samples=5):
    if len(peaks) == 0:
        return np.empty((0,)), []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(peaks)
    return clustering.labels_, clustering

def draw_clusters(θ, φ, peaks, labels, p_mag, t):
    Φ, Θ = np.meshgrid(φ, θ)
    plt.figure(figsize=(8, 6))
    plt.imshow(p_mag, cmap='inferno', extent=[φ[0], φ[-1], θ[0], θ[-1]], origin='lower', aspect='auto')
    plt.colorbar(label=r"$|\vec{p}_{\mathrm{eff}}|$")

    if len(peaks) > 0:
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue  # noise
            members = peaks[labels == cluster_id]
            plt.scatter(φ[members[:, 1]], θ[members[:, 0]], label=f"Cluster {cluster_id}", s=20)
            if len(members) >= 3:
                try:
                    hull = ConvexHull(members)
                    for simplex in hull.simplices:
                        plt.plot(
                            φ[members[simplex, 1]],
                            θ[members[simplex, 0]],
                            'white', linewidth=1, alpha=0.7
                        )
                except:
                    pass

    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\theta$")
    plt.title(rf"Quantum Clusters at $t={t},\ \chi={chi_index}$")
    plt.tight_layout()
    plt.legend(fontsize=8, loc='upper right')
    out_path = os.path.join(output_dir, f"quantum_clusters_t{t}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def process_timestep(t_index):
    print(f"[+] Processing timestep t={t_index}...")
   
    w_path = os.path.join(w_dir, f"w_t{t_index}.npy")
    w_memmap = load_w_memmap(w_path, shape)
    w = np.array(w_memmap, dtype=np.float64)
    w = (w - np.min(w)) / (np.max(w) - np.min(w) + 1e-12)

    dw_theta, dw_phi = compute_gradients(w)
    p_theta, p_phi = compute_peff(w, dw_theta, dw_phi)
    p_mag = np.sqrt(p_theta**2 + p_phi**2)

    peaks = detect_peaks(p_mag)
    labels, _ = cluster_peaks(peaks)

    θ = np.linspace(0, np.pi, shape[0])
    φ = np.linspace(0, 2 * np.pi, shape[1])

    # ذخیره نقاط و برچسب‌ها
    np.save(os.path.join(output_dir, f"clusters_peaks_t{t_index}.npy"), peaks)
    np.save(os.path.join(output_dir, f"clusters_labels_t{t_index}.npy"), labels)

    draw_clusters(θ, φ, peaks, labels, p_mag, t_index)
    print(f"[✓] Saved cluster image and npy for t={t_index}")

for t in critical_timesteps:
    process_timestep(t)
