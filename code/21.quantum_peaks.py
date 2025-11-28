import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import sobel

# پارامترها
grid_shape = (400, 400, 400)  # χ, θ, φ
chi_index = 200
timesteps = [2, 10, 25, 30, 33, 36, 39, 42, 45, 47]
w_path = "w_t{t}.npy"  # memmap file path
output_dir = "quantum_peaks"
os.makedirs(output_dir, exist_ok=True)

for t in timesteps:
    print(f"🔍 Processing quantum structure at t={t}...")
   
    # Load w via memmap
    w = np.memmap(w_path.format(t=t), dtype='float64', mode='r', shape=grid_shape)
    w_slice = np.array(w[chi_index, :, :])  # θ, φ
   
    # Compute angular gradients
    dw_dtheta = sobel(w_slice, axis=0) / (np.pi / 400)
    dw_dphi   = sobel(w_slice, axis=1) / (2 * np.pi / 400)
   
    # Compute effective quantum energy flow vector
    J_theta = w_slice * dw_dtheta
    J_phi   = w_slice * dw_dphi
    J_mag = np.sqrt(J_theta**2 + J_phi**2)
   
    # Smooth and detect local peaks
    J_mag_smooth = gaussian_filter(J_mag, sigma=2)
    local_peaks = (maximum_filter(J_mag_smooth, size=10) == J_mag_smooth) & (J_mag_smooth > np.percentile(J_mag_smooth, 99.5))
    peaks_y, peaks_x = np.where(local_peaks)
   
    # Visualization: Quantum energy streamlines + peak nodes
    plt.figure(figsize=(8, 6))
    plt.title(rf"Quantum Flow at $t={t}, \chi={chi_index}$ with Peaks")
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\theta$")
    plt.streamplot(
        np.linspace(0, 2*np.pi, 400),
        np.linspace(0, np.pi, 400),
        J_phi, J_theta,
        color=J_mag, linewidth=0.7, cmap='inferno', density=1.5
    )
    plt.scatter(
        peaks_x * 2*np.pi/400,
        peaks_y * np.pi/400,
        color='cyan', marker='o', s=30, label='Peaks'
    )
    plt.colorbar(label=r"$|\vec{p}_{\text{eff}}|$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"quantum_peaks_t{t}.png"))
    plt.close()

    # Save coordinates of peaks
    np.save(os.path.join(output_dir, f"peaks_coords_t{t}.npy"), np.stack([peaks_y, peaks_x], axis=1))
    print(f"✅ Saved peak map and coordinates at t={t}")