import numpy as np
import os
import time

# پارامترها
critical_timesteps = [1, 2, 3, 9, 10, 11, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
n_coords = 4
n_chi, n_theta, n_phi = 400, 400, 400
block_size = 10

input_dir = "christoffel_output"
metric_dir = "metric_output"
output_dir = "ricci_output"
os.makedirs(output_dir, exist_ok=True)

dx = [1e-5, np.pi/n_chi, np.pi/n_theta, 2*np.pi/n_phi]

def grad(f, h, axis):
    return np.gradient(f, h, axis=axis, edge_order=2)

for t in critical_timesteps:
    start_time = time.time()
    print(f"🔄 Computing Ricci tensor for t={t}...")

    Gamma = np.memmap(os.path.join(input_dir, f"Gamma_t{t}.npy"),
                  dtype=np.float32, mode='r',
                  shape=(4, 4, 4, 400, 400, 400))
    ginv = np.load(os.path.join(metric_dir, f"g_t{t}.npy"), mmap_mode='r', allow_pickle=True)

    Ricci = np.memmap(os.path.join(output_dir, f"Ricci_t{t}.npy"), dtype=np.float32, mode='w+',
                      shape=(n_coords, n_coords, n_chi, n_theta, n_phi))
    Rscalar = np.memmap(os.path.join(output_dir, f"Rscalar_t{t}.npy"), dtype=np.float32, mode='w+',
                        shape=(n_chi, n_theta, n_phi))

    for i_start in range(0, n_chi, block_size):
        i_end = min(i_start + block_size, n_chi)

        for mu in range(n_coords):
            for nu in range(n_coords):
                term1 = np.zeros((i_end - i_start, n_theta, n_phi), dtype=np.float32)
                term2 = np.zeros_like(term1)
                term3 = np.zeros_like(term1)
                term4 = np.zeros_like(term1)

                for lam in range(n_coords):
                    dG_lam = grad(Gamma[lam, mu, nu, i_start:i_end], dx[lam], axis=0)
                    term1 += dG_lam

                    dG_nu = grad(Gamma[lam, mu, lam, i_start:i_end], dx[nu], axis=1)
                    term2 += dG_nu

                    for sig in range(n_coords):
                        g1 = Gamma[lam, mu, nu, i_start:i_end]
                        g2 = Gamma[sig, lam, sig, i_start:i_end]
                        g3 = Gamma[sig, mu, lam, i_start:i_end]
                        g4 = Gamma[lam, nu, sig, i_start:i_end]

                        term3 += g1 * g2
                        term4 += g3 * g4

                Ricci[mu, nu, i_start:i_end] = term1 - term2 + term3 - term4

        # اسکالر ریچی در این بلوک
        for i_local, i in enumerate(range(i_start, i_end)):
            for j in range(n_theta):
                for k in range(n_phi):
                    Rscalar[i, j, k] = np.tensordot(
                        ginv[..., i, j, k],
                        Ricci[..., i, j, k],
                        axes=2
                    )

    duration = time.time() - start_time
    print(f"✅ Ricci and scalar R saved for t={t} ⏱️ Time: {duration:.2f} sec")