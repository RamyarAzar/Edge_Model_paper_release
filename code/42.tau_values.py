import numpy as np

# مقدار ثابت پلانک (h) در SI
h_planck = 6.62607015e-34  # J·s

# بارگذاری Q(t), τ(t)
q_values = np.load("q_quantum_invariant.npy")           # (اندازه 9)
tau_values = np.load("tau_values.npy")                  # (اندازه 9)

# محاسبه λ(t)
lambda_values = np.sqrt(h_planck / (q_values * tau_values))

# ذخیره نتایج
np.save("lambda_normalization.npy", lambda_values)

# نمایش خروجی
for i, lam in enumerate(lambda_values, start=33):
    print(f"t={i} | λ = {lam:.3e}")