import numpy as np
import matplotlib.pyplot as plt
from sktime.libs.vmdpy import VMD
from subject import Subject

subject_edf_path = 'C:/Users/stz/Documents/GitHub/csp_classifier/preprocessed_subjects/s01.edf'
subject = Subject(subject_edf_path)
signal = subject.raw.get_data()
sss = signal[4][0:2000]

# Time Domain 0 to T
T = 1000
fs = 1 / T
t = np.arange(1, T + 1) / T
freqs = 2 * np.pi * (t - 0.5 - fs) / (fs)

# center frequencies of components
f_1 = 2
f_2 = 24
f_3 = 288

# modes
v_1 = np.cos(2 * np.pi * f_1 * t)
v_2 = 1 / 4 * (np.cos(2 * np.pi * f_2 * t))
v_3 = 1 / 16 * (np.cos(2 * np.pi * f_3 * t))

f = v_1 + v_2 + v_3 + 0.1 * np.random.randn(v_1.size)

# some sample parameters for VMD
alpha = 2000  # moderate bandwidth constraint
tau = 0.0  # noise-tolerance (no strict fidelity enforcement)
K = 5  # 3 modes
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7

# Run VMD
u, u_hat, omega = VMD(sss, alpha, tau, K, DC, init, tol)

# Visualize decomposed modes
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(sss)
plt.title("Original signal")
plt.xlabel("time (s)")
plt.subplot(2, 1, 2)
plt.plot(u.T)
plt.title("Decomposed modes")
plt.xlabel("time (s)")
plt.legend(["Mode %d" % m_i for m_i in range(u.shape[0])])
plt.tight_layout()
plt.show()