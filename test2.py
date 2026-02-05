import numpy as np
import matplotlib.pyplot as plt
from sktime.libs.vmdpy import VMD # Standard VMD for comparison
from  mvmd_2 import MVMD
from subject import Subject

subject_edf_path = 'C:/Users/stz/Documents/GitHub/csp_classifier/preprocessed_subjects/s01.edf'
subject = Subject(subject_edf_path)
raw = subject.raw
raw = raw.filter(l_freq=5, h_freq=25, verbose='ERROR')
signal = raw.get_data()
sss = [signal[12],signal[49]]
sss2 = []
for i in range(len(sss)):
    sss2.append(sss[i][512*2:512*5])
# 1. Setup Synthetic Data
fs = 1000
T = 1
t = np.linspace(0, T, T*fs)

# Common Alpha Wave (12 Hz)
common_alpha = np.cos(2 * np.pi * 12 * t)

# Channel 1: Alpha (12Hz) + Theta (6Hz)
ch1 = common_alpha + 0.5 * np.cos(2 * np.pi * 6 * t)

# Channel 2: Alpha (12Hz) + Gamma (40Hz)
ch2 = common_alpha + 0.5 * np.cos(2 * np.pi * 40 * t) + 0.7 * np.cos(2 * np.pi * 24 * t)

# Stack for MVMD
signal_matrix = np.vstack([ch1, ch2]) # Shape (2, 1000)
signal_matrix = np.vstack([sss2[0], sss2[1]]) # Shape (2, 1000)

# ==========================================
# APPROACH A: Standard VMD (Channel by Channel)
# ==========================================
print("Running Standard VMD...")
# VMD on Ch1
u1_vmd, _, _ = VMD(ch1, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7)
# VMD on Ch2
u2_vmd, _, _ = VMD(ch2, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7)

# ==========================================
# APPROACH B: MVMD (Joint Decomposition)
# ==========================================
print("Running Multivariate VMD...")
mvmd = MVMD(alpha=2000, tau=0, K=4, init=1)
u_mvmd, omega_history = mvmd(signal_matrix)

# u_mvmd shape is (K, Channels, Time) -> (3, 2, 1000)

# ==========================================
# VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 8))

# --- Row 1: VMD (Misaligned) ---
plt.subplot(2, 2, 1)
plt.plot(u1_vmd[1, :], 'b')
plt.title("VMD Ch1 - Mode 2 (Detected: Alpha 12Hz)")
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(u2_vmd[1, :], 'r')
plt.title("VMD Ch2 - Mode 2 (Detected: Gamma 40Hz!)") # MISMATCH
plt.grid(True, alpha=0.3)

# --- Row 2: MVMD (Aligned) ---
plt.subplot(2, 2, 3)
plt.plot(u_mvmd[1, 0, :], 'b') # Mode 1 (Index 1), Channel 0
plt.title("MVMD Ch1 - Mode 2 (Forced: Alpha 12Hz)")
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(u_mvmd[1, 1, :], 'r') # Mode 1 (Index 1), Channel 1
plt.title("MVMD Ch2 - Mode 2 (Forced: Alpha 12Hz)") # ALIGNED
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print Final Center Frequencies
print(f"\nMVMD Final Center Frequencies (Hz): {omega_history[-1, :] * fs}")

fig, ax = plt.subplots(2, 4)
fig.set_size_inches(18.5, 10.5, forward=True)
ax[0][0].plot(u1_vmd[0,  :], 'b')
ax[0][0].set_title('VMD 1 MOD 1')
ax[0][1].plot(u1_vmd[1, :], 'b')
ax[0][1].set_title('VMD 1 MOD 2')
ax[0][2].plot(u2_vmd[0,  :], 'g')
ax[0][2].set_title('VMD 2 MOD 1')
ax[0][3].plot(u2_vmd[1,  :], 'g')
ax[0][3].set_title('VMD 2 MOD 2')
ax[1][0].plot(u_mvmd[2, 0, :], 'r')
ax[1][0].set_title('MVMD 1 MOD 1')
ax[1][1].plot(u_mvmd[3, 0, :], 'r')
ax[1][1].set_title('MVMD 1 MOD 2')
ax[1][2].plot(u_mvmd[2, 1, :], 'r')
ax[1][2].set_title('MVMD 2 MOD 1')
ax[1][3].plot(u_mvmd[3, 1, :], 'r')
ax[1][3].set_title('MVMD 2 MOD 2')

plt.show()