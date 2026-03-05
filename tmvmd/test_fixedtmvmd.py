import numpy as np
from decomposition.FixedMVMD import FixedMVMD
import matplotlib.pyplot as plt

# 1. Generate Signal with offset frequencies
# Real signal has 11 Hz and 19 Hz (slightly off from our target)
fs = 1000
t = np.linspace(0, 1, 1000)
signal = np.cos(2*np.pi*6*t) + 0.5*np.cos(2*np.pi*16*t)
signal = signal.reshape(1, -1) # Make it (1, 1000)

# 2. Define Targets (Normalized Frequency)
# Target Freq / Fs
target_hz = [5, 16]
target_norm = [f/fs for f in target_hz]

# 3. Run Fixed VMD
# We use a lower alpha (bandwidth constraint) to allow some "wiggling"
# around the center frequency, or high alpha to be very strict.
vmd_fixed = FixedMVMD(alpha=10000000, tau=0, K=2)

modes, omegas = vmd_fixed(signal, fixed_freqs=target_norm)

print(f"Requested Frequencies: {target_hz}")
print(f"Final Output Frequencies: {omegas[-1, :] * fs}")
plt.plot(signal[0])
plt.plot(modes[0][0])
plt.plot(modes[1][0])
plt.show()
# Output will be EXACTLY [10. 20.]