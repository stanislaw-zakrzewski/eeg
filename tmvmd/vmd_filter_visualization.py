import numpy as np
import matplotlib.pyplot as plt

# Setup
fs = 1000
freqs = np.linspace(0, 20, 1000) # Plot 0 to 20 Hz
target_freq = 10
interference_freq = 11

# Normalized frequency required for the formula
w = freqs / fs
w_k = target_freq / fs

# Define the VMD Filter Transfer Function
def vmd_filter(w, w_k, alpha):
    # This is the magnitude response of the VMD mode
    return 1.0 / (1.0 + 2 * alpha * (w - w_k)**2)

# Calculate responses
response_low = vmd_filter(w, w_k, alpha=2000)
response_mid = vmd_filter(w, w_k, alpha=50000)
response_high = vmd_filter(w, w_k, alpha=10000000)

# Plot
plt.figure(figsize=(10, 6))

plt.plot(freqs, response_low, label='Alpha = 2,000 (Standard)', linestyle='--')
plt.plot(freqs, response_mid, label='Alpha = 50,000', linestyle='-.')
plt.plot(freqs, response_high, label='Alpha = 10,000,000 (User)', linewidth=2, color='red')

# Add markers for signal vs noise
plt.axvline(target_freq, color='green', alpha=0.3, label='Target (10Hz)')
plt.axvline(interference_freq, color='black', alpha=0.3, label='Interference (11Hz)')

plt.title("Why you need Alpha=10,000,000 for 1Hz Separation")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Filter Magnitude (0-1)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(8, 13)
plt.show()