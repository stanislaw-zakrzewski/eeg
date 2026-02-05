import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Time
# Create a time array from 0 to 4 seconds with 1000 data points
t = np.linspace(0, 4, 1000)

# 2. Define the Base Carrier Wave parameters
# High frequency oscillation (the "fast" wave)
f_carrier = 10  # 10 Hz
carrier_wave = np.sin(2 * np.pi * f_carrier * t)

# --- SCENARIO A: Amplitude Modulation (AM) ---
# The amplitude changes based on a slower sine wave
f_mod = 1.0  # 1 Hz modulation
# We add 1 so the amplitude doesn't go negative (keeps the phase consistent)
amplitude_envelope_am = 1 + 0.5 * np.sin(2 * np.pi * f_mod * t)
signal_am = amplitude_envelope_am * carrier_wave

# --- SCENARIO B: Damped Oscillation ---
# The amplitude decays exponentially over time
decay_rate = 1.0
amplitude_envelope_damped = np.exp(-decay_rate * t)
signal_damped = amplitude_envelope_damped * carrier_wave

# 3. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot AM Signal
ax1.set_title('Scenario 1: Amplitude Modulation (Pulsating)')
ax1.plot(t, signal_am, label='Modulated Signal', color='blue', linewidth=1.5)
# Plot the "Envelope" (visual guide of amplitude)
ax1.plot(t, amplitude_envelope_am, 'r--', label='Amplitude Envelope', linewidth=2)
ax1.plot(t, -amplitude_envelope_am, 'r--', linewidth=2) # Negative envelope
ax1.set_ylabel('Amplitude')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot Damped Signal
ax2.set_title('Scenario 2: Damped Oscillation (Decaying)')
ax2.plot(t, signal_damped, label='Damped Signal', color='green', linewidth=1.5)
# Plot the "Envelope"
ax2.plot(t, amplitude_envelope_damped, 'r--', label='Amplitude Envelope', linewidth=2)
ax2.plot(t, -amplitude_envelope_damped, 'r--', linewidth=2)
ax2.set_ylabel('Amplitude')
ax2.set_xlabel('Time (s)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()