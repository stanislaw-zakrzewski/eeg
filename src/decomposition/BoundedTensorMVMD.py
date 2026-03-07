import numpy as np
# Ensure you have the base MVMD/TensorMVMD logic available


class BoundedTensorMVMD:
    def __init__(self, alpha=2000, tau=0, K=3, DC=0, tol=1e-7):
        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.tol = tol
        self.max_iter = 500

    def __call__(self, tensor_data, freq_bounds=None, fs=1000):
        """
        Args:
            tensor_data: (Trials, Channels, Time)
            freq_bounds: List of tuples [(min_hz, max_hz), ...]. Length must match K.
            fs: Sampling frequency (needed to convert Hz to normalized freq)
        """
        # 1. Validation
        if freq_bounds is None or len(freq_bounds) != self.K:
            raise ValueError(f"freq_bounds list must have length K={self.K}")

        # Convert Hz bounds to Normalized Freq (0 to 0.5)
        # bound_norm = [(8, 12)] -> [(0.008, 0.012)]
        bounds_norm = [(low / fs, high / fs) for low, high in freq_bounds]

        # 2. Tensor Unfolding (Trials*Channels, Time)
        Trials, Channels, Time = tensor_data.shape
        flattened_data = tensor_data.reshape(Trials * Channels, Time)

        # 3. Setup MVMD Internals
        C_flat, T = flattened_data.shape
        T_ext = 2 * T
        freqs = np.arange(0, T_ext // 2 + 1) / T_ext  # Frequency grid

        # Mirror extension & FFT
        f_mirrored = np.zeros((C_flat, T_ext))
        f_mirrored[:, :T // 2] = np.flip(flattened_data[:, :T // 2], axis=1)
        f_mirrored[:, T // 2:T // 2 + T] = flattened_data
        f_mirrored[:, T // 2 + T:] = np.flip(flattened_data[:, T // 2:], axis=1)
        f_hat = np.fft.fft(f_mirrored, axis=1)[:, :T_ext // 2 + 1]

        # 4. Initialization
        u_hat = np.zeros((self.K, C_flat, len(freqs)), dtype=complex)
        u_hat_old = np.zeros_like(u_hat)
        lam = np.zeros((C_flat, len(freqs)), dtype=complex)
        omega = np.zeros((self.max_iter, self.K))

        # Initialize Omega at the CENTER of the bounds
        for k in range(self.K):
            low, high = bounds_norm[k]
            omega[0, k] = (low + high) / 2

        # 5. Main ADMM Loop
        n = 0
        sum_diff = self.tol + 1

        while (sum_diff > self.tol) and (n < self.max_iter - 1):

            # --- Update Modes (Standard) ---
            for k in range(self.K):
                sum_uk = np.sum(np.delete(u_hat, k, axis=0), axis=0)
                denom = 1 + 2 * self.alpha * (freqs - omega[n, k]) ** 2
                u_hat[k, :, :] = (f_hat - sum_uk + lam / 2) / denom[None, :]

            # --- Update Center Frequencies (With Bounds) ---
            for k in range(self.K):
                # Calculate "Natural" center based on power spectrum
                power_spectrum = np.abs(u_hat[k, :, :]) ** 2
                total_power = np.sum(power_spectrum, axis=0)

                num = np.sum(freqs * total_power)
                den = np.sum(total_power)
                proposed_omega = num / den

                # --- APPLY CLIPPING ---
                low_b, high_b = bounds_norm[k]

                # If natural center is inside bounds, keep it.
                # If outside, snap it to the nearest edge.
                if proposed_omega < low_b:
                    omega[n + 1, k] = low_b
                elif proposed_omega > high_b:
                    omega[n + 1, k] = high_b
                else:
                    omega[n + 1, k] = proposed_omega

            # --- Update Lambda ---
            sum_u_all = np.sum(u_hat, axis=0)
            lam = lam + self.tau * (f_hat - sum_u_all)

            # --- Check Convergence ---
            diff = np.sum(np.abs(u_hat - u_hat_old) ** 2)
            norm = np.sum(np.abs(u_hat) ** 2)
            sum_diff = diff / norm if norm > 0 else 0
            u_hat_old = u_hat.copy()
            n += 1

        # 6. Refold Tensor
        u_hat_full = np.zeros((self.K, C_flat, T_ext), dtype=complex)
        u_hat_full[:, :, :T_ext // 2 + 1] = u_hat
        u_hat_full[:, :, T_ext // 2 + 1:] = np.conj(np.flip(u_hat[:, :, 1:T_ext // 2], axis=2))
        u_time = np.real(np.fft.ifft(u_hat_full, axis=2))
        u_final = u_time[:, :, T // 2: T // 2 + T]

        # Reshape to (K, Trials, Channels, Time)
        u_tensor = u_final.reshape(self.K, Trials, Channels, Time)

        return u_tensor, omega[:n, :]