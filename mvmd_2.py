import numpy as np


class MVMD:
    def __init__(self, alpha=2000, tau=0, K=5, DC=0, init=1, tol=1e-7, max_iter=500):
        """
        Multivariate Variational Mode Decomposition (MVMD)

        Parameters:
        - alpha: Bandwidth constraint (lower = wider bandwidths)
        - tau: Noise tolerance (0 = strict fidelity to signal)
        - K: Number of modes to extract
        - DC: 0 (no DC part), 1 (DC part included)
        - init: 1 (Uniform initialization of omegas), 2 (Random)
        - tol: Convergence tolerance
        """
        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.init = init
        self.tol = tol
        self.max_iter = max_iter

    def __call__(self, signal):
        """
        Input:
            signal: numpy array of shape (Channels, Time)
        Returns:
            u: Decomposed modes (K, Channels, Time)
            omega: Center frequencies (Iterations, K)
        """
        C, T = signal.shape

        # 1. Mirror extension to handle boundary effects
        T_ext = 2 * T
        fs = 1 / T

        f_mirrored = np.zeros((C, T_ext))
        f_mirrored[:, :T // 2] = np.flip(signal[:, :T // 2], axis=1)
        f_mirrored[:, T // 2:T // 2 + T] = signal
        f_mirrored[:, T // 2 + T:] = np.flip(signal[:, T // 2:], axis=1)

        # 2. Time domain to Frequency domain
        # Construct frequency vector (0 to 0.5)
        freqs = np.arange(0, T_ext // 2 + 1) / T_ext

        # FFT of signals
        f_hat = np.fft.fft(f_mirrored, axis=1)
        f_hat = f_hat[:, :T_ext // 2 + 1]  # Keep positive freqs

        # 3. Initialization
        # u_hat: (K, C, Freqs)
        u_hat = np.zeros((self.K, C, len(freqs)), dtype=complex)
        u_hat_old = np.zeros_like(u_hat)

        # lambda (dual variables): (C, Freqs)
        lam = np.zeros((C, len(freqs)), dtype=complex)

        # omega: (Iterations, K)
        omega = np.zeros((self.max_iter, self.K))

        if self.init == 1:
            omega[0, :] = np.arange(1, self.K + 1) / self.K / 2
        else:
            omega[0, :] = np.random.rand(self.K) / 2

        # 4. Main ADMM Loop
        n = 0
        sum_diff = self.tol + 1

        while (sum_diff > self.tol) and (n < self.max_iter - 1):

            # --- Update Modes (u_hat) ---
            for k in range(self.K):
                # Sum of all OTHER modes (excluding k)
                sum_uk = np.sum(np.delete(u_hat, k, axis=0), axis=0)

                # Spectrum residue: Signal - Sum(Other Modes) + Lambda/2
                # We handle the denominator (1 + 2*alpha*(w - w_k)^2)
                denom = 1 + 2 * self.alpha * (freqs - omega[n, k]) ** 2

                # Update u_hat for all channels simultaneously
                # shape broadcasting: (C, F) - (C, F) + (C, F) / (F,)
                u_hat[k, :, :] = (f_hat - sum_uk + lam / 2) / denom[None, :]

            # --- Update Center Frequencies (omega) ---
            # This is the Key Difference from VMD:
            # Omega is weighted average across ALL channels
            for k in range(self.K):
                # Energy of mode k across all channels
                # shape: (C, F) -> squared magnitude
                power_spectrum = np.abs(u_hat[k, :, :]) ** 2

                # Sum power across channels
                total_power = np.sum(power_spectrum, axis=0)  # shape (F,)

                # Center of gravity
                num = np.sum(freqs * total_power)
                den = np.sum(total_power)
                omega[n + 1, k] = num / den

            # --- Update Lagrange Multiplier (lambda) ---
            # Sum of all modes
            sum_u_all = np.sum(u_hat, axis=0)

            # Dual ascent
            lam = lam + self.tau * (f_hat - sum_u_all)

            # --- Check Convergence ---
            # Sum of squared differences between updates
            diff = np.sum(np.abs(u_hat - u_hat_old) ** 2)
            norm = np.sum(np.abs(u_hat) ** 2)
            sum_diff = diff / norm if norm > 0 else 0

            u_hat_old = u_hat.copy()
            n += 1

        # 5. Reconstruction (Inverse FFT)
        u_hat_full = np.zeros((self.K, C, T_ext), dtype=complex)
        u_hat_full[:, :, :T_ext // 2 + 1] = u_hat
        # Hermitian symmetry for real signal reconstruction
        u_hat_full[:, :, T_ext // 2 + 1:] = np.conj(np.flip(u_hat[:, :, 1:T_ext // 2], axis=2))

        u_time = np.real(np.fft.ifft(u_hat_full, axis=2))

        # Remove mirror padding
        u_final = u_time[:, :, T // 2: T // 2 + T]

        # Return: Modes (K, Channels, Time), Frequencies
        return u_final, omega[:n, :]