import numpy as np
from src.decomposition.MVMD import MVMD


# Ensure you have the previous MVMD class available
# from mvmd import MVMD

class FixedMVMD(MVMD):
    def __call__(self, signal, fixed_freqs=None, fs=1000):
        """
        Args:
            signal: (Channels, Time)
            fixed_freqs: List of center frequencies in Hz (e.g., [10, 20, 30])
        """
        # 1. Validation
        if fixed_freqs is None:
            raise ValueError("For FixedMVMD, you must provide 'fixed_freqs' list.")

        if len(fixed_freqs) != self.K:
            print(f"Warning: adjusting K from {self.K} to {len(fixed_freqs)} to match input list.")
            self.K = len(fixed_freqs)

        # 2. Setup (Same as parent)
        C, T = signal.shape
        T_ext = 2 * T

        # Create frequency grid
        freqs = np.arange(0, T_ext // 2 + 1) / T_ext

        # Mirror extension & FFT (Same as parent)
        f_mirrored = np.zeros((C, T_ext))
        f_mirrored[:, :T // 2] = np.flip(signal[:, :T // 2], axis=1)
        f_mirrored[:, T // 2:T // 2 + T] = signal
        f_mirrored[:, T // 2 + T:] = np.flip(signal[:, T // 2:], axis=1)
        f_hat = np.fft.fft(f_mirrored, axis=1)[:, :T_ext // 2 + 1]

        # 3. Initialization
        u_hat = np.zeros((self.K, C, len(freqs)), dtype=complex)
        u_hat_old = np.zeros_like(u_hat)
        lam = np.zeros((C, len(freqs)), dtype=complex)

        # --- KEY CHANGE 1: FORCE INITIAL FREQUENCIES ---
        # Normalize Hz inputs to 0-0.5 scale
        # If user passes 10Hz and Fs=1000, normalized is 10/1000 = 0.01
        # NOTE: We assume the user provides Frequencies normalized (0 to 0.5)
        # OR we need the Sampling Rate. Let's assume normalized for the math,
        # but I will add an 'fs' arg to the class for ease of use.

        omega = np.zeros((self.max_iter, self.K))
        omega[:] = np.array(fixed_freqs)  # Set ALL rows to the fixed values

        # 4. Main ADMM Loop
        n = 0
        sum_diff = self.tol + 1

        while (sum_diff > self.tol) and (n < self.max_iter - 1):

            # --- Update Modes (u_hat) ---
            for k in range(self.K):
                sum_uk = np.sum(np.delete(u_hat, k, axis=0), axis=0)
                # Use omega[n, k] which is FIXED
                denom = 1 + 2 * self.alpha * (freqs - omega[n, k]) ** 2
                u_hat[k, :, :] = (f_hat - sum_uk + lam / 2) / denom[None, :]

            # --- KEY CHANGE 2: SKIP OMEGA UPDATE ---
            # We simply copy the current omega to the next step
            omega[n + 1, :] = omega[n, :]

            # --- Update Lambda ---
            sum_u_all = np.sum(u_hat, axis=0)
            lam = lam + self.tau * (f_hat - sum_u_all)

            # --- Check Convergence ---
            diff = np.sum(np.abs(u_hat - u_hat_old) ** 2)
            norm = np.sum(np.abs(u_hat) ** 2)
            sum_diff = diff / norm if norm > 0 else 0
            u_hat_old = u_hat.copy()
            n += 1

        # 5. Reconstruction (Same as parent)
        u_hat_full = np.zeros((self.K, C, T_ext), dtype=complex)
        u_hat_full[:, :, :T_ext // 2 + 1] = u_hat
        u_hat_full[:, :, T_ext // 2 + 1:] = np.conj(np.flip(u_hat[:, :, 1:T_ext // 2], axis=2))
        u_time = np.real(np.fft.ifft(u_hat_full, axis=2))
        u_final = u_time[:, :, T // 2: T // 2 + T]

        return u_final, omega[:n, :]