import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.decomposition.BoundedTensorMVMD import BoundedTensorMVMD
from src.decomposition.FixedMVMD import FixedMVMD


class MVMD2(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=100000, tau=0, K=5, DC=0, init=1, tol=1e-7, max_iter=500, fs=512):
        """
        Multivariate Variational Mode Decomposition (MVMD)

        Parameters:
        - alpha: Bandwidth constraint (lower = wider bandwidths)
        - tau: Noise tolerance (0 = strict fidelity to signal)
        - K: Number of modes to extract
        - DC: 0 (no DC part), 1 (DC part included)
        - init: 1 (Uniform initialization of omegas), 2 (Random)
        - tol: Convergence tolerance
        - max_iter: Maximum number of iterations
        """
        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.bounded_tensor_MVMD = BoundedTensorMVMD(alpha=alpha, K=K)
        self.fixed_MVMD = FixedMVMD(alpha=alpha, K=K)
        self.mode_frequency_centers = []
        self.fs = fs

    def fit(self, X, y=None):
        """
        Fit the model.

        Parameters:
        - X: Input data
        - y: Target values (unused)

        Returns:
        - self
        """
        modes_3d, omegas_3d = self.bounded_tensor_MVMD(X,
                                                  freq_bounds=[(6,28) for _ in range(self.K)], fs=self.fs)
        self.mode_frequency_centers = (omegas_3d[-1, :] * self.fs).flatten()
        return self

    def transform(self, X):
        """
        Transform the data.

        Parameters:
        - X: Input data of shape (n_epochs, n_channels, n_times)

        Returns:
        - X_transformed: Transformed data of shape (n_epochs, K * n_channels, n_times)
        """
        new_x_test = []
        for test_trial_idx, test_trial in enumerate(X):
            modes, omegas = self.fixed_MVMD(test_trial, fixed_freqs=self.mode_frequency_centers, fs=self.fs)
            new_x_test.append(modes)
        new_x_test = np.array(new_x_test)
        new_x_test = new_x_test.reshape(new_x_test.shape[0], new_x_test.shape[1] * new_x_test.shape[2], new_x_test.shape[3])
        return new_x_test
