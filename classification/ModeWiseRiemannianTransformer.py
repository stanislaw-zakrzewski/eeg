import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


class ModeWiseRiemannianTransformer:
    def __init__(self, n_modes):
        """
        Creates independent Riemannian Manifolds for each VMD Mode.
        """
        self.n_modes = n_modes
        self.cov_estimator = Covariances(estimator='oas')

        # A list of independent Tangent Spaces (one per mode)
        self.tangent_spaces = [TangentSpace() for _ in range(n_modes)]
        self.is_fitted = False

    def _get_mode_data(self, X, mode_idx):
        """
        Extracts all data for a specific mode and flattens Batch/Seq.
        Input X: (Trials, Seq, Modes, Chan, Time)
        """
        # Select specific mode -> (Trials, Seq, Chan, Time)
        mode_data = X[:, :, mode_idx, :, :]

        # Flatten Trials and Seq -> (N_samples, Chan, Time)
        B, S, C, T = mode_data.shape
        return mode_data.reshape(B * S, C, T), (B, S)

    def fit(self, X_train):
        """
        Fits K separate Mean Covariance Matrices.
        """
        # Check input dims
        if X_train.ndim != 5:
            raise ValueError("Expected input shape (Trials, Seq, Modes, Chan, Time)")

        print(f"Fitting independent manifolds for {self.n_modes} modes...")

        for k in range(self.n_modes):
            # 1. Get data for just this mode
            X_mode, _ = self._get_mode_data(X_train, k)

            # 2. Estimate Covariances
            covs = self.cov_estimator.fit_transform(X_mode)

            # 3. Fit the Tangent Space for this mode
            self.tangent_spaces[k].fit(covs)

        self.is_fitted = True
        print("Done.")

    def transform(self, X_data):
        """
        Transforms each mode using its specific corresponding Manifold.
        """
        if not self.is_fitted:
            raise RuntimeError("Not fitted!")

        B, S, K, C, T = X_data.shape
        if K != self.n_modes:
            raise ValueError(f"Data has {K} modes but transformer expects {self.n_modes}")

        # List to hold features for each mode
        mode_features_list = []

        for k in range(self.n_modes):
            # 1. Get data
            X_mode, (batch, seq) = self._get_mode_data(X_data, k)

            # 2. Covariance
            covs = self.cov_estimator.transform(X_mode)

            # 3. Project using the Specific Tangent Space for Mode k
            # This uses the Mean Covariance stored in self.tangent_spaces[k]
            vecs = self.tangent_spaces[k].transform(covs)

            # vecs shape: (Batch*Seq, n_features)
            # Reshape to (Batch, Seq, 1, n_features) to stack later
            vecs = vecs.reshape(batch, seq, 1, -1)
            mode_features_list.append(vecs)

        # Concatenate along the Mode dimension (dim 2)
        # Result: (Batch, Seq, Modes, Features)
        return np.concatenate(mode_features_list, axis=2)