import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


class RiemannianTransformer:
    def __init__(self):
        """
        Stateful transformer.
        Fits a Global Mean Covariance on Training Data.
        Projects all data (Train/Test) to the Tangent Space of that Mean.
        """
        # OAS is generally more robust than standard empirical covariance
        self.cov_estimator = Covariances(estimator='oas')
        self.tangent_space = TangentSpace()
        self.is_fitted = False

    def _flatten_input(self, X):
        """
        Helper: Flattens (Batch, Seq, Modes, Chan, Time) -> (N, Chan, Time)
        Returns: Flattened X, and the original shape tuple to restore later.
        """
        original_shape = X.shape

        # Case 5D: (Trials, Sequence, Modes, Channels, Time)
        if X.ndim == 5:
            B, S, K, C, T = X.shape
            # Collapse first 3 dims into one "Sample" dim
            X_flat = X.reshape(B * S * K, C, T)

        # Case 4D: (Trials, Modes, Channels, Time) - if Sequence=1 was squeezed
        elif X.ndim == 4:
            B, K, C, T = X.shape
            X_flat = X.reshape(B * K, C, T)

        # Case 3D: (Trials, Channels, Time) - Standard PyRiemann input
        elif X.ndim == 3:
            X_flat = X

        else:
            raise ValueError(f"Input dimension {X.ndim} not supported. Expected 3, 4, or 5.")

        return X_flat, original_shape

    def fit(self, X_train):
        """
        Calculates the Reference Mean Covariance from X_train.
        Input: (Trials, Sequence, Modes, Channels, Time)
        """
        # 1. Flatten everything to a list of matrices
        X_flat, _ = self._flatten_input(X_train)

        # 2. Estimate Covariances
        # Shape: (N_total_samples, C, C)
        covs = self.cov_estimator.fit_transform(X_flat)

        # 3. Fit Tangent Space (Computes the Geometric Mean)
        self.tangent_space.fit(covs)

        self.is_fitted = True
        print(f"Riemannian fitted on {X_flat.shape[0]} matrices.")

    def transform(self, X_data):
        """
        Projects X_data using the Reference Mean from .fit()
        Input: (Trials, Sequence, Modes, Channels, Time)
        Output: (Trials, Sequence, Modes, Features)
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer not fitted! Call .fit(train_data) first.")

        # 1. Flatten
        X_flat, original_shape = self._flatten_input(X_data)

        # 2. Estimate Covariances
        covs = self.cov_estimator.transform(X_flat)

        # 3. Project to Tangent Space (Using saved Mean)
        # Shape: (N_total_samples, n_features)
        tangent_vecs = self.tangent_space.transform(covs)

        # 4. Restore Dimensions
        # We need to reshape back, but the last 2 dims (C, T) are now replaced by (Features)
        n_features = tangent_vecs.shape[1]

        if len(original_shape) == 5:
            B, S, K, C, T = original_shape
            return tangent_vecs.reshape(B, S, K, n_features)

        elif len(original_shape) == 4:
            B, K, C, T = original_shape
            return tangent_vecs.reshape(B, K, n_features)

        else:
            return tangent_vecs