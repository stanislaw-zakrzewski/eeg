import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance


class AdaptiveRiemannianTransformer:
    def __init__(self, adaptation_rate=0.01):
        """
        Args:
            adaptation_rate: How fast to update the mean (0.01 = slow drift, 0.1 = fast).
        """
        self.cov_estimator = Covariances(estimator='oas')
        self.tangent_space = TangentSpace()
        self.reference_mean = None
        self.adaptation_rate = adaptation_rate
        self.is_fitted = False

    def _flatten_input(self, X):
        # ... (Same flattening logic as before) ...
        # Copy paste the _flatten_input method from previous turn
        original_shape = X.shape
        if X.ndim == 5:
            B, S, K, C, T = X.shape
            X_flat = X.reshape(B * S * K, C, T)
        elif X.ndim == 4:
            B, K, C, T = X.shape
            X_flat = X.reshape(B * K, C, T)
        elif X.ndim == 3:
            X_flat = X
        return X_flat, original_shape

    def fit(self, X_train):
        """Initial fit on Training Data."""
        X_flat, _ = self._flatten_input(X_train)
        covs = self.cov_estimator.fit_transform(X_flat)

        # Calculate initial geometric mean
        self.reference_mean = mean_covariance(covs, metric='riemann')
        self.tangent_space.reference_ = self.reference_mean  # Set internal ref

        self.is_fitted = True
        print("Initial Mean Fitted.")

    def adapt(self, X_new):
        """
        Call this with new incoming data to slightly shift the Reference Mean.
        This fixes the "Calibration Drift" problem.
        """
        if not self.is_fitted: return

        X_flat, _ = self._flatten_input(X_new)
        new_covs = self.cov_estimator.transform(X_flat)

        # Calculate mean of the NEW small batch (or single trial)
        current_batch_mean = mean_covariance(new_covs, metric='riemann')

        # Geodesic interpolation (EMA on the Manifold)
        # New_Ref = Old_Ref * (1-alpha) + Current * alpha
        # Ideally we use geodesic_interpolation, but a weighted average in Euclidian
        # is often a "good enough" approximation for small updates,
        # OR we re-calculate mean using a buffer.

        # ROBUST STRATEGY: Weighted Geodesic Mean
        from pyriemann.utils.geodesic import geodesic_riemann

        # Move the reference mean 1% towards the new data's mean
        self.reference_mean = geodesic_riemann(self.reference_mean, current_batch_mean, self.adaptation_rate)

        # Update the Tangent Space projector
        self.tangent_space.reference_ = self.reference_mean

    def transform(self, X_data):
        """Standard Transform using current Reference Mean"""
        X_flat, original_shape = self._flatten_input(X_data)
        covs = self.cov_estimator.transform(X_flat)

        # Explicitly force the Tangent Space to use our Adaptive Mean
        self.tangent_space.reference_ = self.reference_mean
        tangent_vecs = self.tangent_space.transform(covs)

        # Restore shape
        n_features = tangent_vecs.shape[1]
        if len(original_shape) == 5:
            B, S, K, C, T = original_shape
            return tangent_vecs.reshape(B, S, K, n_features)
        elif len(original_shape) == 4:
            B, K, C, T = original_shape
            return tangent_vecs.reshape(B, K, n_features)
        else:
            return tangent_vecs