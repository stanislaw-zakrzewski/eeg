import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.geodesic import geodesic_riemann


class DomainAwareRiemannianTransformer:
    def __init__(self, n_modes):
        self.n_modes = n_modes
        self.cov_estimator = Covariances(estimator='oas')
        self.tangent_spaces = [TangentSpace() for _ in range(n_modes)]

        # Store the Training Means to define the target "Center"
        self.training_means = [None] * n_modes
        self.is_fitted = False

    def _get_mode_data(self, X, mode_idx):
        # Extracts data for one mode and flattens Batch/Seq
        mode_data = X[:, :, mode_idx, :, :]
        B, S, C, T = mode_data.shape
        return mode_data.reshape(B * S, C, T), (B, S)

    def fit(self, X_train):
        """
        Step 1: Fit on Source Subject (Training Data).
        """
        print(f"Fitting Source Domain (Train) for {self.n_modes} modes...")

        for k in range(self.n_modes):
            # Get Data
            X_mode, _ = self._get_mode_data(X_train, k)

            # Estimate Covariances
            covs = self.cov_estimator.fit_transform(X_mode)

            # Calculate and Store Training Mean
            M_train = mean_covariance(covs, metric='riemann')
            self.training_means[k] = M_train

            # Fit Tangent Space relative to this Mean
            self.tangent_spaces[k].fit(covs)

        self.is_fitted = True

    def transform(self, X_test, calibration_data=None):
        """
        Step 2: Transform Target Subject (Test Data).

        CRITICAL: If 'calibration_data' is provided, we use it to
        Re-Center the test data to align with the Training Space.
        """
        if not self.is_fitted:
            raise RuntimeError("Run .fit(X_train) first!")

        B, S, K, C, T = X_test.shape
        mode_features_list = []

        for k in range(self.n_modes):
            # A. Prepare Test Data
            X_mode_test, (b_test, s_test) = self._get_mode_data(X_test, k)
            covs_test = self.cov_estimator.transform(X_mode_test)

            # B. DOMAIN ADAPTATION (Re-Centering)
            # If we have calibration data, we use it to calculate the "Shift"
            if calibration_data is not None:
                # 1. Get Calibration Data for this mode
                X_cal, _ = self._get_mode_data(calibration_data, k)
                covs_cal = self.cov_estimator.transform(X_cal)

                # 2. Calculate Mean of New Subject (Target Mean)
                M_target = mean_covariance(covs_cal, metric='riemann')

                # 3. Parallel Transport: Test Data -> Identity -> Train Mean
                # Actually, simpler approach: Project Test Data using Target Mean
                # This puts Test Data into the same Tangent Space (centered at Identity)
                # as the Training Data (if we force Training TS to reference Identity).

                # BETTER APPROACH FOR CLASSIFIERS:
                # We simply set the Reference Point of the Tangent Space
                # to be the NEW SUBJECT'S MEAN ($M_target$).
                # This aligns the new subject's "Baseline" to the "Origin" (0,0) of the TS.

                # Since the Classifier expects "Deviation from Mean",
                # we calculate deviation from *Target* Mean.
                self.tangent_spaces[k].reference_ = M_target

            else:
                # Fallback: Use Training Mean (Will cause 50% acc if subjects differ)
                self.tangent_spaces[k].reference_ = self.training_means[k]

            # C. Project
            vecs = self.tangent_spaces[k].transform(covs_test)

            # Reshape
            vecs = vecs.reshape(b_test, s_test, 1, -1)
            mode_features_list.append(vecs)

        return np.concatenate(mode_features_list, axis=2)