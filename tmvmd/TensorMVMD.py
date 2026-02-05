import numpy as np
from mvmd_2 import MVMD


# Assuming the previous MVMD class is saved or available
# from mvmd import MVMD

class TensorMVMD:
    def __init__(self, alpha=2000, tau=0, K=5, DC=0, init=1, tol=1e-7):
        """
        3D Tensor Wrapper for MVMD.
        Input: (Trials, Channels, Time)
        Output: (K, Trials, Channels, Time)
        """
        # We use the standard 2D MVMD engine
        self.mvmd_engine = MVMD(alpha, tau, K, DC, init, tol)

    def __call__(self, tensor_data):
        """
        Args:
            tensor_data: Numpy array of shape (Trials, Channels, Time)
        Returns:
            u_tensor: Decomposed modes (K, Trials, Channels, Time)
            omega: Center frequencies history
        """
        # 1. Capture dimensions
        Trials, Channels, Time = tensor_data.shape

        # 2. Tensor Unfolding (Matricization)
        # We collapse Trials and Channels into a single "Signal Source" dimension
        # Shape becomes (Trials*Channels, Time)
        flattened_data = tensor_data.reshape(Trials * Channels, Time)

        print(f"Decomposing joint tensor with {Trials * Channels} signals...")

        # 3. Run MVMD on the flattened data
        # The algorithm will sum spectral power across ALL trials and channels
        # to find the globally optimal center frequencies.
        u_flat, omega = self.mvmd_engine(flattened_data)

        # u_flat shape is (K, Trials*Channels, Time)

        # 4. Tensor Refolding
        # Reshape back to 4D: (K, Trials, Channels, Time)
        u_tensor = u_flat.reshape(self.mvmd_engine.K, Trials, Channels, Time)

        return u_tensor, omega