import torch
import torch.nn as nn
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

class RiemannianModeNet(nn.Module):
    def __init__(self, n_modes, n_channels, n_classes=2, hidden_dim=64):
        super().__init__()

        # 1. Feature Dimension
        # C * (C+1) / 2
        self.riemann_dim = int(n_channels * (n_channels + 1) / 2)

        # 2. FREQUENCY ATTENTION (Shared)
        self.freq_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # ==========================================
        # PATH A: The "Snapshot" Stream (Global State)
        # ==========================================
        # Used for Seq_Len=1 or capturing overall intensity
        self.snapshot_stream = nn.Sequential(
            nn.Linear(self.riemann_dim, hidden_dim),
            nn.ReLU()
        )

        # ==========================================
        # PATH B: The "Sequence" Stream (Temporal Pattern)
        # ==========================================
        # Used for Seq_Len > 1 to find order of events
        self.SEQ_OUT_SIZE = 4  # Force output to 4 time steps

        self.sequence_stream = nn.Sequential(
            nn.Conv1d(self.riemann_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # Adaptive Pool handles ANY input length (1 or 100)
            nn.AdaptiveMaxPool1d(self.SEQ_OUT_SIZE)
        )

        # ==========================================
        # 4. FUSION CLASSIFIER
        # ==========================================
        # Input = Snapshot (1 step) + Sequence (4 steps) = 5 steps total
        flattened_dim = hidden_dim * (1 + self.SEQ_OUT_SIZE)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x_tangent, center_freqs):
        """
        Args:
            x_tangent: (Batch, Seq, Modes, Feat) OR (Batch, Modes, Feat)
            center_freqs: (Batch, Modes)
        """
        # --- 1. AUTO-FIX DIMENSIONS ---
        # If input is 3D (Batch, Modes, Features), promote to 4D (Batch, 1, Modes, Features)
        if x_tangent.ndim == 3:
            x_tangent = x_tangent.unsqueeze(1)

        batch_size, seq_len, modes, feat_dim = x_tangent.shape

        # --- 2. ATTENTION & FUSION ---
        freqs_flat = center_freqs.view(-1, 1)
        # (Batch, 1, Modes, 1) - Broadcasts across Sequence length automatically
        mode_weights = self.freq_embed(freqs_flat).view(batch_size, 1, modes, 1)

        x_weighted = x_tangent * mode_weights

        # Sum across Modes to get (Batch, Seq_Len, Features)
        x_fused = torch.sum(x_weighted, dim=2)

        # Permute for CNN: (Batch, Features, Seq_Len)
        x_input = x_fused.permute(0, 2, 1)

        # --- 3. DUAL STREAM PROCESSING ---

        # Path A: Snapshot (Global Average)
        # Takes mean across time -> (Batch, Features)
        x_global_raw = torch.mean(x_input, dim=2)
        x_global = self.snapshot_stream(x_global_raw)  # (Batch, Hidden)
        x_global = x_global.unsqueeze(2)  # (Batch, Hidden, 1)

        # Path B: Sequence (Temporal Pattern)
        # Always outputs (Batch, Hidden, 4)
        x_seq = self.sequence_stream(x_input)

        # --- 4. CONCATENATE & CLASSIFY ---
        # Combine (Batch, Hidden, 1) and (Batch, Hidden, 4) -> (Batch, Hidden, 5)
        x_combined = torch.cat([x_global, x_seq], dim=2)

        logits = self.classifier(x_combined)

        return logits, mode_weights


def preprocess_riemannian(eeg_data):
    """
    Converts Raw VMD Output to Tangent Space Features.

    Args:
        eeg_data: Numpy array (Batch, Seq, Modes, Channels, TimePoints)
    Returns:
        torch.Tensor: (Batch, Seq, Modes, RiemannFeatures)
    """
    B, S, K, C, T = eeg_data.shape

    # Collapse Batch/Seq/Modes to treat every window independently for now
    # Shape: (N_samples, C, T)
    reshaped_data = eeg_data.reshape(B * S * K, C, T)

    # 1. Compute Covariance Matrices (OAS estimator is robust)
    cov = Covariances(estimator='oas').fit_transform(reshaped_data)
    # Shape: (N_samples, C, C)

    # 2. Project to Tangent Space
    ts = TangentSpace().fit_transform(cov)
    # Shape: (N_samples, F) where F = C*(C+1)/2

    # Reshape back to (Batch, Seq, Modes, F)
    final_features = ts.reshape(B, S, K, -1)

    return torch.tensor(final_features, dtype=torch.float32)