import torch
import torch.nn as nn


class VMDInceptionNet(nn.Module):
    def __init__(self, n_modes, n_channels, n_time, n_classes=2, dropout_rate=0.5):
        super().__init__()

        # 1. Frequency Attention (Same as before - it works great)
        self.freq_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 2. TEMPORAL CONVOLUTION (Per-Channel, Per-Mode)
        # We use 'groups=n_channels*n_modes' to learn separate filters for every single signal
        # Input: (Batch, Modes*Channels, Time)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=n_modes * n_channels,
                      out_channels=n_modes * n_channels * 2,  # Expansion
                      kernel_size=64, padding=32, groups=n_modes * n_channels, bias=False),
            nn.BatchNorm1d(n_modes * n_channels * 2),
            nn.ELU(),
            nn.Dropout(dropout_rate)
        )

        # 3. SPATIAL CONVOLUTION (Mixes Channels within each Mode)
        # This replaces the Riemannian Covariance Matrix
        # It learns "Spatial Patterns" for each frequency band
        self.spatial_conv = nn.Sequential(
            # We group by 'Modes' so we don't mix Mode 1 with Mode 5 yet
            nn.Conv1d(in_channels=n_modes * n_channels * 2,
                      out_channels=n_modes * 16,  # Compress to 16 spatial features per mode
                      kernel_size=1, groups=n_modes, bias=False),
            nn.BatchNorm1d(n_modes * 16),
            nn.ELU(),
            nn.AvgPool1d(4),  # Downsample time
            nn.Dropout(dropout_rate)
        )

        # 4. CLASSIFIER
        # Calculate flat size based on pooling
        # (Time // 4) * (Modes * 16)
        flat_dim = (n_time // 4) * (n_modes * 16)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, n_classes)
        )

        self.n_modes = n_modes
        self.n_channels = n_channels

    def forward(self, x_raw, center_freqs):
        """
        Args:
            x_raw: (Batch, Seq, Modes, Chan, Time)
            center_freqs: (Batch, Modes)
        """
        # 1. Handle Dimensions
        if x_raw.ndim == 4: x_raw = x_raw.unsqueeze(1)  # Add Seq=1 if missing
        B, S, K, C, T = x_raw.shape

        # Collapse Batch and Sequence (treat every window independently)
        # We reshape to (B*S, K*C, T) for Conv1d
        x = x_raw.view(B * S, K * C, T)

        # 2. Apply Temporal Filter
        x = self.temporal_conv(x)

        # 3. Apply Spatial Filter
        x = self.spatial_conv(x)
        # Output: (B*S, K*16, T_new)

        # 4. FREQUENCY ATTENTION (Re-introduced)
        # We need to reshape x to apply weights per mode
        # Current: (B*S, K*16, T_new) -> View as (B*S, K, 16, T_new)
        x_reshaped = x.view(B * S, K, 16, -1)

        # Get weights
        # (B, K) -> (B*S, K, 1, 1)
        # Note: We repeat weights for the sequence length
        freqs_flat = center_freqs.repeat_interleave(S, dim=0).view(-1, 1)
        weights = self.freq_embed(freqs_flat).view(B * S, K, 1, 1)

        # Apply Attention
        x_weighted = x_reshaped * weights

        # 5. Classify
        logits = self.classifier(x_weighted)

        # Reshape logits back to (B, S, Classes) if needed, or just keep (B*S, Classes)
        if S > 1:
            logits = logits.view(B, S, -1)
            # Optional: Average over sequence for final prediction
            logits = torch.mean(logits, dim=1)

        return logits, weights