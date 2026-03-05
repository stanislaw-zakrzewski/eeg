import torch
import torch.nn as nn


class DualStreamRiemannNet(nn.Module):
    def __init__(self, n_modes, n_channels, n_classes=2, hidden_dim=64, dropout_rate=0.5):
        super().__init__()

        # C * (C+1) / 2
        self.riemann_dim = int(n_channels * (n_channels + 1) / 2)

        # 1. Frequency Attention (Add Dropout)
        self.freq_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # <--- Added
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 2. Path A: Snapshot Stream (Add Dropout)
        self.snapshot_stream = nn.Sequential(
            nn.Linear(self.riemann_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)  # <--- Added
        )

        # 3. Path B: Sequence Stream (Add Spatial Dropout for CNN)
        self.SEQ_OUT_SIZE = 4
        self.sequence_stream = nn.Sequential(
            nn.Conv1d(self.riemann_dim, hidden_dim, kernel_size=3, padding=1),
            # nn.BatchNorm1d(hidden_dim),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # <--- Added
            nn.AdaptiveMaxPool1d(self.SEQ_OUT_SIZE)
        )

        # 4. Fusion Classifier (Add Dropout)
        flattened_dim = hidden_dim * (1 + self.SEQ_OUT_SIZE)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # <--- Added
            nn.Linear(32, n_classes)
        )

    def forward(self, x_tangent, center_freqs):
        # ... (Same forward logic as before) ...
        # Copy-paste the 'forward' method from the previous DualStream version
        # 1. Auto-fix dimensions
        if x_tangent.ndim == 3: x_tangent = x_tangent.unsqueeze(1)
        batch_size, seq_len, modes, feat_dim = x_tangent.shape

        # 2. Attention
        freqs_flat = center_freqs.view(-1, 1)
        mode_weights = self.freq_embed(freqs_flat).view(batch_size, 1, modes, 1)
        x_weighted = x_tangent * mode_weights
        x_fused = torch.sum(x_weighted, dim=2)
        x_input = x_fused.permute(0, 2, 1)

        # 3. Dual Stream
        x_global_raw = torch.mean(x_input, dim=2)
        x_global = self.snapshot_stream(x_global_raw).unsqueeze(2)
        x_seq = self.sequence_stream(x_input)
        x_combined = torch.cat([x_global, x_seq], dim=2)

        return self.classifier(x_combined), mode_weights