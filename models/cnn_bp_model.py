"""
cnn_bp_model.py — 1D CNN model for Blood Pressure Estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_BP_Estimator(nn.Module):
    def __init__(self, input_channels=1, seq_len=375):
        """
        Args:
            input_channels (int): 1 for PPG-only, 2 for PPG+ECG
            seq_len (int): input sequence length (default 3 sec @ 125Hz = 375)
        """
        super(CNN_BP_Estimator, self).__init__()

        # --- Shared feature extractor (Encoder) ---
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # collapses over time dimension
        )

        # --- Fully connected shared layer ---
        self.fc_shared = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)

        # --- Two separate regression heads ---
        self.fc_sbp = nn.Linear(64, 1)
        self.fc_dbp = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: [batch, channels, seq_len]
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # flatten
        shared = F.relu(self.fc_shared(features))
        shared = self.dropout(shared)

        sbp = self.fc_sbp(shared)
        dbp = self.fc_dbp(shared)

        # concatenate outputs
        out = torch.cat([sbp, dbp], dim=1)
        return out
