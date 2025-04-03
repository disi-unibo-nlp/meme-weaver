
import torch
from torch import nn


class Rs_GCN(nn.Module):
    def __init__(self, config):
        """
        Initialize the Rs_GCN module.

        Args:
            hidden_size (int): Dimensionality of the input features.
        """
        super(Rs_GCN, self).__init__()
        # Fully connected layer to compute φ(·) for transforming input embeddings.
        self.phi = nn.Linear(config.hidden_size, config.hidden_size)
        # Fully connected layer to compute γ(·) for transforming input embeddings.
        self.gamma = nn.Linear(config.hidden_size, config.hidden_size)
        # Linear transformation applied to the node features after aggregation.
        self.W_g = nn.Linear(config.hidden_size, config.hidden_size)
        # Residual weights linear layer applied on the aggregated features.
        self.W_r = nn.Linear(config.hidden_size, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, features):
        """
        Forward pass for the Rs_GCN layer.

        Args:
            embeddings (torch.Tensor): Input tensor of shape (batch_size, hidden_size)
                                       Each row represents a node's feature.
        Returns:
            torch.Tensor: Updated node features of shape (batch_size, hidden_size)
        """
        # Transform input features using φ and γ functions.
        phi_out = self.phi(features)  # Shape: (batch_size, hidden_size)
        gamma_out = self.gamma(features)  # Shape: (batch_size, hidden_size)

        # Compute the affinity matrix R as the dot product between transformed features.
        R = torch.matmul(phi_out, gamma_out.t())  # Shape: (batch_size, batch_size)

        # Normalize the affinity matrix by dividing by the number of nodes (i.e., last dimension size).
        R_norm = R / R.size(-1)

        # Apply a linear transformation on the original features.
        features_v = self.W_g(features)  # Shape: (batch_size, hidden_size)

        # Aggregate neighboring features using the normalized affinity matrix.
        RV = torch.matmul(R_norm, features_v)  # Shape: (batch_size, hidden_size)

        # Apply a second linear transformation on the aggregated features.
        transformed = self.W_r(RV)  # Shape: (batch_size, hidden_size)

        # Add a residual connection from the original features.
        out = transformed + features

        # Apply layer normalization.
        # out = self.layer_norm(out)

        return out


