
import torch
from torch import nn

import torch.nn.functional as F


class Rs_GCN_upd(nn.Module):
    def __init__(self, config):
        """
        Initialize the Rs_GCN_upd module.

        Args:
            hidden_size (int): Dimensionality of the input features.
        """
        super(Rs_GCN_upd, self).__init__()
        # Fully connected layer to compute φ(·) for transforming input embeddings.
        self.phi = nn.Linear(config.hidden_size, config.hidden_size)
        # Fully connected layer to compute γ(·) for transforming input embeddings.
        self.psi = nn.Linear(config.hidden_size, config.hidden_size)
        # Linear transformation applied to the node features after aggregation.
        self.W_g = nn.Linear(config.hidden_size, config.hidden_size)
        # Residual weights linear layer applied on the aggregated features.
        self.W_r = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, features):
        """
        Forward pass for the Rs_GCN_upd layer.

        Args:
            embeddings (torch.Tensor): Input tensor of shape (batch_size, hidden_size)
                                       Each row represents a node's feature.
        Returns:
            torch.Tensor: Updated node features of shape (batch_size, hidden_size)
        """
        # Transform input features using φ and γ functions.
        phi_out = self.phi(features)  # Shape: (batch_size, hidden_size)
        psi_out = self.psi(features)  # Shape: (batch_size, hidden_size)

        # Compute the affinity matrix R as the dot product between transformed features.
        R = torch.matmul(phi_out, psi_out.t())  # Shape: (batch_size, batch_size)

        # Normalize the affinity matrix by dividing by the number of nodes (i.e., last dimension size).
        R_norm = R / R.size(-1)
        
        # Apply softmax to the affinity matrix to ensure it sums to 1 across the last dimension.
        R_norm = F.softmax(R_norm, dim=-1)

        # Apply a linear transformation on the original features.
        features_v = self.W_g(features)  # Shape: (batch_size, hidden_size)

        # Aggregate neighboring features using the normalized affinity matrix.
        RV = torch.matmul(R_norm, features_v)  # Shape: (batch_size, hidden_size)

        # Apply a second linear transformation on the aggregated features.
        transformed = self.W_r(RV)  # Shape: (batch_size, hidden_size)

        out = transformed + features
 
        return out, R_norm.cpu()


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
        self.psi_param = nn.Linear(config.hidden_size, config.hidden_size)
        # Linear transformation applied to the node features after aggregation.
        self.W_g = nn.Linear(config.hidden_size, config.hidden_size)
        # Residual weights linear layer applied on the aggregated features.
        self.W_r = nn.Linear(config.hidden_size, config.hidden_size)

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
        psi_out = self.psi_param(features)  # Shape: (batch_size, hidden_size)

        # Compute the affinity matrix R as the dot product between transformed features.
        R = torch.matmul(phi_out, psi_out.t())  # Shape: (batch_size, batch_size)

        # Normalize the affinity matrix by dividing by the number of nodes (i.e., last dimension size).
        R_norm = R / R.size(-1)

        # Apply a linear transformation on the original features.
        features_v = self.W_g(features)  # Shape: (batch_size, hidden_size)

        # Aggregate neighboring features using the normalized affinity matrix.
        RV = torch.matmul(R_norm, features_v)  # Shape: (batch_size, hidden_size)

        # Apply a second linear transformation on the aggregated features.
        transformed = self.W_r(RV)  # Shape: (batch_size, hidden_size)

        out = transformed + features
 
        return out, R_norm.cpu()


class Sim_GCN(nn.Module):
    def __init__(self, config):
        """
        Initialize the Sim_GCN module.

        Args:
            hidden_size (int): Dimensionality of the input features.
        """
        super(Sim_GCN, self).__init__()

        # Linear transformation applied to the node features after aggregation.
        self.W_g = nn.Linear(config.hidden_size, config.hidden_size)
        # Residual weights linear layer applied on the aggregated features.
        self.W_r = nn.Linear(config.hidden_size, config.hidden_size)

        # self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, features):
        """
        Forward pass for the Sim_GCN layer.

        Args:
            features (torch.Tensor): Input tensor of shape (batch_size, hidden_size)
                                       Each row represents a node's feature.
        Returns:
            torch.Tensor: Updated node features of shape (batch_size, hidden_size)
        """


        # Compute the affinity matrix using cosine similarity.
        # The (i,j)-th entry is the cosine similarity between features[i] and features[j].
        R = torch.matmul(features, features.t())  

        # Normalize the affinity matrix by dividing by the number of nodes (i.e., last dimension size).
        R_norm = R / R.size(-1)

        # Apply a linear transformation on the original features.
        features_v = self.W_g(features)  # Shape: (batch_size, hidden_size)

        # Aggregate neighboring features using the normalized affinity matrix.
        RV = torch.matmul(R_norm, features_v)  # Shape: (batch_size, hidden_size)

        # Apply a second linear transformation on the aggregated features.
        transformed = self.W_r(RV)  # Shape: (batch_size, hidden_size)

        out = transformed + features
 
        return out, R_norm.cpu()
    

gcn_map = {
            "learn": Rs_GCN,
            "sim": Sim_GCN,
            "learn_upd": Rs_GCN_upd
        }
