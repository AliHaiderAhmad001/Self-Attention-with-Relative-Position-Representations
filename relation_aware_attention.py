import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

class RelativePosition(nn.Module):
    """
    Relative Position Embeddings Module

    This module generates learnable relative position embeddings to enrich
    the self-attention mechanism with information about the relative distances
    between elements in input sequences.

    Args:
        d_a (int): Number of dimensions in the relative position embeddings.
        k (int): Clipping distance.

    Attributes:
        position_embeddings (nn.Parameter): Learnable parameter for relative position embeddings.

    Example:
        >>> # Create a RelativePosition instance with 16 dimensions and clipping distance of 10
        >>> relative_position = RelativePosition(d_a=16, k=10)
        >>> # Generate relative position embeddings for sequences of lengths 5 and 7
        >>> embeddings = relative_position(length_query=5, length_key=7)
    """

    def __init__(self, d_a: int, k: int):
        """
        Initialize the RelativePosition module.

        Args:
        - d_a (int): Number of dimensions in the relative position embeddings.
        - k (int): Clipping distance.
        """
        super().__init__()
        self.d_a = d_a
        self.k = k
        self.position_embeddings = nn.Parameter(torch.empty((2 * k + 1, d_a)))
        nn.init.xavier_uniform_(self.position_embeddings)

    def forward(self, length_query: int, length_key: int) -> torch.Tensor:
        """
        Compute relative position embeddings.

        Args:
        - length_query (int): Length of the query sequence.
        - length_key (int): Length of the key sequence.

        Returns:
        - embeddings (torch.Tensor): Relative position embeddings (length_query, length_key, embedding_dim).
        """
        # Generate relative position embeddings
        indices_query = torch.arange(length_query, device=self.position_embeddings.device)
        indices_key = torch.arange(length_key, device=self.position_embeddings.device)
        distance_matrix = indices_key.unsqueeze(0) - indices_query.unsqueeze(1)
        distance_matrix_clipped = torch.clamp(distance_matrix, -self.k, self.k)
        final_matrix = distance_matrix_clipped + self.k
        embeddings = self.position_embeddings[final_matrix.to(torch.long)]

        return embeddings
