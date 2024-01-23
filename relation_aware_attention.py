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


class RelationAwareAttentionHead(nn.Module):
    """
    Relation-aware attention head implementation.

    Args:
        hidden_size (int): Hidden size for the model (embedding dimension).
        head_dim (int): Dimensionality of the attention head.
        k_bias_matrix (torch.Tensor): Matrix for relative position attention in query-key interaction.
        v_bias_matrix (torch.Tensor): Matrix for relative position attention in query-value interaction.

    Attributes:
        query_weights (nn.Linear): Linear layer for query projection.
        key_weights (nn.Linear): Linear layer for key projection.
        value_weights (nn.Linear): Linear layer for value projection.
    """

    def __init__(self, hidden_size, head_dim, k_bias_matrix, v_bias_matrix):
        """
        Initializes the RelationAwareAttentionHead.

        Args:
            hidden_size (int): Hidden size for the model (embedding dimension).
            head_dim (int): Dimensionality of the attention head.
            k_bias_matrix (torch.Tensor): Matrix for relative position attention in query-key interaction.
            v_bias_matrix (torch.Tensor): Matrix for relative position attention in query-value interaction.
        """
        super().__init__()
        self.head_dim = head_dim
        self.query_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.key_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.value_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.k_bias_matrix = k_bias_matrix
        self.v_bias_matrix = v_bias_matrix

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated value embeddings after applying attention mechanism.
        """
        query: torch.Tensor = self.query_weights(query) # (b_s, n_t, head_dim)
        key: torch.Tensor = self.key_weights(key) # (b_s, n_t, head_dim)
        value: torch.Tensor = self.value_weights(value) # (b_s, n_t, head_dim)

        # Self-Attention scores
        attn_1: torch.Tensor = torch.matmul(query, key.transpose(1, 2)) # Q*K^T:(b_s, n_t, n_t)

        # Relative Position Attention scores
        attn_2: torch.Tensor = torch.matmul(query.permute(1, 0, 2), self.k_bias_matrix.transpose(1, 2)).transpose(0, 1) # Q*K_shifting^T:(b_s, n_t, n_t)

        # Relation-aware Self-Attention scores
        att_scores: torch.Tensor = (attn_1 + attn_2)/self.head_dim ** 0.5

        if mask is not None:
            mask = mask.to(torch.int)
            att_scores: torch.Tensor = att_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        att_weights: torch.Tensor = F.softmax(att_scores, dim=-1)

        # Weighted sum of values
        values_1: torch.Tensor = torch.matmul(att_weights, value) # (b_s, n_t, head_dim)

        # Relative Position Representation for values
        values_2: torch.Tensor = torch.matmul(att_weights.permute(1, 0, 2), self.v_bias_matrix).transpose(0, 1) # (b_s, n_t, head_dim)

        # Relation-aware values
        n_value  = values_1 + values_2
        return n_value


class RelationAwareMultiHeadAttention(nn.Module):
    """
    Multi-head attention layer implementation.

    Args:
        hidden_size (int): Hidden size for the model (embedding dimension).
        num_heads (int): Number of attention heads.
        k (int): Clipping distance for relative position embeddings.
        seq_len (int): Length of the input sequences.

    Attributes:
        hidden_size (int): Hidden size for the model (embedding dimension).
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        relative_position_k (RelativePosition): Instance of RelativePosition for query-key relative positions.
        relative_position_v (RelativePosition): Instance of RelativePosition for query-value relative positions.
        k_bias_matrix (torch.Tensor): Matrix for relative position attention in query-key interaction.
        v_bias_matrix (torch.Tensor): Matrix for relative position attention in query-value interaction.
        attention_heads (nn.ModuleList): List of RelationAwareAttentionHead layers.
        fc (nn.Linear): Fully connected layer for final projection.
    """

    def __init__(self, hidden_size, num_heads, k, seq_len):
        """
        Initializes the RelationAwareMultiHeadAttention layer.

        Args:
            hidden_size (int): Hidden size for the model (embedding dimension).
            num_heads (int): Number of attention heads.
            k (int): Clipping distance for relative position embeddings.
            seq_len (int): Length of the input sequences.
        """
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_size // num_heads
        self.relative_position_k: torch.Tensor = RelativePosition(self.head_dim, k)
        self.relative_position_v: torch.Tensor = RelativePosition(self.head_dim, k)
        self.k_bias_matrix: torch.Tensor = self.relative_position_k(seq_len, seq_len)
        self.v_bias_matrix: torch.Tensor = self.relative_position_v(seq_len, seq_len)
        self.attention_heads: nn.ModuleList = nn.ModuleList([RelationAwareAttentionHead(self.hidden_size, self.head_dim,
                                                                           self.k_bias_matrix, self.v_bias_matrix) for _ in range(self.num_heads)])
        self.fc: nn.Linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies multi-head attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated hidden state after applying multi-head attention mechanism.
        """
        attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask=mask) for attention_head in self.attention_heads]
        hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
        hidden_state: torch.Tensor = self.fc(hidden_state)
        return hidden_state
