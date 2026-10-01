"""Target-aware attention DSSM with a sparse item tower.

Only the session pooling mechanism differs from the fixed-position baseline:
the item tower, normalized inner-product objective and in-batch negatives stay
unchanged.  This makes the experiment an interpretable pooling ablation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetAwareAttentionDSSM(nn.Module):
    """Build a different history summary for each requested target type."""

    def __init__(
        self,
        *,
        num_items: int,
        embedding_dim: int = 128,
        num_types: int = 4,
        max_sequence_length: int = 50,
        temperature: float = 0.1,
        sparse_item_gradients: bool = True,
    ) -> None:
        super().__init__()
        if (
            num_items < 3
            or embedding_dim <= 0
            or num_types < 4
            or max_sequence_length <= 0
            or temperature <= 0
        ):
            raise ValueError("Invalid attention DSSM dimensions or temperature")
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_types = num_types
        self.max_sequence_length = max_sequence_length
        self.temperature = temperature
        self.sparse_item_gradients = sparse_item_gradients

        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
            padding_idx=0,
            sparse=sparse_item_gradients,
        )
        self.event_type_embedding = nn.Embedding(
            num_types, embedding_dim, padding_idx=0
        )
        self.target_type_embedding = nn.Embedding(
            num_types, embedding_dim, padding_idx=0
        )
        # ID 1 is the most recent real event, ID 2 the second most recent.
        self.recency_embedding = nn.Embedding(
            max_sequence_length + 1,
            embedding_dim,
            padding_idx=0,
        )
        self.query_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # UNK can occur in later query windows, but is never a training target.
        with torch.no_grad():
            self.item_embedding.weight[1].zero_()

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_embedding(item_ids), p=2, dim=-1)

    def _validate_session_inputs(
        self,
        history_item_ids: torch.Tensor,
        history_event_types: torch.Tensor,
        target_types: torch.Tensor,
    ) -> None:
        if history_item_ids.shape != history_event_types.shape:
            raise ValueError("History item and event-type tensors must be aligned")
        if history_item_ids.ndim != 2 or target_types.ndim != 1:
            raise ValueError("Expected padded [batch, length] histories and [batch] targets")
        if history_item_ids.size(0) != target_types.size(0):
            raise ValueError("History and target batch sizes differ")
        if history_item_ids.size(1) > self.max_sequence_length:
            raise ValueError("History is longer than the configured attention window")

    @staticmethod
    def _recency_ids(mask: torch.Tensor) -> torch.Tensor:
        """Return 1 for newest, 2 for second newest, and 0 for padding."""

        lengths = mask.sum(dim=1, keepdim=True)
        chronological_positions = mask.cumsum(dim=1)
        return (lengths - chronological_positions + 1).clamp_min(0) * mask

    def _attend(
        self,
        history_item_ids: torch.Tensor,
        history_event_types: torch.Tensor,
        target_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_session_inputs(
            history_item_ids, history_event_types, target_types
        )
        mask = history_item_ids.ne(0)
        recency_ids = self._recency_ids(mask).long()
        history = (
            self.item_embedding(history_item_ids)
            + self.event_type_embedding(history_event_types)
            + self.recency_embedding(recency_ids)
        )
        target = self.target_type_embedding(target_types)
        queries = self.query_projection(target)
        keys = self.key_projection(history)
        values = self.value_projection(history)
        scores = torch.einsum("bld,bd->bl", keys, queries) / math.sqrt(
            self.embedding_dim
        )

        # Multiplying the softmax by mask and renormalizing also makes an
        # all-PAD row safe: it receives zero history instead of NaN.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=1).to(history.dtype)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pooled = torch.sum(values * weights.unsqueeze(-1), dim=1)
        return pooled + target, weights

    def attention_weights(
        self,
        history_item_ids: torch.Tensor,
        history_event_types: torch.Tensor,
        target_types: torch.Tensor,
    ) -> torch.Tensor:
        """Expose normalized weights for tests and model interpretation."""

        return self._attend(
            history_item_ids, history_event_types, target_types
        )[1]

    def encode_session(
        self,
        history_item_ids: torch.Tensor,
        history_event_types: torch.Tensor,
        target_types: torch.Tensor,
    ) -> torch.Tensor:
        session, _ = self._attend(
            history_item_ids, history_event_types, target_types
        )
        return F.normalize(session, p=2, dim=-1)

    def forward(
        self,
        history_item_ids: torch.Tensor,
        history_event_types: torch.Tensor,
        target_item_ids: torch.Tensor,
        target_types: torch.Tensor,
    ) -> torch.Tensor:
        sessions = self.encode_session(
            history_item_ids, history_event_types, target_types
        )
        items = self.encode_item(target_item_ids)
        return (sessions @ items.T) / self.temperature
