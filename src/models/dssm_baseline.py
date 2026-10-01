"""Fixed-position, target-type-aware DSSM and safe in-batch loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedPositionDSSM(nn.Module):
    """Use fixed-position pooling with the canonical PAD=0, UNK=1 ids."""

    def __init__(
        self,
        *,
        num_items: int,
        embedding_dim: int = 128,
        num_types: int = 4,
        temperature: float = 0.1,
        sparse_item_gradients: bool = True,
    ) -> None:
        super().__init__()
        if num_items < 3 or embedding_dim <= 0 or num_types < 4 or temperature <= 0:
            raise ValueError("Invalid DSSM model dimensions or temperature")
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_types = num_types
        self.temperature = temperature
        self.sparse_item_gradients = sparse_item_gradients
        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
            padding_idx=0,
            sparse=sparse_item_gradients,
        )
        self.type_embedding = nn.Embedding(num_types, embedding_dim, padding_idx=0)
        # UNK is only used in later query histories, never as a training target.
        with torch.no_grad():
            self.item_embedding.weight[1].zero_()

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_embedding(item_ids), p=2, dim=-1)

    def encode_session(
        self,
        history_item_ids: torch.Tensor,
        history_event_types: torch.Tensor,
        target_types: torch.Tensor,
    ) -> torch.Tensor:
        if history_item_ids.shape != history_event_types.shape:
            raise ValueError("History item and event-type tensors must be aligned")
        if history_item_ids.ndim != 2 or target_types.ndim != 1:
            raise ValueError("Expected padded [batch, length] histories and [batch] targets")
        if history_item_ids.size(0) != target_types.size(0):
            raise ValueError("History and target batch sizes differ")

        history = self.item_embedding(history_item_ids) + self.type_embedding(
            history_event_types
        )
        mask = history_item_ids.ne(0)
        # Number only real events; left padding must not change a session vector.
        weights = mask.cumsum(dim=1).to(history.dtype) * mask.to(history.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (history * weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(pooled + self.type_embedding(target_types), p=2, dim=-1)

    def forward(
        self,
        history_item_ids: torch.Tensor,
        history_event_types: torch.Tensor,
        target_item_ids: torch.Tensor,
        target_types: torch.Tensor,
    ) -> torch.Tensor:
        sessions = self.encode_session(history_item_ids, history_event_types, target_types)
        items = self.encode_item(target_item_ids)
        return (sessions @ items.T) / self.temperature


def masked_in_batch_loss(
    logits: torch.Tensor,
    target_item_ids: torch.Tensor,
    sample_weights: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Exclude other columns with the same positive item from each denominator."""

    batch_size = target_item_ids.numel()
    if logits.shape != (batch_size, batch_size):
        raise ValueError("In-batch logits must have shape [batch, batch]")
    if sample_weights.shape != (batch_size,) or (sample_weights <= 0).any():
        raise ValueError("Each sample must have a positive loss weight")
    duplicate = target_item_ids[:, None].eq(target_item_ids[None, :])
    duplicate.fill_diagonal_(False)
    safe_logits = logits.float().masked_fill(duplicate, -torch.inf)
    per_sample = F.cross_entropy(
        safe_logits,
        torch.arange(batch_size, device=logits.device),
        reduction="none",
    )
    weights = sample_weights.float()
    loss = (per_sample * weights).sum() / weights.sum()
    return loss, int(duplicate.sum().item())
