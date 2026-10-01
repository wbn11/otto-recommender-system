"""Stream next-item DSSM examples from session-sequence Parquet row groups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class NextItemSample:
    session: int
    target_position: int
    history_item_ids: tuple[int, ...]
    history_event_types: tuple[int, ...]
    target_item_id: int
    target_type: int


class ParquetNextItemDataset(IterableDataset[NextItemSample]):
    """Assign whole row groups to workers and yield next-item pairs online."""

    def __init__(
        self,
        sequences_dir: str | Path,
        *,
        max_history: int = 50,
        sessions_per_batch: int = 1024,
    ) -> None:
        super().__init__()
        self.sequences_dir = Path(sequences_dir).resolve()
        self.files = tuple(sorted(self.sequences_dir.glob("part-*.parquet")))
        if not self.files:
            raise FileNotFoundError(f"No DSSM sequences found in {self.sequences_dir}")
        if max_history <= 0 or sessions_per_batch <= 0:
            raise ValueError("max_history and sessions_per_batch must be positive")
        self.max_history = max_history
        self.sessions_per_batch = sessions_per_batch
        self.row_groups = tuple(
            (path, group_index)
            for path in self.files
            for group_index in range(pq.ParquetFile(path).num_row_groups)
        )

    def __iter__(self) -> Iterator[NextItemSample]:
        worker = get_worker_info()
        for task_index, (path, group_index) in enumerate(self.row_groups):
            if worker is not None and task_index % worker.num_workers != worker.id:
                continue
            parquet_file = pq.ParquetFile(path)
            for record_batch in parquet_file.iter_batches(
                batch_size=self.sessions_per_batch,
                row_groups=[group_index],
                columns=["session", "item_ids", "event_types"],
                use_threads=False,
            ):
                rows = record_batch.to_pydict()
                for session, item_ids, event_types in zip(
                    rows["session"], rows["item_ids"], rows["event_types"], strict=True
                ):
                    if len(item_ids) != len(event_types) or len(item_ids) < 2:
                        raise ValueError(f"Invalid DSSM sequence for session {session}")
                    for target_position in range(1, len(item_ids)):
                        start = max(0, target_position - self.max_history)
                        yield NextItemSample(
                            session=int(session),
                            target_position=target_position,
                            history_item_ids=tuple(item_ids[start:target_position]),
                            history_event_types=tuple(event_types[start:target_position]),
                            target_item_id=int(item_ids[target_position]),
                            target_type=int(event_types[target_position]),
                        )


def collate_next_item(
    samples: Sequence[NextItemSample],
    *,
    type_weights: Mapping[int, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Left-pad histories; an empty history occupies one PAD position."""

    if not samples:
        raise ValueError("Cannot collate an empty DSSM batch")
    weights = dict(type_weights or {1: 1.0, 2: 3.0, 3: 6.0})
    max_len = max(1, max(len(sample.history_item_ids) for sample in samples))
    history_items: list[list[int]] = []
    history_types: list[list[int]] = []
    for sample in samples:
        if len(sample.history_item_ids) != len(sample.history_event_types):
            raise ValueError("History item and event-type sequences are misaligned")
        if sample.target_item_id < 2 or sample.target_type not in weights:
            raise ValueError("Training targets must be real items with a known event type")
        padding = max_len - len(sample.history_item_ids)
        history_items.append([0] * padding + list(sample.history_item_ids))
        history_types.append([0] * padding + list(sample.history_event_types))
    return {
        "sessions": torch.tensor([sample.session for sample in samples], dtype=torch.long),
        "target_positions": torch.tensor(
            [sample.target_position for sample in samples], dtype=torch.long
        ),
        "history_item_ids": torch.tensor(history_items, dtype=torch.long),
        "history_event_types": torch.tensor(history_types, dtype=torch.long),
        "history_lengths": torch.tensor(
            [len(sample.history_item_ids) for sample in samples], dtype=torch.long
        ),
        "target_item_ids": torch.tensor(
            [sample.target_item_id for sample in samples], dtype=torch.long
        ),
        "target_types": torch.tensor(
            [sample.target_type for sample in samples], dtype=torch.long
        ),
        "sample_weights": torch.tensor(
            [weights[sample.target_type] for sample in samples], dtype=torch.float32
        ),
    }
