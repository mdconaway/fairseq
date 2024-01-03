from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor
from fairseq.nn.padding import PaddingMask


@dataclass
class SequenceBatch:
    """Represents a sequence batch."""

    seqs: Tensor
    """The sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is the batch
    size, :math:`S` is the sequence length, and :math:`*` is any number of
    sequence-specific dimensions including none."""

    padding_mask: Optional[PaddingMask]
    """The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N` is
    the batch size and :math:`S` is the sequence length."""

    example: Any = None
    """The data example from which this batch was constructed."""

    @property
    def batch_size(self) -> int:
        """The size of the batch."""
        return self.seqs.size(0)

    def compute_num_tokens(self) -> Tensor:
        """Compute the number of tokens in this batch."""
        if self.padding_mask is None:
            return torch.full((), self.seqs.numel(), device=self.seqs.device)

        return self.padding_mask.seq_lens.sum()
