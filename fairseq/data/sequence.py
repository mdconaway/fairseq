
from typing import TypedDict
from torch import Tensor


class SequenceData(TypedDict):
    seqs: Tensor
    seq_lens: Tensor
    is_ragged: bool
