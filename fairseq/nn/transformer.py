from __future__ import annotations

import logging
import math
from enum import Enum
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Protocol, Tuple, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Dropout, ReLU
from torch.nn.parameter import Parameter
from torch.nn.functional import dropout, softmax, pad, scaled_dot_product_attention
from torch.utils.hooks import RemovableHandle

from fairseq.nn.module_list import ModuleList
from fairseq.nn.padding import PaddingMask
from fairseq.nn.incremental_state import IncrementalState, IncrementalStateBag
from fairseq.nn.normalization import LayerNorm, StandardLayerNorm
from fairseq.nn.ops import repeat_interleave
from fairseq.nn.position_encoder import PositionEncoder
from fairseq.nn.projection import Linear, Projection
from fairseq.typing import DataType, Device, finaloverride


logger = logging.getLogger(__name__)


class AttentionMask(ABC):
    """Represents an attention mask."""

    materialized: Optional[Tensor]

    def __init__(self) -> None:
        self.materialized = None

    def materialize(self) -> Tensor:
        """Materialize the attention mask tensor."""
        if self.materialized is None:
            self.materialized = self._do_materialize()

        return self.materialized

    @abstractmethod
    def _do_materialize(self) -> Tensor:
        ...


class SDPA(Module, ABC):
    """Computes scaled dot-product attention."""

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        keys: Tensor,
        key_padding_mask: Optional[PaddingMask],
        values: Tensor,
        *,
        attn_mask: Optional[AttentionMask] = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param seqs:
            The sequences to query. *Shape:* :math:`(N,H,S,K)`, where :math:`N`
            is the batch size, :math:`H` is the number of heads, :math:`S` is
            the sequence length, and :math:`K` is the key size.
        :param keys:
            The keys. *Shape:* :math:`(N,H,S_{kv},K)`, where :math:`N` is the
            batch size, :math:`H` is the number of heads, :math:`S_{kv}` is the
            key/value sequence length, and :math:`K` is the key size.
        :param key_padding_mask:
            The padding mask indicating which key positions to ignore for the
            purpose of attention. *Shape:* :math:`(N,S_{kv})`, where :math:`N`
            is the batch size and :math:`S_{kv}` is the key/value sequence
            length.
        :param values:
            The values. *Shape:* :math:`(N,H,S_{kv},V)`, where :math:`N` is the
            batch size, :math:`H` is the number of heads, :math:`S_{kv}` is the
            key/value sequence length, and :math:`V` is the value size.
        :param attn_mask:
            The mask that will be added to attention weights before computing
            the attention. *Shape:* :math:`([H],S,S_{kv})`, where :math:`H` is
            the number of heads, :math:`S` is the sequence length, and
            :math:`S_{kv}` is the key/value sequence length.
        :param needs_weights:
            If ``True``, returns the attention weights.

        :returns:
            - The attention values. *Shape:* :math:`(N,H,S,V)`, where :math:`N`
              is the batch size, :math:`H` is the number of heads, :math:`S` is
              the sequence length, and :math:`V` is the value size.
            - The attention weights. *Shape:* :math:`(N,H,S,S_{kv})`, where
              :math:`N` is the batch size, :math:`H` is the number of heads,
              :math:`S` is the sequence length, and :math:`S_{kv}` is the
              key/value sequence length.
        """


def _naive_scaled_dot_product_attention(
    seqs: Tensor,
    keys: Tensor,
    key_padding_mask: Optional[PaddingMask],
    values: Tensor,
    attn_mask: Optional[AttentionMask],
    dropout_p: float,
    needs_weights: bool,
    training: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    # (N, H, S, K) @ (N, H, K, S_kv) = (N, H, S, S_kv)
    attn_weights = torch.matmul(seqs, keys.transpose(-1, -2))

    attn_weights = attn_weights * (seqs.size(-1) ** -0.5)

    if attn_mask is not None:
        # (S, S_kv)
        m = attn_mask.materialize()

        # (N, H, S, S_kv) + (S, S_kv) -> (N, H, S, S_kv)
        attn_weights = attn_weights + m

    if key_padding_mask is not None:
        # (N, S_kv)
        m = key_padding_mask.materialize()

        m = m[:, None, None, :]

        # (N, H, S, S_kv) + (N, 1, 1, S_kv) -> (N. H, S, S_kv)
        attn_weights = torch.where(m, attn_weights, -torch.inf)

    # For numerical stability run in single precision.
    attn_weights = softmax(attn_weights, dim=-1, dtype=torch.float32)

    attn_weights = attn_weights.type_as(seqs)

    if training and dropout_p > 0.0:
        attn_weights = dropout(attn_weights, dropout_p)

    # (N, H, S, S_kv) @ (N, H, S_kv, V) = (N, H, S, V)
    attn = torch.matmul(attn_weights, values)

    return attn, attn_weights if needs_weights else None


def _create_causal_attention_mask(
    seq_len: int,
    key_len: int,
    attn_window_len: Optional[int],
    device: Optional[Device],
    dtype: Optional[DataType],
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    # As of PyTorch 2.0, `triu` does not support bf16.
    dt = torch.float32 if dtype == torch.bfloat16 else dtype

    mask = torch.ones((seq_len, key_len), device=device, dtype=dt)

    mask.tril_(diagonal=0)

    if attn_window_len is not None:
        mask.triu_(diagonal=1 - attn_window_len)

    mask.log_()

    return mask.to(dtype)


@final
class CausalAttentionMask(AttentionMask):
    """Represents a causal attention mask.

    *Shape:* :math:`(S,S_{kv})`, where :math:`S` is the sequence length and
    :math:`S_{kv}` is the key/value sequence length.

    Usage:

    >>> import torch
    >>>
    >>> from fairseq2.nn.transformer import CausalAttentionMask
    >>>
    >>> mask = CausalAttentionMask(seq_len=4, key_len=6)
    >>> mask.materialize()
    tensor([[0., -inf, -inf, -inf, -inf, -inf],
            [0.,   0., -inf, -inf, -inf, -inf],
            [0.,   0.,   0., -inf, -inf, -inf],
            [0.,   0.,   0.,   0., -inf, -inf]])
    >>>
    >>> mask = CausalAttentionMask(seq_len=4, key_len=4, attn_window_len=2)
    >>> mask.materialize()
    tensor([[0.,   -inf, -inf, -inf],
            [0.,     0., -inf, -inf],
            [-inf,   0.,   0., -inf],
            [-inf, -inf,   0.,   0.]])
    """

    def __init__(
        self,
        seq_len: int,
        key_len: int,
        *,
        attn_window_len: Optional[int] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param seq_len:
            The sequence length.
        :param key_len:
            The key/value sequence length.
        :param attn_window_len:
            The attention window length as described in Section 3.1 of
            :cite:t:`https://doi.org/10.48550/arxiv.2004.05150`. If ``None``,
            constructs a full causal attention mask.
        """
        super().__init__()

        self.seq_len = seq_len
        self.key_len = key_len
        self.attn_window_len = attn_window_len

        self.device, self.dtype = device, dtype

    @finaloverride
    def _do_materialize(self) -> Tensor:
        return _create_causal_attention_mask(
            self.seq_len, self.key_len, self.attn_window_len, self.device, self.dtype
        )


@final
class TorchSDPA(SDPA):
    """Computes scaled dot-product attention using PyTorch SDPA v2."""

    attn_dropout_p: float

    def __init__(self, *, attn_dropout_p: float = 0.0) -> None:
        """
        :param attn_dropout_p:
            The dropout probability on attention weights.
        """
        super().__init__()

        self._has_warned = False

        self.attn_dropout_p = attn_dropout_p

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        keys: Tensor,
        key_padding_mask: Optional[PaddingMask],
        values: Tensor,
        *,
        attn_mask: Optional[AttentionMask] = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if needs_weights:
            if not self._has_warned:
                logger.warning(
                    "`TorchSDPA` has to fall back to the naive SDPA implementation because of `needs_weights` set to `True`."
                )

                self._has_warned = True

            return _naive_scaled_dot_product_attention(
                seqs,
                keys,
                key_padding_mask,
                values,
                attn_mask,
                self.attn_dropout_p,
                needs_weights,
                self.training,
            )

        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.attn_dropout_p

        is_causal = False

        if key_padding_mask is not None:
            mask = key_padding_mask.materialize()

            # (N, S_kv) -> (N, 1, 1, S_kv)
            mask = mask[:, None, None, :]

            # (N, 1, 1, S_kv) -> (N, H, S, S_kv)
            mask = mask.expand(-1, seqs.size(1), seqs.size(2), -1)

            if attn_mask is not None:
                # ([H], S, S_kv)
                m = attn_mask.materialize()

                # (N, H, S, S_kv)
                mask = torch.where(mask, m, -torch.inf)
        elif isinstance(attn_mask, CausalAttentionMask):
            # PyTorch SDPA supports only full causal attention.
            if attn_mask.attn_window_len is None:
                mask = None

                is_causal = True
            else:
                # ([H], S, S_kv)
                mask = attn_mask.materialize()
        elif attn_mask is not None:
            # ([H], S, S_kv)
            mask = attn_mask.materialize()
        else:
            mask = None

        attn = scaled_dot_product_attention(  # type: ignore[attr-defined]
            seqs,
            keys,
            values,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        return attn, None

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"attn_dropout_p={self.attn_dropout_p}"


class SDPAFactory(Protocol):
    """Constructs instances of :class:`SDPA`."""

    def __call__(self, *, attn_dropout_p: float = 0.0) -> SDPA:
        """
        :param attn_dropout_p:
            The dropout probability on attention weights.
        """


def _get_fallback_sdpa_factory() -> SDPAFactory:
    return TorchSDPA


_sdpa_factory: SDPAFactory = _get_fallback_sdpa_factory()


def create_default_sdpa(*, attn_dropout_p: float = 0.0) -> SDPA:
    """Create an instance of the default :class:`SDPA`.

    :param attn_dropout_p:
        The dropout probability on attention weights.
    """
    return _sdpa_factory(attn_dropout_p=attn_dropout_p)


class EncoderLayerOutputHook(Protocol):
    """Represents a hook to pass to
    :meth:`~TransformerEncoder.register_layer_output_hook`."""

    def __call__(
        self,
        layer_idx: int,
        layer_output: Tensor,
        layer_padding_mask: Optional[PaddingMask],
        num_layers: int,
    ) -> bool:
        """
        :param layer_idx:
            The index of the layer in the encoder stack.
        :param layer_output:
            The encoded output of the layer.
        :param layer_padding_mask:
            The padding mask of ``layer_output``.
        :param num_layers:
            The number of layers in the encoder stack.

        :returns:
            ``True`` if the encoder should continue executing the remaining
            layers in the stack; ``False`` if the encoder should treat this
            layer as the final layer in the stack.
        """


class TransformerEncoder(Module, ABC):
    """Represents a Transformer encoder."""

    model_dim: int
    layers: ModuleList

    _layer_output_hooks: Dict[int, EncoderLayerOutputHook]

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

        self._layer_output_hooks = OrderedDict()

    @abstractmethod
    def forward(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """
        :param seqs:
            The sequences to encode. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            - The encoder output. *Shape:* Same as ``seqs``.
            - The padding mask of the encoder output. *Shape:* Same as
              ``padding_mask``.
        """

    def register_layer_output_hook(
        self, hook: EncoderLayerOutputHook
    ) -> RemovableHandle:
        """Register a layer output hook on the module.

        The hook will be called every time after a layer in the encoder stack
        has computed an output.

        :param hook:
            The hook to register.

        :returns:
            A handle that can be used to remove the added hook by calling
            ``handle.remove()``.
        """
        handle = RemovableHandle(self._layer_output_hooks)

        self._layer_output_hooks[handle.id] = hook

        return handle

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


class TransformerNormOrder(Enum):
    """Specifies the Layer Normalization order."""

    POST = 0
    """Apply Layer Normalization after each layer's residual connection as
    described in :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    PRE = 1
    """Apply Layer Normalization at the beginning of each layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2002.04745`."""

    PRE_WITH_NORMFORMER = 2
    """Apply Layer Normalization as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2110.09456`."""


class TransformerEncoderLayer(Module, ABC):
    """Represents a Transformer encoder layer."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """
        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param self_attn_mask:
            The mask that will be added to attention weights before computing
            the self attention. *Shape:* :math:`([H],S,S)`, where :math:`H` is
            the number of attention heads and :math:`S` is the sequence length.

        :returns:
            - The encoder layer output. *Shape:* Same as ``seqs``.
            - The padding mask of the encoder layer output. *Shape:* Same as
              ``padding_mask``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


class LayerNormFactory(Protocol):
    """Constructs instances of :class:`LayerNorm`."""

    def __call__(
        self,
        model_dim: int,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> LayerNorm:
        """
        :param model_dim:
            The dimensionality of the model.
        :param device:
            The device on which to initialize the module.
        :param dtype:
            The data type of the module.
        """


class AttentionMaskFactory(Protocol):
    """Constructs instances of :class:`AttentionMask`."""

    def __call__(
        self,
        seqs: Tensor,
        keys: Tensor,
        *,
        training: bool = True,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Optional[AttentionMask]:
        """
        :param seqs:
            The sequences for which to create a mask. *Shape:* :math:`(N,S,M)`,
            where :math:`N` is the batch size, :math:`S` is the sequence length,
            and :math:`M` is the dimensionality of the model.
        :param keys:
            The keys. *Shape:* :math:`(N,S_{kv},K)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`K` is the key size.
        :param training:
            If ``True``, indicates that the calling module is in training mode.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            An implementation-defined mask for ``seqs``.
        """


class AttentionWeightHook(Protocol):
    """Represents a hook to pass to
    :meth:`~MultiheadAttention.register_attn_weight_hook`."""

    def __call__(
        self, m: MultiheadAttention, attn: Tensor, attn_weights: Tensor
    ) -> None:
        """
        :param m:
            The module that has computed the attention weights.
        :param attn:
            The computed attention values. *Shape:* :math:`(N,S,V)`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`V` is the value size.
        :param attn_weights:
            The computed attention weights. *Shape:* :math:`(N,S,S_{kv})`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`S_{kv}` is the key/value sequence length.
        """


class MultiheadAttention(Module, ABC):
    """Represents a Transformer multi-head attention layer."""

    num_heads: int
    model_dim: int

    _attn_weight_hooks: Dict[int, AttentionWeightHook]

    def __init__(self, model_dim: int, num_heads: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param num_heads:
            The number of attention heads.
        """
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        self._attn_weight_hooks = OrderedDict()

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        keys: Tensor,
        key_padding_mask: Optional[PaddingMask],
        values: Tensor,
        *,
        attn_mask: Optional[AttentionMask] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """
        :param seqs:
            The sequences to query. *Shape:* :math:`(N,S,M)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`M` is
            the dimensionality of the model.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param keys:
            The keys. *Shape:* :math:`(N,S_{kv},K)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`K` is the key size.
        :param key_padding_mask:
            The padding mask indicating which key positions to ignore for the
            purpose of attention. *Shape:* :math:`(N,S_{kv})`, where :math:`N`
            is the batch size and :math:`S_{kv}` is the key/value sequence
            length.
        :param values:
            The values. *Shape:* :math:`(N,S_{kv},V)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`V` is the value size.
        :param attn_mask:
            The mask that will be added to attention weights before computing
            the attention. *Shape:* :math:`([H],S,S_{kv})`, where :math:`H` is
            the number of attention heads, :math:`S` is the sequence length, and
            :math:`S_{kv}` is the key/value sequence length.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            The attention values for ``seqs``. *Shape:* :math:`(N,S,M)`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`M` is the dimensionality of the model.
        """

    def register_attn_weight_hook(self, hook: AttentionWeightHook) -> RemovableHandle:
        """Register an attention weight hook on the module.

        The hook will be called every time after the module has computed
        attention weights.

        :param hook:
            The hook to register.

        :returns:
            A handle that can be used to remove the added hook by calling
            ``handle.remove()``.
        """
        handle = RemovableHandle(self._attn_weight_hooks)

        self._attn_weight_hooks[handle.id] = hook

        return handle

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"num_heads={self.num_heads}, model_dim={self.model_dim}"


class RelativePositionalEncoding(Module):
    """Produces relative positional encodings as described in Appendix B of
    :cite:t:`dai2019transformerxl`."""

    encoding_dim: int
    max_seq_len: int
    freqs: Tensor

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param encoding_dim:
            The dimensionality of positional encodings.
        :param max_seq_len:
            The expected maximum sequence length.
        """
        super().__init__()

        if encoding_dim % 2 != 0:
            raise ValueError(
                f"`encoding_dim` must be even, but is {encoding_dim} instead."
            )

        self.encoding_dim = encoding_dim
        self.max_seq_len = max_seq_len

        freqs = torch.empty(
            ((max_seq_len * 2) - 1, encoding_dim), device=device, dtype=dtype
        )

        self.register_buffer("freqs", freqs, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        """Reset the non-persistent buffers of the module."""
        fp32_freqs = self.freqs.float()

        device, dtype = fp32_freqs.device, fp32_freqs.dtype

        positive_half = fp32_freqs[: self.max_seq_len]
        negative_half = fp32_freqs[self.max_seq_len :]

        # (S)
        steps = torch.arange(self.max_seq_len, device=device, dtype=dtype)

        # (E / 2)
        indices = torch.arange(0, self.encoding_dim, step=2, device=device, dtype=dtype)

        freqs = torch.exp(indices * -math.log(10000.0) / self.encoding_dim)

        # (S) x (E / 2) -> (S, E / 2)
        freqs = torch.outer(steps, freqs)

        flipped_freqs = freqs.flip([0])

        # A mirrored matrix of sinusoidal positive and negative positional
        # encodings to use in shift trick.
        #
        # [max, ...,  3,  2,  1,  0, -1, -2, -3, ..., min]
        torch.sin(flipped_freqs, out=positive_half[:, 0::2])
        torch.cos(flipped_freqs, out=positive_half[:, 1::2])

        torch.sin(-1 * freqs[1:], out=negative_half[:, 0::2])
        torch.cos(-1 * freqs[1:], out=negative_half[:, 1::2])

        self.freqs.copy_(fp32_freqs)

    def forward(self, seqs: Tensor) -> Tensor:
        """
        :param seqs:
            The sequences for which to return positional encodings. *Shape:*
            :math:`(*,S,E)`, where :math:`*` is any number of batch dimensions
            including none, :math:`S` is the sequence length, and :math:`E` is
            the dimensionality of the positional encodings.

        :returns:
            The positional encodings to use in shift trick in
            :class:`RelativePositionSDPA`. *Shape:* :math:`(2 x S - 1, E)`,
            where :math:`S` is the sequence length and :math:`E` is the
            dimensionality of the positional encodings.
        """
        seq_len = seqs.size(-2)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"The input sequence length must be less than or equal to the maximum sequence length ({self.max_seq_len}), but is {seq_len} instead."
            )

        return self.freqs[self.max_seq_len - seq_len : self.max_seq_len + seq_len - 1]

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"encoding_dim={self.encoding_dim}, max_seq_len={self.max_seq_len}"



@final
class CustomAttentionMask(AttentionMask):
    """Represents a custom attention mask provided by the user."""

    def __init__(self, mask: Tensor) -> None:
        """
        :param mask:
            The custom attention mask tensor.
        """
        super().__init__()

        self.mask = mask

    @finaloverride
    def _do_materialize(self) -> Tensor:
        return self.mask


@final
class RelativePositionSDPA(SDPA):
    """Computes scaled dot-product attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1901.02860`."""

    model_dim: int
    num_heads: int
    pos_encoding: RelativePositionalEncoding
    u_bias: Parameter
    v_bias: Parameter
    r_proj: Linear
    inner_sdpa: SDPA

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        pos_encoding: RelativePositionalEncoding,
        *,
        inner_sdpa: Optional[SDPA] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param: num_heads:
            The number of attention heads.
        :param: pos_encoding:
            The relative positional encoding table.
        :param inner_sdpa:
            The actual :class:`SDPA` module to compute head attentions.
        """
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError(
                f"`model_dim` must be a multiple of `num_heads` ({num_heads}), but is {model_dim} instead."
            )

        self.model_dim = model_dim
        self.num_heads = num_heads

        if pos_encoding.encoding_dim != model_dim:
            raise ValueError(
                f"`encoding_dim` of `pos_encoding` must be equal to `model_dim` ({model_dim}), but is {pos_encoding.encoding_dim} instead."
            )

        self.pos_encoding = pos_encoding

        head_dim = model_dim // num_heads

        self.u_bias = Parameter(
            torch.empty((num_heads, head_dim), device=device, dtype=dtype)
        )
        self.v_bias = Parameter(
            torch.empty((num_heads, head_dim), device=device, dtype=dtype)
        )

        self.r_proj = Linear(
            model_dim, model_dim, bias=False, device=device, dtype=dtype
        )

        if inner_sdpa is not None:
            self.inner_sdpa = inner_sdpa
        else:
            self.inner_sdpa = create_default_sdpa()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.xavier_normal_(self.u_bias)
        nn.init.xavier_normal_(self.v_bias)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        keys: Tensor,
        key_padding_mask: Optional[PaddingMask],
        values: Tensor,
        *,
        attn_mask: Optional[AttentionMask] = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        q = seqs
        k = keys

        # (H, K_h) -> (H, 1, K_h)
        u_bias = self.u_bias.unsqueeze(1)
        v_bias = self.v_bias.unsqueeze(1)

        # (N, H, S, K_h) + (H, 1, K_h) -> (N, H, S, K_h)
        q_with_u_bias = q + u_bias
        q_with_v_bias = q + v_bias

        # (N, H, 2 x S - 1, K_h)
        r = self._compute_r(k, batch_size=q.size(0))

        # (N, H, S, K_h) @ (N, H, K_h, 2 x S - 1) = (N, H, S, 2 x S - 1)
        bd = torch.matmul(q_with_v_bias, r.transpose(-1, -2))

        # (N, H, S, 2 x S - 1) -> (N, H, S, S)
        bd = self._shift_bd(bd)

        # We treat `bd` as an attention mask to take advantage of efficient SDPA
        # implementations.
        bd = bd * (q.size(-1) ** -0.5)

        if attn_mask is None:
            mask = bd
        else:
            mask = bd + attn_mask.materialize()

        attn_mask = CustomAttentionMask(mask)

        return self.inner_sdpa(  # type: ignore[no-any-return]
            q_with_u_bias,
            k,
            key_padding_mask,
            values,
            attn_mask=attn_mask,
            needs_weights=needs_weights,
        )

    def _compute_r(self, k: Tensor, batch_size: int) -> Tensor:
        # (2 x S - 1, K)
        r = self.pos_encoding(k)

        # (2 x S - 1, K) -> (2 x S - 1, K)
        r = self.r_proj(r)

        # (2 x S - 1, K) -> (1, 2 x S - 1, H, K_h)
        r = r.view(1, -1, self.num_heads, k.size(-1))

        # (1, 2 x S - 1, H, K_h) -> (N, H, 2 x S - 1, K_h)
        r = r.transpose(1, 2).expand(batch_size, -1, -1, -1)

        return r  # type: ignore[no-any-return]

    def _shift_bd(self, bd: Tensor) -> Tensor:
        # (N, H, S, 2 x S - 1) -> (N, H, S, 2 x S)
        x = pad(bd, (1, 0))

        # (N, H, S, 2 x S) -> (N, H, 2 x S, S)
        x = x.view(x.size(0), x.size(1), x.size(3), x.size(2))

        # Discard the first set of positive positions.
        # (N, H, 2 x S, S) -> (N, H, 2 x S - 1, S)
        x = x[:, :, 1:, :]

        # This op effectively shifts each row by an extra step.
        # (N, H, 2 x S - 1, S) -> (N, H, S, 2 x S - 1)
        x = x.view_as(bd)

        # Discard positions used for shift.
        # (N, H, S, 2 x S - 1) -> (N, H, S, S)
        x = x[..., : bd.size(2)]

        return x

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}, num_heads={self.num_heads}"



class AttentionState(IncrementalState):
    """Holds the projected keys and values of a :class:`MultiheadAttention`
    module during incremental decoding."""

    @abstractmethod
    def append(self, k: Tensor, v: Tensor) -> None:
        """Update the state with ``k`` and ``v``.

        :param k:
            The projected keys of the current step. *Shape:*
            :math:`(N,H,1,K_{proj})`, where :math:`N` is the batch size,
            :math:`H` is the number of heads, :math:`1` is the step length, and
            :math:`K_{proj}` is the projected key size.
        :param v:
            The projected values of the current step. *Shape:*
            :math:`(N,H,1,V_{proj})`, where :math:`N` is the batch size,
            :math:`H` is the number of heads, :math:`1` is the step length, and
            :math:`V_{proj}` is the projected value size.
        """

    @abstractmethod
    def get(self) -> Tuple[Tensor, Tensor]:
        """Return the state that should be used to compute the attention.

        :returns:
            - The projected keys.
            - The projected values.
        """


class AttentionStateFactory(Protocol):
    """Constructs instances of :class:`AttentionState`."""

    def __call__(self, k: Tensor, v: Tensor, max_seq_len: int) -> AttentionState:
        """
        :param k:
            The initial projected keys.
        :param v:
            The initial projected values.
        :param max_seq_len:
            The expected maximum sequence length.
        """



def init_qkv_projection(proj: Linear) -> None:
    """Initialize ``proj`` as a multi-head attention input projection."""
    # Empirically observed the convergence to be much better with the scaled
    # initialization.
    nn.init.xavier_uniform_(proj.weight, gain=2**-0.5)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)



def init_output_projection(proj: Linear) -> None:
    """Initialize ``proj`` as a multi-head attention output projection."""
    nn.init.xavier_uniform_(proj.weight)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)


@final
class FullAttentionState(AttentionState):
    """Holds the past projected keys and values of a :class:`MultiheadAttention`
    module during incremental decoding."""

    seq_len: int
    """The current sequence length of :attr:`k` and :attr:`v`."""

    k: Tensor
    """The projected keys accumulated from the past decoding steps. *Shape:*
    :math:`(N,H,S_{res},K_{proj})`, where :math:`N` is the batch size, :math:`H`
    is the number of heads, :math:`S_{res}` is the reserved sequence length
    capacity, and :math:`K_{proj}` is the projected key size."""

    v: Tensor
    """The projected values accumulated from the past decoding steps. *Shape:*
    :math:`(N,H,S_{res},V_{proj})`, where :math:`N` is the batch size, :math:`H`
    is the number of heads, :math:`S_{res}` is the reserved sequence length
    capacity, and :math:`V_{proj}` is the projected value size."""

    def __init__(self, k: Tensor, v: Tensor, max_seq_len: int) -> None:
        batch_size, num_heads, seq_len, head_dim = k.shape

        self.k = k.new_empty((batch_size, num_heads, max_seq_len, head_dim))
        self.v = v.new_empty((batch_size, num_heads, max_seq_len, head_dim))

        self.k[:, :, :seq_len] = k
        self.v[:, :, :seq_len] = v

        self.seq_len = seq_len

    @finaloverride
    def append(self, k: Tensor, v: Tensor) -> None:
        pos = self.seq_len

        self.k[:, :, pos : pos + 1] = k
        self.v[:, :, pos : pos + 1] = v

        self.seq_len += 1

    @finaloverride
    def get(self) -> Tuple[Tensor, Tensor]:
        k = self.k[:, :, : self.seq_len]
        v = self.v[:, :, : self.seq_len]

        return k, v

    @finaloverride
    def reorder(self, new_order: Tensor) -> None:
        self.k = self.k.index_select(0, new_order)
        self.v = self.v.index_select(0, new_order)


@final
class StaticAttentionState(AttentionState):
    """Holds the static projected keys and values (e.g. encoder-decoder) of a
    :class:`MultiheadAttention` module during incremental decoding."""

    k: Tensor
    v: Tensor

    def __init__(self, k: Tensor, v: Tensor, max_seq_len: int) -> None:
        self.k = k
        self.v = v

    @finaloverride
    def append(self, k: Tensor, v: Tensor) -> None:
        raise ValueError("`append()` on `StaticAttentionState` is not supported.")

    @finaloverride
    def get(self) -> Tuple[Tensor, Tensor]:
        return self.k, self.v

    @finaloverride
    def reorder(self, new_order: Tensor) -> None:
        self.k = self.k.index_select(0, new_order)
        self.v = self.v.index_select(0, new_order)


@final
class StandardMultiheadAttention(MultiheadAttention):
    """Represents a Transformer multi-head attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    num_key_value_heads: int
    q_proj: Projection
    k_proj: Projection
    v_proj: Projection
    attn_mask_factory: Optional[AttentionMaskFactory]
    pos_encoder: Optional[PositionEncoder]
    bias_k: Optional[Parameter]
    bias_v: Optional[Parameter]
    add_zero_attn: bool
    sdpa: SDPA
    head_scale_weight: Optional[Parameter]
    output_proj: Projection
    state_factory: Optional[AttentionStateFactory]

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        num_key_value_heads: Optional[int] = None,
        q_proj: Optional[Projection] = None,
        k_proj: Optional[Projection] = None,
        v_proj: Optional[Projection] = None,
        attn_mask_factory: Optional[AttentionMaskFactory] = None,
        pos_encoder: Optional[PositionEncoder] = None,
        sdpa: Optional[SDPA] = None,
        scale_heads: bool = False,
        output_proj: Optional[Projection] = None,
        bias: bool = True,
        state_factory: Optional[AttentionStateFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param num_heads:
            The number of attention heads.
        :param num_key_value_heads:
            The number of key/value heads for Grouped Query Attention as
            described in :cite:t:`https://doi.org/10.48550/arXiv.2305.13245`.
            If ``None`` or set to ``num_heads``, it is equivalent to standard
            Multi Head Attention (MHA); if set to 1, it is equivalent to Multi
            Query Attention (MQA).
        :param q_proj:
            The projection to apply to sequences before computing attention. If
            ``None``, a default projection will be used.
        :param k_proj:
            The projection to apply to keys before computing attention. If
            ``None``, a default projection will be used.
        :param v_proj:
            The projection to apply to values before computing attention. If
            ``None``, a default projection will be used.
        :param attn_mask_factory:
            The attention mask factory.
        :param pos_encoder:
            The position encoder to apply to sequences and keys after projection.
        :param sdpa:
            The :class:`SDPA` module to compute head attentions. If ``None``, a
            default implementation will be used.
        :param scale_heads:
            If ``True``, applies head scaling as described in
            :cite:t:`https://doi.org/10.48550/arxiv.2110.09456`
        :param output_proj:
            The projection to produce final attentions. If ``None``, a default
            projection will be used.
        :param bias:
            If ``True``, query, key, value, and output projections learn an
            additive bias. Ignored for explicitly specified projections.
        :param state_factory:
            The factory to construct :class:`AttentionState` instances for
            incremental decoding.
        """
        super().__init__(model_dim, num_heads)

        if num_key_value_heads is None:
            self.num_key_value_heads = num_heads
        else:
            if num_heads < num_key_value_heads:
                raise ValueError(
                    f"`num_heads` must be greater than or equal to `num_key_value_heads` ({num_key_value_heads}), but is {num_heads} instead."
                )

            if num_heads % num_key_value_heads != 0:
                raise ValueError(
                    f"`num_heads` must be a multiple of `num_key_value_heads` ({num_key_value_heads}), but is {num_heads} instead."
                )

            self.num_key_value_heads = num_key_value_heads

        head_dim = model_dim // num_heads

        num_query_groups = num_heads // self.num_key_value_heads

        if q_proj is None and k_proj is None and v_proj is None:
            q_proj = Linear(
                model_dim,
                model_dim,
                bias,
                init_fn=init_qkv_projection,
                device=device,
                dtype=dtype,
            )
            k_proj = Linear(
                model_dim,
                head_dim * self.num_key_value_heads,
                bias,
                init_fn=init_qkv_projection,
                device=device,
                dtype=dtype,
            )
            v_proj = Linear(
                model_dim,
                head_dim * self.num_key_value_heads,
                bias,
                init_fn=init_qkv_projection,
                device=device,
                dtype=dtype,
            )
        else:
            if q_proj is None or k_proj is None or v_proj is None:
                raise ValueError(
                    "`q_proj`, `k_proj`, and `v_proj` must be all specified."
                )

            if q_proj.input_dim != model_dim:
                raise ValueError(
                    f"`input_dim` of `q_proj` must be equal to `model_dim` ({model_dim}), but is {q_proj.input_dim} instead."
                )

            if (k_dim := k_proj.output_dim * num_query_groups) != q_proj.output_dim:
                raise ValueError(
                    f"`output_dim` of `q_proj` and `output_dim` of `k_proj` (times the number of query groups when GQA) must be equal, but are {q_proj.output_dim} and {k_dim} instead."
                )

            if k_proj.output_dim % self.num_key_value_heads != 0:
                raise ValueError(
                    f"`output_dim` of `k_proj` must be a multiple of `num_key_value_heads` ({self.num_key_value_heads}), but is {k_proj.output_dim} instead."
                )

            if v_proj.output_dim % self.num_key_value_heads != 0:
                raise ValueError(
                    f"`output_dim` of `v_proj` must be a multiple of `num_key_value_heads` ({self.num_key_value_heads}), but is {v_proj.output_dim} instead."
                )

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

        self.attn_mask_factory = attn_mask_factory

        if pos_encoder is not None:
            if head_dim != pos_encoder.encoding_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` must be equal to the size of the header dimension ({head_dim}), but is {pos_encoder.encoding_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if sdpa is not None:
            self.sdpa = sdpa
        else:
            self.sdpa = create_default_sdpa()

        if scale_heads:
            self.head_scale_weight = Parameter(
                torch.empty(num_heads, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("head_scale_weight", None)

        v_dim = v_proj.output_dim * num_query_groups

        if output_proj is None:
            self.output_proj = Linear(
                v_dim,
                model_dim,
                bias,
                init_fn=init_output_projection,
                device=device,
                dtype=dtype,
            )
        else:
            if v_dim != output_proj.input_dim:
                raise ValueError(
                    f"`output_dim` of `v_proj` (times the number of query groups when GQA) and `input_dim` of `output_proj` must be equal, but are {v_dim} and {output_proj.input_dim} instead."
                )

            if output_proj.output_dim != model_dim:
                raise ValueError(
                    f"`output_dim` of `output_proj` must be equal to `model_dim` ({model_dim}), but is {output_proj.output_dim} instead."
                )

            self.output_proj = output_proj

        self.state_factory = state_factory

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.head_scale_weight is not None:
            nn.init.ones_(self.head_scale_weight)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        keys: Tensor,
        key_padding_mask: Optional[PaddingMask],
        values: Tensor,
        *,
        attn_mask: Optional[AttentionMask] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        # (N, S, M) -> (N, H, S, K_h)
        q = self._project_q(seqs, padding_mask, state_bag)

        if self.training or state_bag is None:
            # k: (N, S_kv, M) -> (N, H_kv, S_kv, K_h)
            # v: (N, S_kv, M) -> (N, H_kv, S_kv, V_h)
            k, v = self._project_kv(keys, key_padding_mask, values)
        else:
            if seqs is keys:  # Self attention
                if key_padding_mask is not None:
                    raise ValueError(
                        "`key_padding_mask` must be `None` during incremental decoding."
                    )

                # k: (N, S_step, M) -> (N, H_kv, S_step, K_h)
                # v: (N, S_step, M) -> (N, H_kv, S_step, V_h)
                k, v = self._project_kv(keys, key_padding_mask, values, state_bag)

                state = state_bag.get_state(self, AttentionState)
                if state is None:
                    state_factory = self.state_factory or FullAttentionState

                    state = state_factory(k, v, state_bag.max_num_steps)

                    state_bag.set_state(self, state)
                else:
                    state.append(k, v)

                    # k: (N, H_kv, S_kv, K_h)
                    # v: (N, H_kv, S_kv, V_h)
                    k, v = state.get()
            else:
                state = state_bag.get_state(self, AttentionState)
                if state is None:
                    # k: (N, S_kv, M) -> (N, H_kv, S_kv, K_h)
                    # v: (N, S_kv, M) -> (N, H_kv, S_kv, V_h)
                    k, v = self._project_kv(keys, key_padding_mask, values)

                    state_factory = self.state_factory or StaticAttentionState

                    state = state_factory(k, v, max_seq_len=k.size(2))

                    state_bag.set_state(self, state)
                else:
                    # k: (N, H_kv, S_kv, K_h)
                    # v: (N, H_kv, S_kv, V_h)
                    k, v = state.get()

        # With Grouped Query Attention, each key/value head is repeated.
        if (num_query_groups := self.num_heads // self.num_key_value_heads) > 1:
            # (N, H_kv, S_kv, K_h) -> (N, H, S_kv, K_h)
            k = repeat_interleave(k, dim=1, repeat=num_query_groups)
            # (N, H_kv, S_kv, K_h) -> (N, H, S_kv, V_h)
            v = repeat_interleave(v, dim=1, repeat=num_query_groups)

        if self.attn_mask_factory is not None:
            attn_mask = self.attn_mask_factory(
                seqs, keys=keys, training=self.training, state_bag=state_bag
            )

        needs_weights = len(self._attn_weight_hooks) > 0

        # attn:         (N, H, S, V_h)
        # attn_weights: (N, H, S, S_kv)
        attn, attn_weights = self.sdpa(
            q,
            k,
            key_padding_mask,
            v,
            attn_mask=attn_mask,
            needs_weights=needs_weights,
        )

        if attn_weights is not None:
            for hook in self._attn_weight_hooks.values():
                hook(self, attn, attn_weights)

        # (N, H, S, V_h) -> (N, S, H, V_h)
        attn = attn.transpose(1, 2)

        if self.head_scale_weight is not None:
            attn = torch.einsum("nshv,h->nshv", attn, self.head_scale_weight)

        # (N, S, H, V_h) -> (N, S, V_proj)
        attn = attn.flatten(2, 3)

        # (N, S, V_proj) -> (N, S, M)
        attn = self.output_proj(attn)

        return attn  # type: ignore[no-any-return]

    def _project_q(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        # (N, S, M) -> (N, S, K_proj)
        q = self.q_proj(seqs)

        # (N, S, K_proj) -> (N, H, S, K_h)
        q = q.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

        if self.pos_encoder is not None:
            q = self.pos_encoder(q, padding_mask, state_bag=state_bag)

        return q  # type: ignore[no-any-return]

    def _project_kv(
        self,
        keys: Tensor,
        key_padding_mask: Optional[PaddingMask],
        values: Tensor,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Tensor]:
        # (N, S, K) -> (N, S, K_proj)
        k = self.k_proj(keys)
        # (N, S, V) -> (N, S, V_proj)
        v = self.v_proj(values)

        # (N, S, K_proj) -> (N, H, S, K_h)
        k = k.unflatten(-1, (self.num_key_value_heads, -1)).transpose(1, 2)
        # (N, S, V_proj) -> (N, H, S, V_h)
        v = v.unflatten(-1, (self.num_key_value_heads, -1)).transpose(1, 2)

        if self.pos_encoder is not None:
            k = self.pos_encoder(k, key_padding_mask, state_bag=state_bag)

        return k, v

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.num_key_value_heads != self.num_heads:
            s = f"{s}, num_key_value_heads={self.num_key_value_heads}"

        if self.state_factory is not None:
            state_factory = getattr(self.state_factory, "__name__", self.state_factory)

            s = f"{s}, state_factory={state_factory}"

        return s


class FeedForwardNetwork(Module, ABC):
    """Represents a Transformer feed-forward network."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    @abstractmethod
    def forward(self, seqs: Tensor) -> Tensor:
        """
        :param seqs:
            The sequences to project. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.

        :returns:
            The projected sequences. *Shape:* Same as ``seqs``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"



@final
class StandardFeedForwardNetwork(FeedForwardNetwork):
    """Represents a Transformer feed-forward network as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    inner_proj: Linear
    inner_activation: Module
    inner_dropout: Optional[Dropout]
    inner_norm: Optional[LayerNorm]
    output_proj: Linear

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        bias: bool,
        *,
        inner_activation: Optional[Module] = None,
        inner_dropout_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param inner_dim:
            The dimensionality of the inner projection layer.
        :param bias:
            If ``True``, both the inner and output projection learn an additive
            bias.
        :param inner_activation:
            The activation to apply to outputs of the inner projection layer. If
            ``None``, :func:`~torch.nn.ReLU` will be used.
        :param inner_dropout_p:
            The dropout probability on outputs of the inner projection layer.
        :param norm_order:
            The Layer Normalization order.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization module.
        """
        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.inner_proj = Linear(model_dim, inner_dim, bias, device=device, dtype=dtype)

        if inner_activation is None:
            self.inner_activation = ReLU()
        else:
            self.inner_activation = inner_activation

        if inner_dropout_p > 0.0:
            self.inner_dropout = Dropout(inner_dropout_p)
        else:
            self.register_module("inner_dropout", None)

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.inner_layer_norm = layer_norm_factory(
                inner_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("inner_layer_norm", None)

        self.output_proj = Linear(
            inner_dim, model_dim, bias, device=device, dtype=dtype
        )

    @finaloverride
    def forward(self, seqs: Tensor) -> Tensor:
        seqs = self.inner_proj(seqs)

        seqs = self.inner_activation(seqs)

        if self.inner_layer_norm is not None:
            seqs = self.inner_layer_norm(seqs)

        if self.inner_dropout is not None:
            seqs = self.inner_dropout(seqs)

        seqs = self.output_proj(seqs)

        return seqs


def create_standard_layer_norm(
    model_dim: int, *, device: Optional[Device] = None, dtype: Optional[DataType] = None
) -> LayerNorm:
    """Create an instance of :class:`StandardLayerNorm`."""
    return StandardLayerNorm(model_dim, bias=True, device=device, dtype=dtype)
