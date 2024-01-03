from typing import Optional, Tuple, Literal, Optional

from torch import Tensor
from torch.nn import Dropout, GLU, BatchNorm1d, Conv1d, Module, SiLU
from torch.nn.functional import pad

from fairseq.nn.normalization import LayerNorm
from fairseq.nn.padding import PaddingMask, apply_padding_mask
from fairseq.nn.transformer import (
    AttentionMask,
    FeedForwardNetwork,
    LayerNormFactory,
    MultiheadAttention,
    TransformerEncoderLayer,
    create_standard_layer_norm,
)
from fairseq.nn.normalization import LayerNorm, StandardLayerNorm
from fairseq.typing import DataType, Device, finaloverride



class ConformerConvolution(Module):
    """Represents a Conformer convolution module as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2005.08100`."""

    model_dim: int
    pointwise_conv1: Conv1d
    pointwise_conv1_activation: GLU
    depthwise_conv: Conv1d
    causal_depthwise_conv: bool
    batch_norm: Optional[BatchNorm1d]
    layer_norm: Optional[LayerNorm]
    depthwise_activation: Module
    pointwise_conv2: Conv1d

    def __init__(
        self,
        model_dim: int,
        depthwise_kernel_size: int,
        *,
        causal_depthwise_conv: bool = False,
        norm_type: Literal["batch_norm", "layer_norm"] = "batch_norm",
        depthwise_activation: Optional[Module] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param depthwise_kernel_size:
            The kernel size of the depthwise convolution.
        :param causal_depthwise_conv:
            If ``True``, uses a causal depthwise convolution similar to that
            described in Section 2.1 of :cite:t:`https://doi.org/10.48550/arxiv.1609.03499`.
        :param norm_type:
            The type of normalization to apply after the depthwise convolution.
        :param depthwise_activation:
            The activation to apply to outputs of the depthwise convolution. If
            ``None``, :func:`~torch.nn.SiLU` (a.k.a. swish) will be used.
        """
        super().__init__()

        self.model_dim = model_dim

        # We treat the dimensionality of the model as the number of input
        # channels to the first pointwise convolution.
        self.pointwise_conv1 = Conv1d(
            model_dim,
            # We apply GLU to outputs to bring them back to `model_dim`.
            model_dim * 2,
            kernel_size=1,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.pointwise_conv1_activation = GLU(dim=1)

        self.depthwise_conv = Conv1d(
            model_dim,
            model_dim,
            depthwise_kernel_size,
            padding="same" if not causal_depthwise_conv else 0,
            # We want to perform depthwise convolution.
            groups=model_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.causal_depthwise_conv = causal_depthwise_conv

        if norm_type not in ("batch_norm", "layer_norm"):
            raise ValueError(
                f"`norm_type` must be 'batch_norm' or 'layer_norm', but is '{norm_type}' instead."
            )

        if norm_type == "batch_norm":
            self.batch_norm = BatchNorm1d(model_dim, device=device, dtype=dtype)
        else:
            self.register_module("batch_norm", None)

        if norm_type == "layer_norm":
            self.layer_norm = StandardLayerNorm(
                model_dim, bias=True, device=device, dtype=dtype
            )
        else:
            self.register_module("layer_norm", None)

        if depthwise_activation is None:
            self.depthwise_activation = SiLU()  # a.k.a. swish
        else:
            self.depthwise_activation = depthwise_activation

        self.pointwise_conv2 = Conv1d(
            model_dim, model_dim, kernel_size=1, bias=False, device=device, dtype=dtype
        )

    def forward(self, seqs: Tensor, padding_mask: Optional[PaddingMask]) -> Tensor:
        """
        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            The processed sequences. *Shape:* Same as ``seqs``.
        """
        # Ensure that we do not leak padded positions in depthwise convolution.
        seqs = apply_padding_mask(seqs, padding_mask)

        # (N, S, M) -> (N, M, S)
        seqs = seqs.transpose(1, 2)

        # This is mathematically equivalent to a dot-product.
        # (N, M, S) -> (N, 2 * M, S)
        seqs = self.pointwise_conv1(seqs)

        # (N, 2 * M, S) -> (N, M, S)
        seqs = self.pointwise_conv1_activation(seqs)

        # Pad the sequence entirely on the left in case of a causal convolution.
        if self.causal_depthwise_conv:
            seqs = pad(seqs, (self.depthwise_conv.kernel_size[0] - 1, 0))

        # (N, M, S) -> (N, M, S)
        seqs = self.depthwise_conv(seqs)

        if self.batch_norm is not None:
            seqs = self.batch_norm(seqs)
        else:
            assert self.layer_norm is not None

            # (N, M, S) -> (N, S, M)
            seqs = seqs.transpose(1, 2)

            seqs = self.layer_norm(seqs)

            # (N, S, M) -> (N, M, S)
            seqs = seqs.transpose(1, 2)

        seqs = self.depthwise_activation(seqs)

        # This is mathematically equivalent to a dot-product.
        # (N, M, S) -> (N, M, S)
        seqs = self.pointwise_conv2(seqs)

        # (N, M, S) -> (N, S, M)
        seqs = seqs.transpose(1, 2)

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"



class ConformerBlock(TransformerEncoderLayer):
    """Represents a Conformer block as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2005.08100`."""

    ffn1_layer_norm: LayerNorm
    ffn1: FeedForwardNetwork
    ffn1_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    self_attn: MultiheadAttention
    self_attn_dropout: Optional[Dropout]
    conv_layer_norm: LayerNorm
    conv: ConformerConvolution
    conv_dropout: Optional[Dropout]
    ffn2_layer_norm: LayerNorm
    ffn2: FeedForwardNetwork
    ffn2_dropout: Optional[Dropout]
    layer_norm: LayerNorm

    def __init__(
        self,
        ffn1: FeedForwardNetwork,
        self_attn: MultiheadAttention,
        conv: ConformerConvolution,
        ffn2: FeedForwardNetwork,
        *,
        dropout_p: float = 0.1,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param ffn1:
            The bottom macaron-like feed-forward network.
        :param self_attn:
            The self attention layer.
        :param conv:
            The Conformer convolution module.
        :param ffn2:
            The top macaron-like feed-forward network.
        :param dropout_p:
            The dropout probability on outputs of the self attention layer, the
            feed-forward networks, and the Conformer convolution module.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization modules.
        """
        model_dim = self_attn.model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.ffn1_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        self.ffn1 = ffn1

        if dropout_p > 0.0:
            self.ffn1_dropout = Dropout(dropout_p)
        else:
            self.register_module("ffn1_dropout", None)

        self.self_attn_layer_norm = layer_norm_factory(
            model_dim, device=device, dtype=dtype
        )

        self.self_attn = self_attn

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)

        self.conv_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        self.conv = conv

        if dropout_p > 0.0:
            self.conv_dropout = Dropout(dropout_p)
        else:
            self.register_module("conv_dropout", None)

        self.ffn2_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        self.ffn2 = ffn2

        if dropout_p > 0.0:
            self.ffn2_dropout = Dropout(dropout_p)
        else:
            self.register_module("ffn2_dropout", None)

        self.layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs = self._forward_ffn1(seqs)

        seqs = self._forward_self_attn(seqs, padding_mask, self_attn_mask)

        seqs = self._forward_conv(seqs, padding_mask)

        seqs = self._forward_ffn2(seqs)

        seqs = self.layer_norm(seqs)

        return seqs, padding_mask

    def _forward_ffn1(self, seqs: Tensor) -> Tensor:
        residual = seqs

        seqs = self.ffn1_layer_norm(seqs)

        seqs = self.ffn1(seqs) * 0.5

        if self.ffn1_dropout is not None:
            seqs = self.ffn1_dropout(seqs)

        return seqs + residual

    def _forward_self_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask],
    ) -> Tensor:
        residual = seqs

        seqs = self.self_attn_layer_norm(seqs)

        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            key_padding_mask=padding_mask,
            values=seqs,
            attn_mask=self_attn_mask,
        )

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        return seqs + residual

    def _forward_conv(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tensor:
        residual = seqs

        seqs = self.conv_layer_norm(seqs)

        seqs = self.conv(seqs, padding_mask)

        if self.conv_dropout is not None:
            seqs = self.conv_dropout(seqs)

        return seqs + residual

    def _forward_ffn2(self, seqs: Tensor) -> Tensor:
        residual = seqs

        seqs = self.ffn2_layer_norm(seqs)

        seqs = self.ffn2(seqs) * 0.5

        if self.ffn2_dropout is not None:
            seqs = self.ffn2_dropout(seqs)

        return seqs + residual
