from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Iterable, Sequence, Tuple, List, final
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Conv1d, Dropout, GroupNorm, Parameter, GELU, Module, Sequential, SiLU
from torch.nn.functional import cross_entropy, layer_norm, group_norm, gumbel_softmax
from torch.nn.utils.weight_norm import remove_weight_norm, weight_norm
from fairseq.models.conformer import ConformerBlock, ConformerConvolution
from fairseq.models.feature_extractor import SequenceFeatureExtractor
from fairseq.models.transformer import TransformerFrontend
from fairseq.models.sequence import SequenceBatch
from fairseq.nn.transformer import SDPA, RelativePositionSDPA, MultiheadAttention, StandardMultiheadAttention, FeedForwardNetwork, StandardFeedForwardNetwork, AttentionMask, TransformerEncoderLayer, TransformerNormOrder, TransformerEncoder, AttentionMaskFactory, LayerNormFactory, create_standard_layer_norm, create_default_sdpa
from fairseq.nn.incremental_state import IncrementalStateBag
from fairseq.nn.normalization import LayerNorm, StandardLayerNorm
from fairseq.nn.ops import repeat_interleave
from fairseq.nn.padding import PaddingMask, apply_padding_mask
from fairseq.nn.position_encoder import PositionEncoder, RotaryEncoder
from fairseq.nn.projection import Linear
from fairseq.nn.utils.mask import compute_row_mask
from fairseq.nn.utils.grad import scale_grad
from fairseq.nn.module_list import ModuleList
from fairseq.typing import DataType, Device, finaloverride, override


class Wav2Vec2Masker(Module):
    """Masks extracted features as described in Section 3.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    temporal_span_len: int
    max_temporal_mask_prob: float
    temporal_mask_embed: Parameter
    spatial_span_len: int
    max_spatial_mask_prob: float

    def __init__(
        self,
        model_dim: int,
        temporal_span_len: int = 10,
        max_temporal_mask_prob: float = 0.65,
        spatial_span_len: int = 10,
        max_spatial_mask_prob: float = 0.0,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param temporal_span_len:
            The length of each temporal mask span that is applied over time
            steps.
        :param max_temporal_mask_prob:
            The maximum probability of masking a time step. Note that, due to
            mask span overlap, the effective probability might be smaller.
        :param spatial_span_len:
            The length of each spatial mask span that is applied over features.
        :param max_spatial_mask_prob:
            The maximum probability of masking a feature. Note that, due to mask
            span overlap, the effective probability might be smaller.
        """
        super().__init__()

        if max_temporal_mask_prob == 0.0:
            raise ValueError("`max_temporal_mask_prob` must be greater than 0.")

        self.temporal_span_len = temporal_span_len
        self.max_temporal_mask_prob = max_temporal_mask_prob

        self.temporal_mask_embed = Parameter(
            torch.empty((model_dim,), device=device, dtype=dtype)
        )

        self.spatial_span_len = spatial_span_len
        self.max_spatial_mask_prob = max_spatial_mask_prob

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.uniform_(self.temporal_mask_embed)

    def forward(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Tensor]:
        """
        :param seqs:
            The sequences to mask. *Shape:* :math:`(N,S,M)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`M` is
            the dimensionality of the model.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.

        :returns:
            - The input sequences with mask applied. *Shape:* Same as ``seqs``.
            - The temporal mask that has been applied to ``seqs``. *Shape:*
              :math:`(N,S)`, where :math:`N` is the batch size and :math`S` is
              the sequence length.
        """
        batch_size, seq_len, model_dim = seqs.shape

        # Temporal mask over time steps.
        temporal_mask = compute_row_mask(
            shape=(batch_size, seq_len),
            span_len=self.temporal_span_len,
            max_mask_prob=self.max_temporal_mask_prob,
            row_lens=padding_mask.seq_lens if padding_mask is not None else None,
            min_num_spans=2,
            device=seqs.device,
        )

        assert temporal_mask is not None

        seqs[temporal_mask] = self.temporal_mask_embed

        if self.max_spatial_mask_prob > 0.0:
            # Spatial mask over features.
            # (N, M)
            spatial_mask = compute_row_mask(
                shape=(batch_size, model_dim),
                span_len=self.spatial_span_len,
                max_mask_prob=self.max_spatial_mask_prob,
                min_num_spans=2,
                device=seqs.device,
            )

            assert spatial_mask is not None

            # (N, M) -> (N, S, M)
            spatial_mask = spatial_mask.unsqueeze(1).expand(-1, seq_len, -1)

            seqs[spatial_mask] = 0.0

        return seqs, temporal_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"temporal_span_len={self.temporal_span_len}, "
            f"max_temporal_mask_prob={self.max_temporal_mask_prob}, "
            f"spatial_span_len={self.spatial_span_len}, "
            f"max_spatial_mask_prob={self.max_spatial_mask_prob}"
        )


def extract_masked_elements(seqs: Tensor, temporal_mask: Tensor) -> Tensor:
    """Extract masked elements from ``seqs``.

    :param seqs:
        The sequences. *Shape:* :math:`(N,S,M)`, where :math:`N` is the batch
        size, :math:`S` is the sequence length, and :math:`M` is the
        dimensionality of the model.
    :param temporal_mask:
        The temporal mask. *Shape:* :math:`(N,S)`, where :math:`N` is the batch
        size and :math`S` is the sequence length.
    """
    batch_size = seqs.size(0)

    # (N, S, M) -> (N x T, M)
    seqs = seqs[temporal_mask]

    # (N x T, M) -> (N, T, M)
    return seqs.unflatten(0, (batch_size, -1))  # type: ignore[no-any-return]


@dataclass
class Wav2Vec2EncoderConfig:
    """Holds the configuration of a wav2vec 2.0 encoder."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The expected maximum sequence length after feature extraction."""

    # Features
    feature_dim: int
    """The dimensionality of extracted features."""

    use_fbank: bool
    """If ``True``, uses log-mel filterbanks instead of waveforms as input."""

    first_pass_dropout_p: float
    """The dropout probability on extracted features before masking and
    positional encoding."""

    layer_norm_features: bool
    """If ``True``, applies Layer Normalization to extracted features."""

    # Waveform Feature Extractor
    feature_extractor_layer_descs: List[Tuple[int, int, int]]
    """A tuple of output dimension, kernel size, and stride for each feature
    extraction layer."""

    feature_extractor_bias: bool
    """If ``True``, convolutions in feature extraction layers learn an additive
    bias."""

    feature_extractor_layer_norm_convs: bool
    """If ``True``, applies Layer Normalization to outputs of convolutions in
    feature extraction layers."""

    feature_grad_scale: float
    """The scale factor for gradients of extracted features. Setting to a value
    less than 1.0 allows the feature extractor to learn at a lower rate than the
    rest of the model."""

    # Filterbank Feature Extractor
    num_fbank_channels: int
    """The number of source log-mel filterbank channels."""

    fbank_stride: int

    sample_fbank_every_k: int

    # Position Encoder
    pos_encoder_type: str
    """The type of position encoder ('conv', 'relative', 'rotary')."""

    # Convolutional Position Encoder
    pos_encoder_depth: int
    """The number of stacked position encoder layers."""

    pos_conv_kernel_size: int
    """The total kernel size of 1D convolutions in position encoder layers."""

    num_pos_conv_groups: int
    """The number of convolution groups in position encoder layers."""

    # Encoder (i.e. Context Network)
    use_conformer: bool
    """If ``True``, uses Conformer blocks instead of Transformer encoder layers."""

    num_encoder_layers: int
    """The number of Transformer encoder layers."""

    num_encoder_attn_heads: int
    """The number of attention heads in Transformer encoder layers."""

    ffn_inner_dim: int
    """The inner dimensionality of Transformer feed-forward networks."""

    dropout_p: float
    """The dropout probability in Transformer layers."""

    attn_dropout_p: float
    """The dropout probability on Transformer attention weights."""

    layer_drop_p: float
    """If greater than zero, applies LayerDrop to Transformer encoder layers
    as described in :cite:t:`https://doi.org/10.48550/arxiv.1909.11556`."""

    norm_order: TransformerNormOrder
    """The Layer Normalization order."""

    depthwise_conv_kernel_size: int
    """The kernel size of depthwise convolutions in Conformer blocks."""


@dataclass
class Wav2Vec2Config:
    """Holds the configuration of a wav2vec 2.0 model."""

    encoder_config: Wav2Vec2EncoderConfig
    """The configuration of the wav2vec 2.0 encoder."""

    final_dim: int
    """The dimensionality of the final projection that is applied to context
    network outputs and quantized targets."""

    final_proj_bias: bool
    """If ``True``, the final projection learns an additive bias."""

    # Mask
    temporal_mask_span_len: int
    """The length of each temporal mask span that is applied over time steps."""

    max_temporal_mask_prob: float
    """The maximum probability of masking a time step. Note that, due to mask
    span overlap, the effective probability might be smaller."""

    spatial_mask_span_len: int
    """The length of each spatial mask span that is applied over features."""

    max_spatial_mask_prob: float
    """The maximum probability of masking a feature. Note that, due to mask span
    overlap, the effective probability might be smaller."""

    # Quantization
    quantized_dim: int
    """The output dimensionality of vector quantizer."""

    num_codebooks: int
    """The number of codebooks."""

    num_codebook_entries: int
    """The number of entries per codebook."""

    codebook_sampling_temperature: Tuple[float, float, float]
    """A tuple of start temperature, end temperature, and decay factor for
    codebook entry sampling."""

    # Loss
    num_distractors: int
    """The number of distractors to use in contrastive prediction."""

    logit_temp: float
    """The temperature to divide logits by."""

    diversity_loss_weight: float
    """The weight of diversity in loss computation."""


@final
class Wav2Vec2Frontend(TransformerFrontend):
    """Represents a Transformer encoder front-end as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    feature_dim: int
    feature_extractor: Optional[SequenceFeatureExtractor]
    post_extract_layer_norm: LayerNorm
    model_dim_proj: Optional[Linear]
    first_pass_dropout: Optional[Dropout]
    pos_encoder: Optional[PositionEncoder]
    layer_norm: Optional[LayerNorm]
    dropout: Optional[Dropout]

    def __init__(
        self,
        model_dim: int,
        feature_dim: int,
        feature_extractor: Optional[SequenceFeatureExtractor],
        pos_encoder: Optional[PositionEncoder],
        *,
        first_pass_dropout_p: float = 0.0,
        layer_norm: bool = False,
        dropout_p: float = 0.1,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param feature_dim:
            The dimensionality of extracted features.
        :param feature_extractor:
            The feature extractor. If ``None``, features are assumed to be
            extracted externally before being fed to the model.
        :param pos_encoder:
            The position encoder.
        :param first_pass_dropout_p:
            The dropout probability on extracted features before masking and
            positional encoding.
        :param layer_norm:
            If ``True``, applies Layer Normalization to extracted features
            before dropout.
        :param dropout_p:
            The dropout probability on extracted features.
        """
        super().__init__(model_dim)

        self.feature_dim = feature_dim

        if feature_extractor is not None:
            if feature_dim != feature_extractor.feature_dim:
                raise ValueError(
                    f"`feature_dim` of `feature_extractor` must be equal to `feature_dim` ({feature_dim}), but is {feature_extractor.feature_dim} instead."
                )

            self.feature_extractor = feature_extractor
        else:
            self.register_module("feature_extractor", None)

        self.post_extract_layer_norm = StandardLayerNorm(
            feature_dim, bias=True, device=device, dtype=dtype
        )

        if feature_dim != model_dim:
            self.model_dim_proj = Linear(
                feature_dim, model_dim, bias=True, device=device, dtype=dtype
            )
        else:
            self.register_module("model_dim_proj", None)

        if first_pass_dropout_p > 0.0:
            self.first_pass_dropout = Dropout(first_pass_dropout_p)
        else:
            self.register_module("first_pass_dropout", None)

        if pos_encoder is not None:
            if pos_encoder.encoding_dim != model_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` must be equal to `model_dim` ({model_dim}), but is {pos_encoder.encoding_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if layer_norm:
            self.layer_norm = StandardLayerNorm(
                model_dim, bias=True, device=device, dtype=dtype
            )
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if state_bag is not None:
            raise ValueError(
                "`Wav2Vec2Frontend` does not support incremental decoding."
            )

        seqs, padding_mask = self.extract_features(seqs, padding_mask)

        seqs, padding_mask, _ = self.process_features(seqs, padding_mask)

        return seqs, padding_mask

    def extract_features(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """Extract features from the specified sequences.

        :param seqs:
            The sequences from which to extract features. *Shape:*
            :math:`(N,S,*)`, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            - The extracted features. *Shape:* :math:`(N,S_{out},F)`, where
              :math:`N` is the batch size, :math:`S_{out}` is the output
              sequence length, and :math:`F` is the dimensionality of the
              extracted features.
            - The padding mask of the extracted features. *Shape:*
              :math:`(N,S_{out})`, where :math:`N` is the batch size and
              :math:`S_{out}` is the output sequence length.
        """
        if self.feature_extractor is not None:
            seqs, padding_mask = self.feature_extractor(seqs, padding_mask)

        seqs = self.post_extract_layer_norm(seqs)

        return seqs, padding_mask

    def process_features(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        masker: Optional[Wav2Vec2Masker] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask], Tensor]:
        """Process extracted features.

        :param seqs:
            The features to process. *Shape:* :math:`(N,S,F)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`F`
            is the dimensionality of the features.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param masker:
            If not ``None``, the features will be masked and the applied
            temporal mask will be returned as the third tuple element.

        :returns:
            - The processed sequences to pass to a Transformer encoder. *Shape:*
              :math:`(N,S,M)`, where :math:`N` is the batch size, :math:`S` is
              the sequence length, and :math:`M` is the dimensionality of the
              model.
            - The padding mask of the processed sequences. *Shape:* :math:`(N,S)`,
              where :math:`N` is the batch size and :math:`S` is the output
              sequence length.
            - The temporal mask that has been applied to the processed sequences.
              *Shape:* :math:`(N,S)`, where :math:`N` is the batch size and
              :math`S` is the sequence length.
        """
        if self.model_dim_proj is not None:
            seqs = self.model_dim_proj(seqs)

        if self.first_pass_dropout is not None:
            seqs = self.first_pass_dropout(seqs)

        if masker is not None:
            seqs, temporal_mask = masker(seqs, padding_mask)
        else:
            temporal_mask = None

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, padding_mask)

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs, padding_mask, temporal_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, feature_dim={self.feature_dim}"


class VectorQuantizer(Module, ABC):
    """Quantizes incoming data in a differentiable way."""

    input_dim: int
    output_dim: int
    num_codebooks: int
    num_codebook_entries: int

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of quantized outputs.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor) -> "VectorQuantizerOutput":
        pass


@dataclass
class VectorQuantizerOutput(ABC):
    """Holds the output of a vector quantizer."""

    quantized_vectors: Tensor
    """The quantized vector output."""

    @abstractmethod
    def compute_loss(self) -> Tensor:
        """Compute the loss."""

    @abstractmethod
    def get_target_indices(self, num_codebooks: int) -> Tensor:
        pass


@final
@dataclass
class GumbelVectorQuantizerOutput(VectorQuantizerOutput):
    cb: Tensor
    num_codebooks: int
    num_codebook_entries: int
    code_perplexity: Tensor
    prob_perplexity: Tensor
    temperature: float

    @finaloverride
    def compute_loss(self) -> Tensor:
        num_entries = self.num_codebooks * self.num_codebook_entries

        return (num_entries - self.prob_perplexity) / num_entries  # type: ignore[no-any-return]

    @finaloverride
    def get_target_indices(self, num_codebooks: int) -> Tensor:
        batch_size, seq_len = self.quantized_vectors.shape[:2]

        cb = self.cb.view(batch_size * seq_len * self.num_codebooks, -1)

        indices = cb.argmax(dim=-1).view(-1, self.num_codebooks)

        indices = indices[..., :num_codebooks]

        return indices.detach()


def init_entry_projection(proj: Linear) -> None:
    nn.init.normal_(proj.weight, mean=0.0, std=1.0)

    assert proj.bias is not None

    nn.init.zeros_(proj.bias)


@final
class GumbelVectorQuantizer(VectorQuantizer):
    """Quantizes incoming data using Gumbel-Softmax."""

    input_dim: int
    output_dim: int
    num_codebooks: int
    num_codebook_entries: int
    min_temp: float
    max_temp: float
    temp_decay: float
    entry_proj: Linear
    entries: Parameter
    num_updates: Tensor

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_codebooks: int,
        num_codebook_entries: int,
        *,
        codebook_sampling_temperature: Tuple[float, float, float],
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        """
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of quantized outputs.
        :param num_codebooks:
            number of groups for vector quantization
        :param num_codebook_entries:
            number of quantized vectors per group
        :param codebook_sampling_temperature:
            The temperature for training. A tuple of maximum temperature,
            minimum temperature, and decay factor.
        """
        super().__init__(input_dim, output_dim)

        if output_dim % num_codebooks != 0:
            raise ValueError(
                f"`output_dim` must be a multiple of `num_codebooks` ({num_codebooks}), but is {output_dim} instead."
            )

        entry_dim = output_dim // num_codebooks

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_codebooks = num_codebooks
        self.num_codebook_entries = num_codebook_entries
        self.max_temp, self.min_temp, self.temp_decay = codebook_sampling_temperature

        num_total_entries = num_codebooks * num_codebook_entries

        self.entry_proj = Linear(
            self.input_dim,
            num_total_entries,
            bias=True,
            init_fn=init_entry_projection,
            device=device,
            dtype=dtype,
        )

        self.entries = Parameter(
            torch.empty((1, num_total_entries, entry_dim), device=device, dtype=dtype)
        )

        num_updates = torch.empty((), device=device, dtype=torch.int64)

        self.register_buffer("num_updates", num_updates)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.uniform_(self.entries)

        self.num_updates.zero_()

    @finaloverride
    def forward(self, x: Tensor) -> "GumbelVectorQuantizerOutput":
        current_temp = self._compute_current_temp()

        bsz, tsz, fsz = x.shape

        x = self.entry_proj(x)

        #        x = x.unflatten(-1, (self.num_codebooks, self.num_codebook_entries))
        #
        #        k = x.argmax(-1, keepdim=True)
        #
        #        hard_x = torch.zeros_like(x, dtype=torch.float32).scatter_(-1, k, 1.0)
        #
        #        hard_probs = hard_x.mean(dim=0)
        x = x.view(bsz * tsz * self.num_codebooks, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.num_codebooks, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)

        code_perplexity = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.num_codebooks, -1).float(), dim=-1
        ).mean(dim=0)

        prob_perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        if self.training:
            x = gumbel_softmax(x.float(), tau=current_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        cb = x

        x = x.unsqueeze(-1) * self.entries
        x = x.view(bsz * tsz, self.num_codebooks, self.num_codebook_entries, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        return GumbelVectorQuantizerOutput(
            x,
            cb,
            self.num_codebooks,
            self.num_codebook_entries,
            code_perplexity,
            prob_perplexity,
            current_temp,
        )

    def _compute_current_temp(self) -> float:
        temp = self.max_temp * self.temp_decay ** int(self.num_updates)

        if self.training:
            self.num_updates.add_(1)

        return max(temp, self.min_temp)


@dataclass
class Wav2Vec2Loss:
    """Holds the loss of a wav2vec 2.0 model."""

    total: Tensor
    """The weighted total loss."""

    contrastive: Tensor
    """The contrastive loss."""

    diversity: Tensor
    """The diversity loss."""

    def backward(self) -> None:
        """Compute the gradient of the loss."""
        self.total.backward()


@dataclass
class Wav2Vec2Output:
    """Holds the output of a wav2vec 2.0 model."""

    logits: Tensor
    """The logits for contrastive feature prediction. *Shape:*
    :math:`(N,S_{msk},L)`, where :math:`N` is the batch size, :math:`S_{msk}`
    is the masked sequence length, and :math:`L` is the number of candidates
    (i.e. the number of distractors plus 1 for the target)."""

    quantized_targets: Tensor
    """The quantized context network targets that have been extracted from the
    input sequences. *Shape:* :math:`(N,S_{msk},M)`, where :math:`N` is the
    batch size, :math:`S_{msk}` is the masked sequence length, and :math:`M` is
    the dimensionality of the model."""

    temporal_mask: Tensor
    """The temporal mask that has been applied to extract the context network
    targets. *Shape:* :math:`(N,S_{enc})`, where :math:`N` is the batch size and
    :math`S_{enc}` is the encoder output sequence length."""

    quantizer_output: VectorQuantizerOutput
    """The output of the vector quantizer."""

    diversity_loss_weight: float
    """The weight of diversity in loss computation."""

    def compute_loss(self) -> Wav2Vec2Loss:
        """Compute the loss."""
        contrastive_loss = self.compute_contrastive_loss()

        diversity_loss = self.compute_diversity_loss()

        loss = contrastive_loss + self.diversity_loss_weight * diversity_loss

        return Wav2Vec2Loss(loss, contrastive_loss, diversity_loss)

    def compute_contrastive_loss(self) -> Tensor:
        """Compute the contrastive loss."""
        batch_size, seq_len, num_logits = self.logits.shape

        # (N, S, L) -> (S x N, L)
        logits = self.logits.transpose(0, 1).reshape(-1, num_logits)

        # The target is always at index 0 in the candidate list.
        target_indices = logits.new_zeros((batch_size * seq_len,), dtype=torch.int64)

        return cross_entropy(logits, target_indices, reduction="sum")

    def compute_diversity_loss(self) -> Tensor:
        """Compute the diversity loss."""
        batch_size, seq_len = self.logits.shape[:2]

        return self.quantizer_output.compute_loss() * batch_size * seq_len


@dataclass
class Wav2Vec2Loss:
    """Holds the loss of a wav2vec 2.0 model."""

    total: Tensor
    """The weighted total loss."""

    contrastive: Tensor
    """The contrastive loss."""

    diversity: Tensor
    """The diversity loss."""

    def backward(self) -> None:
        """Compute the gradient of the loss."""
        self.total.backward()


class Wav2Vec2Model(Module):
    """Represents a wav2vec 2.0 model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    model_dim: int
    encoder_frontend: Wav2Vec2Frontend
    encoder: TransformerEncoder
    masker: Wav2Vec2Masker
    quantizer: VectorQuantizer
    final_proj: Linear
    final_target_proj: Linear
    num_distractors: int
    logit_temp: float
    diversity_loss_weight: float

    def __init__(
        self,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        masker: Wav2Vec2Masker,
        quantizer: VectorQuantizer,
        final_dim: int,
        *,
        final_proj_bias: bool = True,
        num_distractors: int = 100,
        logit_temp: float = 0.1,
        diversity_loss_weight: float = 0.1,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder (i.e. context network).
        :param masker:
            The temporal/spatial feature masker.
        :param quantizer:
            The quantizer to discretize context network targets.
        :param final_dim:
            The dimensionality of the final projection that is applied to
            context network outputs and quantized targets.
        :param final_proj_bias:
            If ``True``, the final projection learns an additive bias.
        :param num_distractors:
            The number of distractors to use in contrastive prediction.
        :param logit_temp:
            The temperature to divide logits by.
        :param diversity_loss_weight:
            The weight of diversity in loss computation.
        """
        super().__init__()

        model_dim = encoder.model_dim

        self.model_dim = model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.masker = masker

        if quantizer.input_dim != encoder_frontend.feature_dim:
            raise ValueError(
                f"`input_dim` of `quantizer` and `feature_dim` of `encoder_frontend` must be equal, but are {quantizer.input_dim} and {encoder_frontend.feature_dim} instead."
            )

        self.quantizer = quantizer

        self.final_proj = Linear(
            model_dim, final_dim, final_proj_bias, device=device, dtype=dtype
        )

        self.final_target_proj = Linear(
            self.quantizer.output_dim,
            final_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

        self.num_distractors = num_distractors
        self.logit_temp = logit_temp
        self.diversity_loss_weight = diversity_loss_weight

    def forward(self, batch: SequenceBatch) -> Wav2Vec2Output:
        """
        :param batch:
            The batch of sequences to process.
        """
        seqs, padding_mask, targets, temporal_mask = self.run_frontend(
            batch.seqs, batch.padding_mask
        )

        # TODO: Should pad for fp16?
        encoder_output, _ = self.encoder(seqs, padding_mask)

        return self.quantize_and_contrast(encoder_output, targets, temporal_mask)

    def run_frontend(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask], Tensor, Tensor]:
        """Run the encoder frontend in pretraining mode.

        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            - The processed sequences to pass to the Transformer encoder.
              *Shape:* :math:`(N,S_{out},M)`, where :math:`N` is the batch size,
              :math:`S_{out}` is the output sequence length, and :math:`M` is
              the dimensionality of the model.
            - The padding mask of the processed sequences. *Shape:*
              :math:`(N,S_{out})`, where :math:`N` is the batch size and
              :math:`S_{out}` is the output sequence length.
            - The non-quantized context network targets that have been extracted
              from the input sequences. *Shape:* :math:`(N,S_{msk},M)`, where
              :math:`N` is the batch size, :math:`S_{msk}` is the masked
              sequence length, and :math:`M` is the dimensionality of the model.
            - The temporal mask that has been applied to extract the context
              network targets. *Shape:* :math:`(N,S_{out})`, where :math:`N` is
              the batch size and :math`S_{out}` is the output sequence length.
        """
        frontend = self.encoder_frontend

        seqs, padding_mask = frontend.extract_features(seqs, padding_mask)

        # We use the extracted features as context network targets after masking
        # and quantization.
        targets = seqs.clone().detach()

        if frontend.first_pass_dropout is not None:
            targets = frontend.first_pass_dropout(targets)

        seqs, padding_mask, temporal_mask = frontend.process_features(
            seqs, padding_mask, self.masker
        )

        assert temporal_mask is not None

        targets = extract_masked_elements(targets, temporal_mask)

        return seqs, padding_mask, targets, temporal_mask

    def quantize_and_contrast(
        self, encoder_output: Tensor, targets: Tensor, temporal_mask: Tensor
    ) -> "Wav2Vec2Output":
        """Quantize targets and produce logits for contrastive prediction.

        :param encoder_output:
            The encoder output. *Shape:* :math:`(N,S_{enc},M)`, where :math:`N`
            is the batch size, :math:`S_{enc}` is the encoder output sequence
            length, and :math:`M` is the dimensionality of the model.
        :param targets:
            The non-quantized context network targets that have been extracted
            from the input sequences. *Shape:* :math:`(N,S_{msk},M)`, where
            :math:`N` is the batch size, :math:`S_{msk}` is the masked sequence
            length, and :math:`M` is the dimensionality of the model.
        :param temporal_mask:
            The temporal mask that has been used to extract the context network
            targets. *Shape:* :math:`(N,S_{enc})`, where :math:`N` is the batch
            size and :math`S_{enc}` is the encoder output sequence length.
        """
        seqs = extract_masked_elements(encoder_output, temporal_mask)

        seqs = self.final_proj(seqs)

        quantizer_output = self.quantizer(targets)

        targets = self.final_target_proj(quantizer_output.quantized_vectors)

        distractors = self._sample_distractors(targets)

        logits = self._compute_logits(seqs, targets, distractors)

        return Wav2Vec2Output(
            logits,
            targets,
            temporal_mask,
            quantizer_output,
            self.diversity_loss_weight,
        )

    def _sample_distractors(self, targets: Tensor) -> Tensor:
        batch_size, seq_len, model_dim = targets.shape

        device = targets.device

        # (N, S, M) -> (N x S, M)
        targets = targets.view(-1, model_dim)

        # (S)
        indices = torch.arange(seq_len, device=device)

        # (S) -> (S x L)
        indices = repeat_interleave(indices, dim=0, repeat=self.num_distractors)

        # (N, S x L)
        rand_indices = torch.randint(
            low=0,
            high=seq_len - 1,
            size=(batch_size, seq_len * self.num_distractors),
            device=device,
        )

        # (N, S x L)
        rand_indices[rand_indices >= indices] += 1

        # (N, 1)
        k = torch.arange(batch_size, device=device).unsqueeze(1) * seq_len

        # (N, S x L)
        rand_indices += k

        # (N, S x L) -> (N x S x L)
        rand_indices = rand_indices.view(-1)

        # (N x S x L, M)
        distractors = targets[rand_indices]

        # (N x S x L) -> (N, S, L, M)
        distractors = distractors.view(
            batch_size, seq_len, self.num_distractors, model_dim
        )

        return distractors

    def _compute_logits(
        self, seqs: Tensor, targets: Tensor, distractors: Tensor
    ) -> Tensor:
        # (N, S, M) -> (N, S, 1, M)
        seqs, targets = seqs.unsqueeze(2), targets.unsqueeze(2)

        # The target will be always at index 0 in the candidate list.
        # (N, S, 1, M) + (N, S, L, M) -> (N, S, L + 1, M)
        candidates = torch.cat([targets, distractors], dim=2)

        # Perform in fp32.
        # (N, S, L + 1, M) -> (N, S, L + 1)
        logits = torch.cosine_similarity(seqs.float(), candidates.float(), dim=-1)

        if self.logit_temp != 1.0:
            logits = logits / self.logit_temp

        distractor_is_target = (targets == distractors).all(-1)

        # If `True`, it means codebook utilization is low. In such case we
        # mask the corresponding logits.
        if distractor_is_target.any():
            logits[:, :, 1:][distractor_is_target] = -torch.inf

        return logits.type_as(seqs)

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"num_distractors={self.num_distractors}, "
            f"logit_temp={self.logit_temp}, "
            f"diversity_loss_weight={self.diversity_loss_weight}"
        )


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


class Wav2Vec2FbankFeatureExtractor(SequenceFeatureExtractor):
    num_fbank_channels: int
    stride: int
    sample_every_k: int

    def __init__(
        self, num_fbank_channels: int, stride: int, *, sample_every_k: int = 1
    ):
        super().__init__(feature_dim=num_fbank_channels * stride)

        self.num_fbank_channels = num_fbank_channels
        self.stride = stride
        self.sample_every_k = sample_every_k

    @finaloverride
    def forward(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """See the base :meth:`SequenceFeatureExtractor.forward`.

        :param seqs:
            The input log-mel filterbanks. *Shape:* :math:`(N,S,C)`, where
            :math:`N` is the batch size, :math:`S` is the number of frames, and
            :math:`C` is the number of channels.
        """
        batch_size, num_frames, num_channels = seqs.shape

        if padding_mask is None:
            seq_lens = None
        else:
            seq_lens = padding_mask.seq_lens

        if (r := num_frames % self.stride) != 0:
            num_frames -= r

            seqs = seqs[:, :num_frames, :]

            if seq_lens is not None:
                seq_lens = seq_lens.clone()

                seq_lens[seq_lens > num_frames] = num_frames

        seqs = seqs.view(
            batch_size, num_frames // self.stride, num_channels * self.stride
        )

        if self.sample_every_k > 1:
            indices = torch.arange(0, batch_size, device=seqs.device)

            seqs = seqs[indices % self.sample_every_k != 0]

        if seq_lens is not None:
            # Since we contracted the temporal dimension, we should re-compute
            # the sequence lengths.
            seq_lens = self._contract_seq_lens(seq_lens)

            padding_mask = PaddingMask(seq_lens, batch_seq_len=seqs.size(1))

        return seqs, padding_mask

    def _contract_seq_lens(self, num_frames: Tensor) -> Tensor:
        num_frames = num_frames // self.stride

        if self.sample_every_k > 1:
            num_frames //= self.sample_every_k + 1

        return num_frames

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"num_fbank_channels={self.num_fbank_channels}, "
            f"stride={self.stride}, "
            f"sample_every_k={self.sample_every_k}"
        )


class Float32LayerNorm(LayerNorm):
    """Applies Layer Normalization in single-precision."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        w, b = self.weight, self.bias

        fp32_x = x.float()
        fp32_w = w.float() if w is not None else None
        fp32_b = b.float() if b is not None else None

        y = layer_norm(fp32_x, self.normalized_shape, fp32_w, fp32_b, self.eps)

        return y.type_as(x)


class Float32GroupNorm(GroupNorm):
    """Applies Group Normalization in single-precision."""

    @override(check_signature=False)
    def forward(self, x: Tensor) -> Tensor:
        w, b = self.weight, self.bias

        fp32_x = x.float()
        fp32_w = w.float()
        fp32_b = b.float() if b is not None else None

        y = group_norm(fp32_x, self.num_groups, fp32_w, fp32_b, self.eps)

        return y.type_as(x)


class Wav2Vec2FeatureConv1d(Conv1d):
    """Represents the convolution used in
    :class:`Wav2Vec2FeatureExtractionLayer`."""

    @override
    def reset_parameters(self) -> None:
        if self.bias is not None:
            # Call the base since we want to initialize bias as in `Conv1d`.
            super().reset_parameters()

        nn.init.kaiming_normal_(self.weight)


class Wav2Vec2FeatureExtractionLayer(Module):
    """Represents a feature extraction layer used in
    :class:`Wav2Vec2FeatureExtractor`."""

    conv: Conv1d
    dropout: Optional[Dropout]
    group_norm: Optional[GroupNorm]
    layer_norm: Optional[LayerNorm]
    activation: GELU

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        *,
        dropout_p: float = 0.0,
        group_norm: Optional[GroupNorm] = None,
        layer_norm: Optional[LayerNorm] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__()

        self.conv = Wav2Vec2FeatureConv1d(
            input_dim,
            output_dim,
            kernel_size,
            stride=stride,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

        if group_norm is not None:
            self.group_norm = group_norm
        else:
            self.register_module("group_norm", None)

        if layer_norm is not None:
            self.layer_norm = layer_norm
        else:
            self.register_module("layer_norm", None)

        self.activation = GELU()

    def forward(self, seqs: Tensor) -> Tensor:
        # (N, C_inp, S) -> (N, C_out, S)
        seqs = self.conv(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        if self.group_norm is not None:
            seqs = self.group_norm(seqs)

        if self.layer_norm is not None:
            seqs = seqs.transpose(1, 2)

            seqs = self.layer_norm(seqs)

            seqs = seqs.transpose(1, 2)

        seqs = self.activation(seqs)

        return seqs


@final
class Wav2Vec2FeatureExtractor(SequenceFeatureExtractor):
    """Extracts features from raw audio waveforms and embeds them in a latent
    space as described in Section 2 of
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    layers: Sequential
    layer_descs: List[Tuple[int, int, int]]
    grad_scale: float

    def __init__(
        self,
        layer_descs: Sequence[Tuple[int, int, int]],
        bias: bool,
        *,
        dropout_p: float = 0.0,
        layer_norm: bool = False,
        grad_scale: float = 1.0,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param layer_descs:
            A tuple of output dimension, kernel size, and stride for each
            feature extraction layer.
        :param bias:
            If ``True``, convolutions learn an additive bias.
        :param dropout_p:
            The dropout probability on outputs of convolutions.
        :param layer_norm:
            If ``True``, applies Layer Normalization to outputs of convolutions
            after dropout.
        :param grad_scale:
            The scale factor for gradients of extracted features. Setting to a
            value less than 1.0 allows the feature extractor to learn at a lower
            rate than the rest of the model.
        """
        # The output dimensionality of the last feature extraction layer.
        feature_dim = layer_descs[-1][0]

        super().__init__(feature_dim)

        if len(layer_descs) == 0:
            raise ValueError("`layer_descs` must be non-empty.")

        self.layers = Sequential()

        # We expect the input waveforms to be one dimensional.
        input_dim = 1

        for i, layer_desc in enumerate(layer_descs):
            output_dim, kernel_size, stride = layer_desc

            # If Layer Normalization is requested, apply it in all layers.
            if layer_norm:
                layer_norm_ = Float32LayerNorm(
                    output_dim, bias=True, device=device, dtype=dtype
                )

                group_norm_ = None

            # Otherwise, apply Group Normalization in the first layer, and do
            # not apply any normalization in other layers.
            elif i == 0:
                group_norm_ = Float32GroupNorm(
                    output_dim, output_dim, device=device, dtype=dtype
                )

                layer_norm_ = None
            else:
                group_norm_ = None
                layer_norm_ = None

            layer = Wav2Vec2FeatureExtractionLayer(
                input_dim,
                output_dim,
                kernel_size,
                stride,
                bias,
                dropout_p=dropout_p,
                group_norm=group_norm_,
                layer_norm=layer_norm_,
                device=device,
                dtype=dtype,
            )

            self.layers.append(layer)

            input_dim = output_dim

        self.layer_descs = list(layer_descs)

        if grad_scale <= 0.0 or grad_scale > 1.0:
            raise ValueError(
                f"`grad_scale` must be greater than 0.0 and less than or equal to 1.0, but is {grad_scale} instead."
            )

        self.grad_scale = grad_scale

    @finaloverride
    def forward(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """See the base :meth:`SequenceFeatureExtractor.forward`.

        :param seqs:
            The input waveforms. *Shape:* :math:`(N,S)`, where :math:`N` is the
            batch size and :math:`(S)` is the sequence length.
        """
        # (N, S) -> (N, C, S)
        seqs = seqs.unsqueeze(1)

        # (N, C, S) -> (N, E, S)
        features = self.layers(seqs)

        if self.grad_scale != 1.0:
            features = scale_grad(features, self.grad_scale)

        # (N, E, S) -> (N, S, E)
        features = features.transpose(1, 2)

        # Since we contracted the temporal dimension, we should re-compute
        # the sequence lengths.
        if padding_mask is not None:
            seq_lens = self._contract_seq_lens(padding_mask.seq_lens)

            padding_mask = PaddingMask(seq_lens, batch_seq_len=features.size(1))

        return features, padding_mask

    def _contract_seq_lens(self, num_frames: Tensor) -> Tensor:
        seq_lens = num_frames.clone()

        for desc in self.layer_descs:
            kernel_size, stride = desc[1], desc[2]

            seq_lens = (((seq_lens - kernel_size) / stride) + 1.0).floor()

        return seq_lens.type_as(num_frames)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, grad_scale={self.grad_scale}"


@final
class Wav2Vec2FeatureExtractor(SequenceFeatureExtractor):
    """Extracts features from raw audio waveforms and embeds them in a latent
    space as described in Section 2 of
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    layers: Sequential
    layer_descs: List[Tuple[int, int, int]]
    grad_scale: float

    def __init__(
        self,
        layer_descs: Sequence[Tuple[int, int, int]],
        bias: bool,
        *,
        dropout_p: float = 0.0,
        layer_norm: bool = False,
        grad_scale: float = 1.0,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param layer_descs:
            A tuple of output dimension, kernel size, and stride for each
            feature extraction layer.
        :param bias:
            If ``True``, convolutions learn an additive bias.
        :param dropout_p:
            The dropout probability on outputs of convolutions.
        :param layer_norm:
            If ``True``, applies Layer Normalization to outputs of convolutions
            after dropout.
        :param grad_scale:
            The scale factor for gradients of extracted features. Setting to a
            value less than 1.0 allows the feature extractor to learn at a lower
            rate than the rest of the model.
        """
        # The output dimensionality of the last feature extraction layer.
        feature_dim = layer_descs[-1][0]

        super().__init__(feature_dim)

        if len(layer_descs) == 0:
            raise ValueError("`layer_descs` must be non-empty.")

        self.layers = Sequential()

        # We expect the input waveforms to be one dimensional.
        input_dim = 1

        for i, layer_desc in enumerate(layer_descs):
            output_dim, kernel_size, stride = layer_desc

            # If Layer Normalization is requested, apply it in all layers.
            if layer_norm:
                layer_norm_ = Float32LayerNorm(
                    output_dim, bias=True, device=device, dtype=dtype
                )

                group_norm_ = None

            # Otherwise, apply Group Normalization in the first layer, and do
            # not apply any normalization in other layers.
            elif i == 0:
                group_norm_ = Float32GroupNorm(
                    output_dim, output_dim, device=device, dtype=dtype
                )

                layer_norm_ = None
            else:
                group_norm_ = None
                layer_norm_ = None

            layer = Wav2Vec2FeatureExtractionLayer(
                input_dim,
                output_dim,
                kernel_size,
                stride,
                bias,
                dropout_p=dropout_p,
                group_norm=group_norm_,
                layer_norm=layer_norm_,
                device=device,
                dtype=dtype,
            )

            self.layers.append(layer)

            input_dim = output_dim

        self.layer_descs = list(layer_descs)

        if grad_scale <= 0.0 or grad_scale > 1.0:
            raise ValueError(
                f"`grad_scale` must be greater than 0.0 and less than or equal to 1.0, but is {grad_scale} instead."
            )

        self.grad_scale = grad_scale

    @finaloverride
    def forward(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """See the base :meth:`SequenceFeatureExtractor.forward`.

        :param seqs:
            The input waveforms. *Shape:* :math:`(N,S)`, where :math:`N` is the
            batch size and :math:`(S)` is the sequence length.
        """
        # (N, S) -> (N, C, S)
        seqs = seqs.unsqueeze(1)

        # (N, C, S) -> (N, E, S)
        features = self.layers(seqs)

        if self.grad_scale != 1.0:
            features = scale_grad(features, self.grad_scale)

        # (N, E, S) -> (N, S, E)
        features = features.transpose(1, 2)

        # Since we contracted the temporal dimension, we should re-compute
        # the sequence lengths.
        if padding_mask is not None:
            seq_lens = self._contract_seq_lens(padding_mask.seq_lens)

            padding_mask = PaddingMask(seq_lens, batch_seq_len=features.size(1))

        return features, padding_mask

    def _contract_seq_lens(self, num_frames: Tensor) -> Tensor:
        seq_lens = num_frames.clone()

        for desc in self.layer_descs:
            kernel_size, stride = desc[1], desc[2]

            seq_lens = (((seq_lens - kernel_size) / stride) + 1.0).floor()

        return seq_lens.type_as(num_frames)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, grad_scale={self.grad_scale}"


class Wav2Vec2PositionalConv1d(Conv1d):
    """Represents the convolution used in :class:`Wav2Vec2PositionEncoder`."""

    @override
    def reset_parameters(self) -> None:
        model_dim, kernel_size = self.in_channels, self.kernel_size[0]

        try:
            remove_weight_norm(self)
        except ValueError:
            # Raised during the `__init__` call since we don't have the weight
            # norm hook registered yet. Safe to ignore.
            pass

        nn.init.normal_(
            self.weight, mean=0.0, std=(4.0 / (kernel_size * model_dim)) ** 0.5
        )

        weight_norm(self, dim=2)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


@final
class Wav2Vec2PositionEncoder(PositionEncoder):
    """Encodes sequences with relative positional information as described in
    Section 2 of :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    conv: Conv1d
    remove_pad: bool
    activation: GELU

    def __init__(
        self,
        model_dim: int,
        kernel_size: int,
        num_groups: int,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param kernel_size:
            The kernel size of the 1D convolution.
        :param num_groups:
            The number of convolution groups.
        """
        super().__init__(model_dim, max_seq_len=None)

        self.conv = Wav2Vec2PositionalConv1d(
            model_dim,
            model_dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=num_groups,
            device=device,
            dtype=dtype,
        )

        self.remove_pad = kernel_size % 2 == 0

        self.activation = GELU()

    @finaloverride
    def _do_forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        """:meta private:"""
        if state_bag is not None:
            raise ValueError(
                "`Wav2Vec2PositionEncoder` does not support incremental decoding."
            )

        # We have to ensure that the padded elements are correctly set to
        # zero; otherwise, noise will leak into the feature maps.
        seqs = apply_padding_mask(seqs, padding_mask)

        # (N, S, E) -> (N, E, S)
        encodings = seqs.transpose(1, 2)

        # (N, E, S) -> (N, E, S)
        encodings = self.conv(encodings)

        if self.remove_pad:
            encodings = encodings[:, :, :-1]

        encodings = self.activation(encodings)

        # (N, E, S) -> (N, S, E)
        encodings = encodings.transpose(1, 2)

        return seqs + encodings


class Wav2Vec2PositionEncoderLayer(Module):
    """Represents a layer used in :class:`Wav2Vec2StackedPositionEncoder`."""

    conv: Conv1d
    layer_norm: LayerNorm
    activation: GELU

    def __init__(
        self,
        model_dim: int,
        kernel_size: int,
        num_groups: int,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__()

        self.conv = Conv1d(
            model_dim,
            model_dim,
            kernel_size,
            padding="same",
            groups=num_groups,
            device=device,
            dtype=dtype,
        )

        self.layer_norm = StandardLayerNorm(
            model_dim, bias=True, elementwise_affine=False, device=device, dtype=dtype
        )

        self.activation = GELU()

    def forward(self, encodings: Tensor) -> Tensor:
        # (N, E, S) -> (N, E, S)
        encodings = self.conv(encodings)

        # (N, E, S) -> (N, S, E)
        encodings = encodings.transpose(1, 2)

        encodings = self.layer_norm(encodings)

        # (N, S, E) -> (N, E, S)
        encodings = encodings.transpose(1, 2)

        encodings = self.activation(encodings)

        return encodings


@final
class Wav2Vec2StackedPositionEncoder(PositionEncoder):
    """Encodes sequences with relative positional information using a stack
    of 1D convolutions.

    This position encoder is not mentioned in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`, but exists in the
    reference implementation.
    """

    layers: Sequential

    def __init__(
        self,
        model_dim: int,
        kernel_size: int,
        num_groups: int,
        num_layers: int,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param kernel_size:
            The total kernel size of the 1D convolutions. Each convolution uses
            a kernel size of ``max(3, kernel_size // num_layers)``.
        :param num_groups:
            The number of convolution groups.
        :param num_layers:
            The number of convolution layers.
        """
        super().__init__(model_dim, max_seq_len=None)

        k = max(3, kernel_size // num_layers)

        self.layers = Sequential()

        for _ in range(num_layers):
            layer = Wav2Vec2PositionEncoderLayer(
                model_dim,
                k,
                num_groups,
                device=device,
                dtype=dtype,
            )

            self.layers.append(layer)

    @finaloverride
    def _do_forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        """:meta private:"""
        if state_bag is not None:
            raise ValueError(
                "`Wav2Vec2StackedPositionEncoder` does not support incremental decoding."
            )

        # We have to ensure that the padded elements are correctly set to
        # zero; otherwise, noise will leak into the feature maps.
        seqs = apply_padding_mask(seqs, padding_mask)

        # (N, S, E) -> (N, E, S)
        encodings = seqs.transpose(1, 2)

        # (N, E, S) -> (N, E, S)
        encodings = self.layers(encodings)

        # (N, E, S) -> (N, S, E)
        encodings = encodings.transpose(1, 2)

        return seqs + encodings


@final
class StandardTransformerEncoder(TransformerEncoder):
    """Represents a Transformer encoder as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    self_attn_mask_factory: Optional[AttentionMaskFactory]
    layer_norm: Optional[LayerNorm]
    norm_order: TransformerNormOrder

    def __init__(
        self,
        layers: Iterable[TransformerEncoderLayer],
        *,
        self_attn_mask_factory: Optional[AttentionMaskFactory] = None,
        layer_drop_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param layers:
            The encoder layers.
        :param self_attn_mask_factory:
            The self attention mask factory.
        :param layer_drop_p:
            If greater than zero, applies LayerDrop to the encoder layers as
            described in :cite:t:`https://doi.org/10.48550/arxiv.1909.11556`.
        :param norm_order:
            The Layer Normalization order.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization module.
        """
        layer_list = ModuleList(layers, drop_p=layer_drop_p)
        if not layer_list:
            raise ValueError("`layers` must be non-empty.")

        model_dim = layer_list[0].model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.self_attn_mask_factory = self_attn_mask_factory

        self.layers = layer_list

        if norm_order != TransformerNormOrder.POST:
            self.layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        self.norm_order = norm_order

    @finaloverride
    def forward(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if self._layer_output_hooks and self.layers.drop_p > 0.0:
            raise RuntimeError(
                "The layer output hooks cannot be run when LayerDrop is enabled."
            )

        num_layers = len(self.layers)

        if self.self_attn_mask_factory is None:
            self_attn_mask = None
        else:
            self_attn_mask = self.self_attn_mask_factory(
                seqs, keys=seqs, training=self.training
            )

        for layer_idx, layer in enumerate(self.layers.drop_iter()):
            seqs, padding_mask = layer(seqs, padding_mask, self_attn_mask)

            for hook in self._layer_output_hooks.values():
                if not hook(layer_idx, seqs, padding_mask, num_layers):
                    break

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        return seqs, padding_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.self_attn_mask_factory is not None:
            self_attn_mask_factory = getattr(
                self.self_attn_mask_factory, "__name__", self.self_attn_mask_factory
            )

            s = f"{s}, self_attn_mask_factory={self_attn_mask_factory}"

        return f"{s}, norm_order={self.norm_order}"


@final
class StandardTransformerEncoderLayer(TransformerEncoderLayer):
    """Represents a Transformer encoder layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.
    """

    self_attn: MultiheadAttention
    self_attn_norm: Optional[LayerNorm]
    self_attn_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    ffn: FeedForwardNetwork
    ffn_dropout: Optional[Dropout]
    residual_scale: Optional[Parameter]
    ffn_layer_norm: LayerNorm
    norm_order: TransformerNormOrder

    def __init__(
        self,
        self_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        *,
        scale_residual: bool = False,
        dropout_p: float = 0.1,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param self_attn:
            The self attention layer.
        :param ffn:
            The feed-forward network.
        :param scale_residual:
            If ``True``, scales residuals before adding them to the output of
            the feed-forward network as described in
            :cite:t:`https://doi.org/10.48550/arxiv.2110.09456`.
        :param dropout_p:
            The dropout probability on outputs of the self attention layer and
            the feed-forward network.
        :param norm_order:
            The Layer Normalization order.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization modules.
        """
        model_dim = self_attn.model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self_attn_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        if norm_order != TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        self.self_attn = self_attn

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.self_attn_norm = layer_norm_factory(
                model_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("self_attn_norm", None)

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)

        if norm_order == TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        ffn_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        if norm_order != TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.ffn = ffn

        if dropout_p > 0.0:
            self.ffn_dropout = Dropout(dropout_p)
        else:
            self.register_module("ffn_dropout", None)

        if scale_residual:
            self.residual_scale = Parameter(
                torch.empty((model_dim,), device=device, dtype=dtype)
            )
        else:
            self.register_parameter("residual_scale", None)

        if norm_order == TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.norm_order = norm_order

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.residual_scale is not None:
            nn.init.ones_(self.residual_scale)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs = self._forward_self_attn(seqs, padding_mask, self_attn_mask)

        seqs = self._forward_ffn(seqs)

        return seqs, padding_mask

    def _forward_self_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask],
    ) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            key_padding_mask=padding_mask,
            values=seqs,
            attn_mask=self_attn_mask,
        )

        if self.self_attn_norm is not None:
            seqs = self.self_attn_norm(seqs)

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        return seqs

    def _forward_ffn(self, seqs: Tensor) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.ffn_layer_norm(seqs)

        seqs = self.ffn(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)

        if self.residual_scale is not None:
            residual = self.residual_scale * residual

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.ffn_layer_norm(seqs)

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, norm_order={self.norm_order}"


class Wav2Vec2EncoderBuilder:
    """Builds modules of a wav2vec 2.0 encoder as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: Wav2Vec2EncoderConfig
    rel_pos_encoding: Optional[RelativePositionalEncoding]
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: Wav2Vec2EncoderConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        if config.use_conformer and config.norm_order != TransformerNormOrder.POST:
            raise ValueError(
                f"`config.norm_order` must be `POST` when `config.use_conformer` is `True`, but is `{config.norm_order}` instead."
            )

        self.config = config

        self.rel_pos_encoding = None

        self.device, self.dtype = device, dtype

    def build_frontend(self) -> Wav2Vec2Frontend:
        """Build a wav2vec 2.0 Transformer encoder front-end."""
        feature_extractor = self.build_feature_extractor()

        pos_encoder = self.build_position_encoder()

        return Wav2Vec2Frontend(
            self.config.model_dim,
            self.config.feature_dim,
            feature_extractor,
            pos_encoder,
            first_pass_dropout_p=self.config.first_pass_dropout_p,
            layer_norm=self.config.layer_norm_features,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_feature_extractor(self) -> Optional[SequenceFeatureExtractor]:
        """Build a feature extractor."""
        if self.config.use_fbank:
            return Wav2Vec2FbankFeatureExtractor(
                self.config.num_fbank_channels,
                self.config.fbank_stride,
                sample_every_k=self.config.sample_fbank_every_k,
            )

        return Wav2Vec2FeatureExtractor(
            self.config.feature_extractor_layer_descs,
            self.config.feature_extractor_bias,
            layer_norm=self.config.feature_extractor_layer_norm_convs,
            grad_scale=self.config.feature_grad_scale,
            device=self.device,
            dtype=self.dtype,
        )

    def build_position_encoder(self) -> Optional[PositionEncoder]:
        """Build a position encoder."""
        if self.config.pos_encoder_type != "conv":
            return None

        if self.config.pos_encoder_depth == 1:
            return Wav2Vec2PositionEncoder(
                self.config.model_dim,
                self.config.pos_conv_kernel_size,
                self.config.num_pos_conv_groups,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            return Wav2Vec2StackedPositionEncoder(
                self.config.model_dim,
                self.config.pos_conv_kernel_size,
                self.config.num_pos_conv_groups,
                self.config.pos_encoder_depth,
                device=self.device,
                dtype=self.dtype,
            )

    def build_encoder(self) -> TransformerEncoder:
        """Build a Transformer encoder."""
        num_layers = self.config.num_encoder_layers

        layers = [self.build_encoder_layer() for _ in range(num_layers)]

        return StandardTransformerEncoder(
            layers,
            layer_drop_p=self.config.layer_drop_p,
            norm_order=self.config.norm_order,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build a Transformer encoder layer."""
        if self.config.use_conformer:
            return self.build_conformer_block()

        self_attn = self.build_attention()

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.config.dropout_p,
            norm_order=self.config.norm_order,
            device=self.device,
            dtype=self.dtype,
        )

    def build_conformer_block(self) -> TransformerEncoderLayer:
        """Build a Conformer block."""
        ffn1 = self.build_ffn(use_swish=True)

        self_attn = self.build_attention()

        conv = self.build_conformer_conv()

        ffn2 = self.build_ffn(use_swish=True)

        return ConformerBlock(
            ffn1,
            self_attn,
            conv,
            ffn2,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention(self) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        if self.config.pos_encoder_type == "rotary":
            pos_encoder = RotaryEncoder(
                self.config.model_dim // self.config.num_encoder_attn_heads,
                self.config.max_seq_len,
                device=self.device,
            )
        else:
            pos_encoder = None

        sdpa = self.build_sdpa()

        return StandardMultiheadAttention(
            self.config.model_dim,
            self.config.num_encoder_attn_heads,
            pos_encoder=pos_encoder,
            sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )

    def build_sdpa(self) -> SDPA:
        sdpa = create_default_sdpa(attn_dropout_p=self.config.attn_dropout_p)

        if self.config.pos_encoder_type == "relative":
            if self.rel_pos_encoding is None:
                self.rel_pos_encoding = RelativePositionalEncoding(
                    self.config.model_dim,
                    self.config.max_seq_len,
                    device=self.device,
                    dtype=self.dtype,
                )

            sdpa = RelativePositionSDPA(
                self.config.model_dim,
                self.config.num_encoder_attn_heads,
                self.rel_pos_encoding,
                inner_sdpa=sdpa,
                device=self.device,
                dtype=self.dtype,
            )

        return sdpa

    def build_conformer_conv(self) -> ConformerConvolution:
        return ConformerConvolution(
            self.config.model_dim,
            self.config.depthwise_conv_kernel_size,
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self, use_swish: bool = False) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            bias=True,
            inner_activation=SiLU() if use_swish else GELU(),
            norm_order=self.config.norm_order,
            device=self.device,
            dtype=self.dtype,
        )


class Wav2Vec2Builder:
    """Builds modules of a wav2vec 2.0 model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: Wav2Vec2Config
    encoder_builder: Wav2Vec2EncoderBuilder
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: Wav2Vec2Config,
        encoder_builder: Wav2Vec2EncoderBuilder,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration.
        :param encoder_builder_cls:
            The wav2vec 2.0 encoder builder.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config

        self.encoder_builder = encoder_builder

        self.device, self.dtype = device, dtype

    def build_model(self) -> Wav2Vec2Model:
        """Build a model."""
        encoder_frontend = self.encoder_builder.build_frontend()

        encoder = self.encoder_builder.build_encoder()

        masker = self.build_masker()

        quantizer = self.build_quantizer()

        return Wav2Vec2Model(
            encoder_frontend,
            encoder,
            masker,
            quantizer,
            self.config.final_dim,
            final_proj_bias=self.config.final_proj_bias,
            num_distractors=self.config.num_distractors,
            logit_temp=self.config.logit_temp,
            diversity_loss_weight=self.config.diversity_loss_weight,
            device=self.device,
            dtype=self.dtype,
        )

    def build_masker(self) -> Wav2Vec2Masker:
        """Build a temporal/spatial feature masker."""
        return Wav2Vec2Masker(
            self.config.encoder_config.model_dim,
            self.config.temporal_mask_span_len,
            self.config.max_temporal_mask_prob,
            self.config.spatial_mask_span_len,
            self.config.max_spatial_mask_prob,
            device=self.device,
            dtype=self.dtype,
        )

    def build_quantizer(self) -> VectorQuantizer:
        """Build a vector quantizer."""
        return GumbelVectorQuantizer(
            self.config.encoder_config.feature_dim,
            self.config.quantized_dim,
            self.config.num_codebooks,
            self.config.num_codebook_entries,
            codebook_sampling_temperature=self.config.codebook_sampling_temperature,
            device=self.device,
            dtype=self.dtype,
        )
