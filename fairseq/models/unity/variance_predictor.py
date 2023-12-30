from typing import Optional
from fairseq.nn.normalization import LayerNorm
from fairseq.nn.padding import PaddingMask, apply_padding_mask
from fairseq.nn.projection import Linear
from fairseq.nn.layer_norm import create_standard_layer_norm
from fairseq.typing import DataType, Device
from torch import Tensor
from torch.nn import Conv1d, Dropout, Module, ReLU, Sequential
from fairseq.models.unity.film import FiLM


class VariancePredictor(Module):
    """Represents the duration/pitch/energy predictor as described in
    :cite:t:`https://arxiv.org/pdf/2006.04558.pdf`"""

    conv1: Sequential
    ln1: LayerNorm
    dropout_module: Dropout
    conv2: Sequential
    ln2: LayerNorm
    proj: Linear
    film: Optional[FiLM]

    def __init__(
        self,
        encoder_embed_dim: int,
        var_pred_hidden_dim: int,
        var_pred_kernel_size: int,
        var_pred_dropout: float,
        bias: bool = True,
        use_film: bool = False,
        film_cond_dim: int = 512,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()

        self.conv1 = Sequential(
            Conv1d(
                encoder_embed_dim,
                var_pred_hidden_dim,
                var_pred_kernel_size,
                stride=1,
                padding="same",
                bias=bias,
                device=device,
                dtype=dtype,
            ),
            ReLU(),
        )

        layer_norm_factory = create_standard_layer_norm

        self.ln1 = layer_norm_factory(var_pred_hidden_dim, device=device, dtype=dtype)

        self.dropout_module = Dropout(p=var_pred_dropout)

        self.conv2 = Sequential(
            Conv1d(
                var_pred_hidden_dim,
                var_pred_hidden_dim,
                var_pred_kernel_size,
                stride=1,
                padding="same",
                bias=bias,
                device=device,
                dtype=dtype,
            ),
            ReLU(),
        )

        self.ln2 = layer_norm_factory(var_pred_hidden_dim, device=device, dtype=dtype)

        self.proj = Linear(
            var_pred_hidden_dim, 1, bias=True, device=device, dtype=dtype
        )

        if use_film:
            self.film = FiLM(
                film_cond_dim, var_pred_hidden_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("film", None)

    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask] = None,
        film_cond_emb: Optional[Tensor] = None,
    ) -> Tensor:
        # Ensure that we do not leak padded positions in the convolution layer.
        seqs = apply_padding_mask(seqs, padding_mask)

        # (N, S, M) -> (N, M, S)
        seqs = seqs.transpose(1, 2)

        # (N, M, S) -> (N, H, S)
        seqs = self.conv1(seqs)

        # (N, H, S) -> (N, S, H)
        seqs = seqs.transpose(1, 2)

        seqs = self.ln1(seqs)

        seqs = self.dropout_module(seqs)

        seqs = apply_padding_mask(seqs, padding_mask)

        # (N, S, H) -> (N, H, S)
        seqs = seqs.transpose(1, 2)

        # (N, H, S) -> (N, H, S)
        seqs = self.conv2(seqs)

        # (N, H, S) -> (N, S, H)
        seqs = seqs.transpose(1, 2)

        seqs = self.ln2(seqs)

        seqs = self.dropout_module(seqs)

        seqs = apply_padding_mask(seqs, padding_mask)

        if self.film is not None and film_cond_emb is not None:
            seqs = self.film(seqs, film_cond_emb)
            seqs = apply_padding_mask(seqs, padding_mask)

        # (N, S, H) -> (N, S, 1) -> (N, S)
        seqs = self.proj(seqs).squeeze(dim=2)

        return seqs
