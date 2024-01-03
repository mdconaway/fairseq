from typing import Any, Dict, List
from dataclasses import dataclass


@dataclass
class VocoderConfig:
    """Holds the configuration of a Vocoder model."""

    upsample_rates: List[int]
    upsample_kernel_sizes: List[int]
    upsample_initial_channel: int
    resblock_kernel_sizes: List[int]
    resblock_dilation_sizes: List[List[int]]
    model_in_dim: int
    num_embeddings: int
    embedding_dim: int
    dur_predictor_params: Dict[str, float]
    lang_embedding_dim: int
    num_langs: int
    spkr_embedding_dim: int
    num_spkrs: int
    lang_spkr_idx_map: Dict[str, Any]


def _base_vocoder() -> VocoderConfig:
    return VocoderConfig(
        upsample_rates=[5, 4, 4, 2, 2],
        upsample_kernel_sizes=[11, 8, 8, 4, 4],
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        model_in_dim=1792,
        num_embeddings=10000,
        embedding_dim=1280,
        dur_predictor_params={
            "encoder_embed_dim": 1280,
            "var_pred_hidden_dim": 1280,
            "var_pred_kernel_size": 3,
            "var_pred_dropout": 0.5,
        },
        lang_embedding_dim=256,
        num_langs=36,
        spkr_embedding_dim=256,
        num_spkrs=200,
        lang_spkr_idx_map={},
    )
