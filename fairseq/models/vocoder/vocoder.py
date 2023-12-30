# Add support to run https://huggingface.co/facebook/seamless-m4t-vocoder/resolve/main/vocoder_36langs.pt
# based on units from kmeans 10k: https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy
from typing import Any, Dict, Optional, List, Union
import torch
from torch import Tensor
from torch.nn import Module
from fairseq.typing import Device, DataType
from fairseq.models.vocoder.config import VocoderConfig
from fairseq.models.vocoder.vocoder_36_config import config
from fairseq.models.vocoder.codehifigan import CodeGenerator
from fairseq.models.vocoder.loader import ModelLoader, ConfigLoader
from fairseq.models.vocoder.checkpoint import convert_vocoder_checkpoint


class Vocoder(Module):
    def __init__(
        self,
        code_generator: CodeGenerator,
        lang_spkr_idx_map: Dict[str, Any],
    ):
        super().__init__()
        self.code_generator = code_generator
        self.lang_spkr_idx_map = lang_spkr_idx_map

    def forward(
        self,
        units: Tensor,
        lang_list: Union[List[str], str],
        spkr_list: Union[Optional[List[int]], int] = None,
        dur_prediction: bool = True,
    ) -> Tensor:
        # TODO: Do we need this backward compatibility, or just update all calling sites? 
        if len(units.shape) == 1:
            units = units.unsqueeze(0) # add batch dim
        if isinstance(lang_list, str):
            lang_list = [lang_list] * units.size(0)
        if isinstance(spkr_list, int):
            spkr_list = [spkr_list] * units.size(0)
        lang_idx_list = [self.lang_spkr_idx_map["multilingual"][l] for l in lang_list]
        if not spkr_list:
            spkr_list = [-1 for _ in range(len(lang_list))]
        spkr_list = [self.lang_spkr_idx_map["multispkr"][lang_list[i]][0] if spkr_list[i] == -1 else spkr_list[i] for i in range(len(spkr_list))]
        x = {
            "code": units.view(units.size(0), -1),
            "spkr": torch.tensor([spkr_list], device=units.device).t(),
            "lang": torch.tensor([lang_idx_list], device=units.device).t(),
        }
        return self.code_generator(x, dur_prediction)  # type: ignore[no-any-return]


class VocoderBuilder:
    """Builds modules of a vocoder model (Code Hifigan) as described in
    :cite:t`https://github.com/facebookresearch/speech-resynthesis`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: VocoderConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: VocoderConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config
        self.device, self.dtype = device, dtype

    def build_model(self) -> Vocoder:
        """Build a model."""

        code_generator = CodeGenerator(
            self.config.upsample_rates,
            self.config.upsample_kernel_sizes,
            self.config.upsample_initial_channel,
            self.config.resblock_kernel_sizes,
            self.config.resblock_dilation_sizes,
            self.config.model_in_dim,
            self.config.num_embeddings,
            self.config.embedding_dim,
            self.config.dur_predictor_params,
            self.config.lang_embedding_dim,
            self.config.num_langs,
            self.config.spkr_embedding_dim,
            self.config.num_spkrs,
        )
        code_generator.to(device=self.device, dtype=self.dtype)
        vocoder = Vocoder(code_generator, self.config.lang_spkr_idx_map)
        return vocoder


def create_vocoder_model(
    config: VocoderConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> Vocoder:
    """Create a Vocoder model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """

    return VocoderBuilder(config, device=device, dtype=dtype).build_model()



load_vocoder_config = ConfigLoader[VocoderConfig]()


load_vocoder_model = ModelLoader[Vocoder, VocoderConfig](
    load_vocoder_config,
    create_vocoder_model,
    convert_vocoder_checkpoint,
)


def load_vocoder_36(path: str, **kwargs):
    vocoder_instance = load_vocoder_model(model_path=path, model_config=config, **kwargs).eval()
    assert isinstance(vocoder_instance, Vocoder)
    # can be called like: wav = vocoder_instance(units, src_lang="cmn", spkr_list=-1, dur_prediction=True)
    return vocoder_instance
