import logging
from copy import deepcopy
from functools import partial
from typing import Any, Generic, Optional, Protocol, TypeVar, Mapping
import torch
from torch.nn import Module
from fairseq.models.vocoder.checkpoint import load_checkpoint
from fairseq.models.vocoder.nn_utils import (
    infer_device,
    reset_non_persistent_buffers,
    to_empty,
)
from fairseq.typing import CPU, META, DataType, Device
from fairseq.utils import update_dataclass
from fairseq.models.unit_extractor.wav2vec2_layer_output import Wav2Vec2Config, _xlsr2_1b_v2
from fairseq.models.wav2vec2 import Wav2Vec2Model, Wav2Vec2EncoderBuilder, Wav2Vec2Builder
from fairseq.models.vocoder.checkpoint import convert_fairseq_checkpoint
from fairseq.nn.transformer import TransformerNormOrder

logger = logging.getLogger("fairseq2.models")

ConfigT = TypeVar("ConfigT")
ConfigT_contra = TypeVar("ConfigT_contra", contravariant=True)

class ConfigLoader(Generic[ConfigT]):
    """Loads model configurations of type ``ConfigT``."""

    base_config: Wav2Vec2Config

    def __init__(self) -> None:
        self.base_config = _xlsr2_1b_v2()

    def __call__(self, wav2vec2_config: dict | None = None) -> ConfigT:
        """
        :param wav2vec2_config:
            The wav2vec2_config dictionary.

        :returns:
            The model configuration of ``xlsr2_1b_v2``.
        """
        
        # Load the model configuration.
        config = self.base_config

        if wav2vec2_config:
            try:
                update_dataclass(config, deepcopy(wav2vec2_config))
            except (TypeError, ValueError) as ex:
                raise RuntimeError(
                    f"The config cannot be updated."
                ) from ex

        return config


ModelT = TypeVar("ModelT", bound=Module)
ModelT_co = TypeVar("ModelT_co", bound=Module, covariant=True)


class ModelFactory(Protocol[ConfigT_contra, ModelT_co]):
    """Constructs models of type ``ModelT``."""

    def __call__(
        self,
        config: ConfigT_contra,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> ModelT_co:
        """
        :param config:
            The model configuration.
        :param device:
            The device on which to initialize the model.
        :param dtype:
            The data type of the model parameters and buffers.
        """


class CheckpointConverter(Protocol[ConfigT_contra]):
    """Converts checkpoints to fairseq2."""

    def __call__(
        self, checkpoint: Mapping[str, Any], config: ConfigT_contra
    ) -> Mapping[str, Any]:
        """
        :param checkpoint:
            The checkpoint to convert.
        :param config:
            The configuration of the model about to be constructed.

        :returns:
            A converted checkpoint that is compatible with fairseq2.
        """


class ModelLoader(Generic[ModelT, ConfigT]):
    """Loads models of type ``ModelT``."""

    config_loader: ConfigLoader[ConfigT]
    model_factory: ModelFactory[ConfigT, ModelT]
    checkpoint_converter: Optional[CheckpointConverter[ConfigT]]
    restrict_checkpoints: bool

    def __init__(
        self,
        config_loader: ConfigLoader[ConfigT],
        model_factory: ModelFactory[ConfigT, ModelT],
        checkpoint_converter: Optional[CheckpointConverter[ConfigT]] = None,
        restrict_checkpoints: bool = True,
    ) -> None:
        """
        :param config_loader:
            The configuration loader.
        :param model_factory:
            The factory to construct models.
        :param checkpoint_converter:
            The converter to which loaded checkpoints will be passed for further
            processing.
        :param restrict_checkpoints:
            If ``True``, restricts the Python unpickler to load only tensors,
            primitive types, and dictionaries.
        """
        self.config_loader = config_loader
        self.model_factory = model_factory
        self.checkpoint_converter = checkpoint_converter
        self.restrict_checkpoints = restrict_checkpoints

    def __call__(
        self,
        model_path: str,
        model_config: dict,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
        out: Optional[ModelT] = None,
    ) -> ModelT:
        """
        :param model_path:
            The model path to load.
        :param model_config:
            The model config dict to use while loading.
        :param device:
            The device on which to load the model.
        :param dtype:
            The data type of the model parameters and buffers.
        :param out:
            The output model to load.

        :returns:
            A model loaded from the checkpoint of ``model_path``.
        """

        config = self.config_loader(model_config)

        path = model_path

        if self.checkpoint_converter is None:
            checkpoint_converter = None
        else:
            checkpoint_converter = partial(self.checkpoint_converter, config=config)

        try:
            checkpoint = load_checkpoint(
                path,
                map_location=CPU,
                restrict=self.restrict_checkpoints,
                converter=checkpoint_converter,
            )
        except (IOError, KeyError, ValueError) as ex:
            raise RuntimeError(
                f"The checkpoint of cannot be loaded. See nested exception for details."
            ) from ex

        if out is not None:
            model = out
        else:
            try:
                # Try to construct the model on the meta device.
                model = self.model_factory(config, device=META, dtype=dtype)
            except NotImplementedError:
                logger.warning(
                    f"One or more operators in constructor do not support the meta device. Skipping lazy initialization."
                )

                # If we are here, it means the model has at least one operator that
                # does not support meta device. Do regular model initialization.
                model = self.model_factory(config, device=device, dtype=dtype)

        model_device = infer_device(model)

        if model_device == META:
            # Move the model to the actual device without initializing. Its
            # state will be overwritten by the checkpoint anyways.
            to_empty(model, device=device or CPU)

        # Load the model.
        try:
            state_dict = checkpoint["model"]
        except KeyError:
            raise RuntimeError(
                f"The checkpoint of does not contain a 'model' entry."
            )

        try:
            model.load_state_dict(state_dict)
        except (KeyError, ValueError) as ex:
            raise RuntimeError(
                f"The checkpoint of cannot be loaded. See nested exception for details."
            ) from ex

        if model_device == META:
            # Non-persistent buffers are not included in the checkpoint, so we
            # have to explicitly initialize them.
            reset_non_persistent_buffers(model)

        return model



def convert_wav2vec2_checkpoint(
    checkpoint: Mapping[str, Any], config: Wav2Vec2Config
) -> Mapping[str, Any]:
    """Convert a fairseq wav2vec 2.0 checkpoint to fairseq2."""
    state_dict = checkpoint["model"]

    # Check if we have a fairseq2 checkpoint.
    if "final_target_proj.weight" in state_dict:
        return checkpoint

    if config.encoder_config.norm_order == TransformerNormOrder.POST:
        # fmt: off
        state_dict["encoder_frontend.layer_norm.weight"] = state_dict["encoder.layer_norm.weight"]
        state_dict["encoder_frontend.layer_norm.bias"]   = state_dict["encoder.layer_norm.bias"]
        # fmt: on

        del state_dict["encoder.layer_norm.weight"]
        del state_dict["encoder.layer_norm.bias"]

    state_dict["quantizer.num_updates"] = torch.zeros((), device="cpu")

    key_map = {
        # fmt: off
        r"^encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.": r"encoder.layers.\1.self_attn.output_proj.",
        r"^encoder\.layers\.([0-9]+)\.fc1\.":                 r"encoder.layers.\1.ffn.inner_proj.",
        r"^encoder\.layers\.([0-9]+)\.fc2\.":                 r"encoder.layers.\1.ffn.output_proj.",
        r"^encoder\.layers\.([0-9]+)\.final_layer_norm\.":    r"encoder.layers.\1.ffn_layer_norm.",
        r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":    r"decoder.layers.\1.ffn_layer_norm.",
        r"^encoder\.embed_tokens\.":                          r"encoder_frontend.embed.",
        r"^encoder\.pos_conv\.0\.":                           r"encoder_frontend.pos_encoder.conv.",
        r"^feature_extractor\.conv_layers\.([0-9]+)\.0\.":    r"encoder_frontend.feature_extractor.layers.\1.conv.",
        r"^feature_extractor\.conv_layers\.([0-9]+)\.2\.1\.": r"encoder_frontend.feature_extractor.layers.\1.layer_norm.",
        r"^feature_extractor\.conv_layers\.0\.2\.":           r"encoder_frontend.feature_extractor.layers.0.group_norm.",
        r"^layer_norm\.":                                     r"encoder_frontend.post_extract_layer_norm.",
        r"^post_extract_proj\.":                              r"encoder_frontend.model_dim_proj.",
        r"^mask_emb":                                         r"masker.temporal_mask_embed",
        r"^quantizer\.vars":                                  r"quantizer.entries",
        r"^quantizer\.weight_proj\.":                         r"quantizer.entry_proj.",
        r"^project_q\.":                                      r"final_target_proj.",
        # fmt: on
    }

    return convert_fairseq_checkpoint(checkpoint, key_map)


def create_wav2vec2_model(
    config: Wav2Vec2Config,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> Wav2Vec2Model:
    """Create a wav2vec 2.0 model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    encoder_builder = Wav2Vec2EncoderBuilder(
        config.encoder_config, device=device, dtype=dtype
    )

    builder = Wav2Vec2Builder(config, encoder_builder, device=device, dtype=dtype)

    return builder.build_model()


load_wav2vec2_config = ConfigLoader[Wav2Vec2Config]()

load_wav2vec2_model = ModelLoader[Wav2Vec2Model, Wav2Vec2Config](
    load_wav2vec2_config,
    create_wav2vec2_model,
    convert_wav2vec2_checkpoint,
)
