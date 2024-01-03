import logging
from typing import Union
import torch
from torch.nn.functional import layer_norm
import torchaudio
from torchaudio.transforms import Resample
from fairseq.models.sequence import SequenceBatch
from fairseq.models.unit_extractor.wav2vec2_loaders import Wav2Vec2Model, load_wav2vec2_model
from fairseq.nn.padding import get_seqs_and_padding_mask
from fairseq.typing import DataType, Device, CPU
from torch import Tensor, nn

from fairseq.models.unit_extractor.kmeans import KmeansModel
from fairseq.models.unit_extractor.wav2vec2_layer_output import (
    Wav2Vec2LayerOutputModel,
)
from fairseq.models.vocoder.vocoder import load_vocoder_36, Vocoder
from fairseq.utils import move_to_cuda

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class UnitExtractor(nn.Module):
    """Unit Extractor which converts raw audio into units."""

    def __init__(
        self,
        xlsr_path: str,
        kmeans_path: str,
        device: Device,
        dtype: DataType = torch.float32,
    ):
        super().__init__()

        wav2vec2_model = load_wav2vec2_model(
            model_path=xlsr_path, model_config=None, device=device, dtype=dtype
        )
        wav2vec2_model.eval()
        assert isinstance(wav2vec2_model, Wav2Vec2Model)
        self.model = Wav2Vec2LayerOutputModel(wav2vec2_model)
        self.kmeans_model = KmeansModel(kmeans_path, device, dtype)
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def predict(
        self,
        audio: Union[str, Tensor],
        out_layer_idx: int,
        input_sample_rate: int = 16000,
        xlsr_sample_rate: int = 16000,
    ) -> Tensor:
        if isinstance(audio, str):
            waveform, original_sample_rate = torchaudio.load(
                audio, normalize=True
            )
            transform = Resample(original_sample_rate, xlsr_sample_rate)
            src: Tensor = transform(waveform)
        else:
            transform = Resample(input_sample_rate, xlsr_sample_rate)
            src: Tensor = transform(waveform)
        if src.dim() == 1:
            src = src.unsqueeze(1)
        elif src.dim() == 2 and src.size(0) < src.size(1):
            src = src.transpose(0, 1)
        src = src.to(dtype=self.dtype)
        if self.device != CPU:
            src = move_to_cuda(src, device=self.device)
        seqs, padding_mask = get_seqs_and_padding_mask({
            "is_ragged": False,
            "seqs": src
        })
        seqs = seqs.view(1, -1)
        seqs = layer_norm(seqs, seqs.shape)
        batch = SequenceBatch(seqs=seqs, padding_mask=padding_mask)
        features = self.model(batch, out_layer_idx).squeeze(0)
        units = self.kmeans_model(features)
        return units  # type: ignore[no-any-return]

    @staticmethod
    def resynthesize_audio(
        units: Tensor,
        src_lang: str,
        device: Device,
        dtype: DataType,
        vocoder_path: str = "vocoder_36langs.pt",
    ) -> Tensor:
        vocoder = load_vocoder_36(path=vocoder_path, device=device, dtype=dtype)
        assert isinstance(vocoder, Vocoder)
        wav = vocoder(units,  lang_list=src_lang, spkr_list=-1, dur_prediction=True)
        return wav  # type: ignore[no-any-return]
