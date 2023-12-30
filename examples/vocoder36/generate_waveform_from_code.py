import os
import argparse
import logging
from pathlib import Path
import torchaudio
from torch import Tensor, LongTensor, cuda, float32
from tqdm import tqdm
from fairseq import utils
from fairseq.typing import CPU, GPU
from fairseq.models.vocoder.vocoder import load_vocoder_36


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dump_result(args, sample_id, pred_wav: Tensor, suffix=""):
    torchaudio.save(
        f"{args.results_path}/{sample_id}{suffix}_pred.wav", 
        pred_wav[0].to(float32).cpu(), 
        16000,
    )


def load_code(in_file):
    with open(in_file) as f:
        out = [list(map(int, line.strip().split())) for line in f]
    return out


def main(args):
    logger.info(args)

    use_cuda = cuda.is_available() and not args.cpu

    if not os.path.exists(args.vocoder):
        raise RuntimeError("Invalid vocoder path!")
    device = GPU if use_cuda else CPU
    vocoder = load_vocoder_36(path=args.vocoder, device=device, dtype=float32)
    data = load_code(args.in_code_file)
    Path(args.results_path).mkdir(exist_ok=True, parents=True)
    for i, d in tqdm(enumerate(data), total=len(data)):
        x = LongTensor(d).view(1, -1)
        suffix = ""
        x = utils.move_to_cuda(x) if use_cuda else x
        wav: Tensor = vocoder(x, lang_list=args.language, spkr_list=args.speaker_id, dur_prediction=args.dur_prediction)
        dump_result(args, i, wav, suffix=suffix)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-code-file", type=str, required=True, help="one unit sequence per line"
    )
    parser.add_argument(
        "--vocoder", type=str, required=True, help="path to the vocoder36 checkpoint"
    )
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument(
        "--dur-prediction",
        action="store_true",
        help="enable duration prediction (for reduced/unique code sequences)",
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=-1,
        help="Speaker id (for vocoder that supports multispeaker). Set to -1 to randomly sample speakers.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="cmn",
        help="Language code to pass to vocoder (eg 'cmn' or 'eng')",
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()
