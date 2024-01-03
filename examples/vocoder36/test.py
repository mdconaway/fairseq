# Download the vocoder here: https://huggingface.co/facebook/seamless-m4t-vocoder/resolve/main/vocoder_36langs.pt
# It is based on units from kmeans 10k: https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy
# With units extracted from layer 34 of this model: https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/xlsr2_1b_v2.pt
import os
import torch
from torch import randint, Tensor
import argparse
from fairseq.typing import CPU, GPU
from fairseq.models.vocoder.vocoder import load_vocoder_36
from fairseq import utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocoder-path",
        default="./vocoder_36langs.pt",
        type=str,
        metavar="LOC",
        help="Path to vocoder checkpoint file",
    )
    parser.add_argument(
        "--language",
        default="cmn",
        type=str,
        help="Language code to test (eg 'cmn' or 'eng')",
    )
    return parser


def main(args):
    if not os.path.exists(args.vocoder_path):
        raise RuntimeError("Invalid vocoder path!")
    random_tensor = randint(1, 9999, (200, 1))
    use_cuda = torch.cuda.is_available()
    device = GPU if use_cuda else CPU
    if use_cuda:
        random_tensor = utils.move_to_cuda(random_tensor)
    vocoder = load_vocoder_36(
        path=args.vocoder_path, device=device, dtype=torch.float32
    )
    wav: Tensor = vocoder(
        random_tensor, lang_list=args.language, spkr_list=-1, dur_prediction=True
    )
    print(wav.detach().cpu().numpy())


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
