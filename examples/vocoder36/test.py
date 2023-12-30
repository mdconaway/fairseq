# Download the vocoder here: https://huggingface.co/facebook/seamless-m4t-vocoder/resolve/main/vocoder_36langs.pt
# It is based on units from kmeans 10k: https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy
# With units extracted from layer 34 of this model: https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/xlsr2_1b_v2.pt
import os
import torch
from torch import randint
import argparse
from fairseq.typing import CPU
from fairseq.models.vocoder.vocoder import load_vocoder_36


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocoder-path", default="./vocoder_36langs.pt", type=str, metavar="LOC", help="Path to vocoder checkpoint file"
    )
    parser.add_argument(
        "--language", default="cmn", type=str, metavar="LANG", help="Language code to test"
    )
    return parser


def main(args):
    if not os.path.exists(args.vocoder_path):
        raise RuntimeError("Invalid vocoder path!")
    
    vocoder = load_vocoder_36(path=args.vocoder_path, device=CPU, dtype=torch.float32)
    wav = vocoder(randint(1,9999, (200,1)), lang_list=args.language, spkr_list=-1, dur_prediction=True)
    print(wav)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
