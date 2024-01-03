# Proto script to use new unit extractor from outside library
import os
import argparse
import logging

import torch
from torch import Tensor

from tqdm import tqdm
from fairseq.models.unit_extractor.unit_extractor import UnitExtractor
from examples.textless_nlp.gslm.speech2unit.clustering.utils import get_audio_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw audio to units (and optionally audio) using UnitExtractor."
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Manifest file containing the root dir and file names",
    )
    parser.add_argument(
        "--kmeans_path",
        type=str,
        help="File path to the K-Means 10k model.",
        default="kmeans_10k.npy",
    )
    parser.add_argument(
        "--xlsr_path",
        type=str,
        help="Feature extraction model path to the 'xlsr2_1b_v2' model checkpoint",
        default="xlsr2_1b_v2.pt",
    )
    parser.add_argument(
        "--out_layer_number",
        type=int,
        help="Layer number of the feature extraction model to pull out features from.",
        default=35,
    )
    parser.add_argument(
        "--out_quantized_file_path",
        required=True,
        type=str,
        help="File path of quantized output.",
    )
    parser.add_argument(
        "--extension", type=str, default=".wav", help="Audio file extension"
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("Running unit_extraction on the GPU.")
    else:
        device = torch.device("cpu")
        logger.info("Running unit_extraction on the CPU.")

    root_dir, fnames, _ = get_audio_files(args.manifest_path)

    unit_extractor = UnitExtractor(args.xlsr_path, args.kmeans_path, device=device)

    os.makedirs(os.path.dirname(args.out_quantized_file_path), exist_ok=True)
    logger.info(f"Writing quantized predictions to {args.out_quantized_file_path}")

    with open(args.out_quantized_file_path, "w") as fout:
        for i in tqdm(fnames, total=len(fnames)):
            units: Tensor = unit_extractor.predict(os.path.join(root_dir, i), args.out_layer_number - 1)
            units_list = units.tolist()
            pred_str = ' '.join(str(x) for x in units_list)
            base_fname = os.path.basename(i).rstrip(args.extension)
            fout.write(f"{base_fname}|{pred_str}\n")


if __name__ == "__main__":
    main()
