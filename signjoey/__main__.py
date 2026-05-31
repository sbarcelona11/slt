import argparse
import os

import sys
from signjoey.training import train
from signjoey.prediction import test

sys.path.append("/vol/research/extol/personal/cihan/code/SignJoey")


def main():
    ap = argparse.ArgumentParser("Joey NMT")

    ap.add_argument("mode", choices=["train", "test"], help="train a model or test")

    ap.add_argument("config_path", type=str, help="path to YAML config file")

    ap.add_argument("--ckpt", type=str, help="checkpoint for prediction")

    ap.add_argument(
        "--output_path", type=str, help="path for saving translation output"
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run on. 'auto' picks cuda/mps/cpu if available.",
    )
    ap.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="CUDA GPU id for CUDA runs only (sets CUDA_VISIBLE_DEVICES).",
    )
    args = ap.parse_args()

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "train":
        train(cfg_file=args.config_path, device=args.device)
    elif args.mode == "test":
        test(
            cfg_file=args.config_path,
            ckpt=args.ckpt,
            output_path=args.output_path,
            device=args.device,
        )
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
