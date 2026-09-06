"""
Entry point for training an OpenSTARLab event model.

Thin wrapper around the package's own API — it does not change any model code:

    from event import Event_Model
    Event_Model(model_name, config_yaml).train()

Usage:
    python run_training.py --config configs/train_NMSTPP_demo.yaml
    python run_training.py --config configs/train_NMSTPP_full.yaml --model NMSTPP
"""

import argparse
import time

import torch

from event import Event_Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the training yaml.")
    parser.add_argument(
        "--model",
        default="NMSTPP",
        choices=["FMS", "LEM", "LEM_action", "MAJ", "NMSTPP", "Seq2Event"],
        help="Which event model to train.",
    )
    args = parser.parse_args()

    print("=" * 62)
    print(f"  Model  : {args.model}")
    print(f"  Config : {args.config}")
    if torch.cuda.is_available():
        print(f"  Device : cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("  Device : cpu")
    print("=" * 62)

    start = time.time()
    model = Event_Model(args.model, args.config)
    model.train()
    elapsed = time.time() - start

    print("=" * 62)
    print(f"  Training finished in {elapsed / 60:.1f} min")
    print("=" * 62)


if __name__ == "__main__":
    main()
