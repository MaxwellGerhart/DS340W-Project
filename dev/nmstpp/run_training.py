"""
Entry point for training an OpenSTARLab event model.

Thin wrapper around the package's own API — it does not change any model code:

    from event import Event_Model
    Event_Model(model_name, config_yaml).train()

Usage:
    python run_training.py --config configs/train_NMSTPP_demo.yaml
    python run_training.py --config configs/train_NMSTPP_full.yaml --epochs 2
"""

import argparse
import random
import time

import numpy as np
import torch

from event import Event_Model


def apply_positional_encoding_fix():
    """Repair NMSTPP's positional encoding at runtime, without editing event/.

    Two defects in event/sports/soccer/models/NMSTPP.py combine to make the
    paper's documented seq_len=40 configuration untrainable:

    1. The pe buffer is built sequence-first, shape (max_len, 1, d_model), but
       forward() slices it as self.pe[:x.size(0), :]. The encoder is built with
       batch_first=True, so x is (batch, seq, d_model) and x.size(0) is the batch
       size. The slice is therefore indexed by a sample's position in the batch
       and broadcast identically over every timestep: time carries no positional
       signal at all, and a spurious batch-slot signal is injected instead.

    2. forward() returns dropout(x + pe), but the caller computes
       X_concatenate + self.positional_encoding(X_concatenate), so the input is
       added twice -- the encoder actually sees 2*X + pe.

    Returning only the positional term, indexed by the sequence axis, fixes both:
    the caller's X + pe(X) then evaluates to the intended X + pe.
    """
    from event.sports.soccer.models import NMSTPP as M

    def forward(self, x):
        # x is (batch, seq, d_model); index the SEQUENCE axis, not the batch axis.
        # Derived from the existing (max_len, 1, d_model) buffer rather than a new
        # one, so the package's reported parameter count is unchanged.
        return self.dropout(self.pe[: x.size(1)].transpose(0, 1))

    M.PositionalEncoding.forward = forward
    print("  NOTE  : positional-encoding fix ACTIVE (event/ itself is unmodified)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the training yaml.")
    parser.add_argument(
        "--model",
        default="NMSTPP",
        choices=["FMS", "LEM", "LEM_action", "MAJ", "NMSTPP", "Seq2Event"],
        help="Which event model to train.",
    )
    parser.add_argument(
        "--fix-positional-encoding",
        action="store_true",
        help="repair the batch/sequence axis swap and double-add in NMSTPP's "
             "PositionalEncoding at runtime; event/ is left unmodified.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the yaml's num_epoch.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for weight initialisation, shuffling and dropout. The package "
             "sets none, so without this no two runs print the same numbers.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.fix_positional_encoding:
        apply_positional_encoding_fix()

    print("=" * 62)
    print(f"  Model  : {args.model}")
    print(f"  Config : {args.config}")
    print(f"  Seed   : {args.seed}")
    if torch.cuda.is_available():
        print(f"  Device : cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("  Device : cpu")
    print("=" * 62)

    start = time.time()
    model = Event_Model(args.model, args.config)
    if args.epochs is not None:
        model.config["num_epoch"] = args.epochs
    model.train()
    elapsed = time.time() - start

    print("=" * 62)
    print(f"  Training finished in {elapsed / 60:.1f} min")
    print("=" * 62)


if __name__ == "__main__":
    main()
