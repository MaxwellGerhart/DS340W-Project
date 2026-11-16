"""
Train NMSTPP with or without the lateral mirror symmetry (DS340W phase 2).

Three arms, identical except for how the mirror is used:

    baseline   the reproduction configuration, no symmetry
    augment    each training sample is mirrored with probability 1/2
               (learns the symmetry from data -- the usual control)
    frameavg   the network is wrapped in FrameAveraged, so it is exactly
               equivariant by construction (TacticAI's approach)

The model class, preprocessing and cost function are the authors', imported
unchanged. The training loop is a re-implementation of trainers/UEID_train.py
(same optimiser, loss, batch size, shuffling, drop_last and early-stopping
rule) with two differences:

  * samples come from symmetry.FastUEID, which is asserted identical to
    UEID_loader but serves the whole split from GPU memory;
  * the best epoch's weights are deep-copied, so the checkpoint written is the
    one early stopping selected (issue 12).

The positional-encoding fix from run_training.py is applied to every arm (see
--no-pe-fix). Outputs follow the package's layout, so run_downstream.py and
probe_equivariance.py work on them unchanged:

    <save>/<arm>_seed<k>/min_max_dict.json
    <save>/<arm>_seed<k>/run_1/_model_<best>.pth     plain NMSTPP state_dict (FA wrapper stripped)
    <save>/<arm>_seed<k>/run_1/hyperparameters.json
    <save>/<arm>_seed<k>/run_1/_loss.csv
    <save>/<arm>_seed<k>/results.json                 test metrics, equivariance gap

Usage:
    python train_symmetry.py --config configs/train_NMSTPP_reproduction.yaml --arm frameavg --seed 0
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml

from event.sports.soccer.models.NMSTPP import NMSTPP
from event.sports.soccer.utils.UEID_cost_function import cost_function
from event.sports.soccer.utils.UEID_preprocessing import UEID_preprocessing
from run_training import apply_positional_encoding_fix
from symmetry import (METRICS, FastUEID, FrameAveraged, Mirror, equivariance_gap,
                      package_metrics, random_mirror)

ARMS = ["baseline", "augment", "frameavg"]


def build_model(config, n_features, device):
    return NMSTPP(config["num_actions"], config["action_embedding_out_len"], config["scale_grad_by_freq"],
                  n_features - 1, config["continuous_embedding_output_len"],
                  config["multihead_attention"], config["hidden_dim"], config["feature_embedding_output_len"],
                  config["NN_deltaT_num_layers"], config["NN_location_num_layers"], config["NN_action_num_layers"],
                  1, 2, config["num_actions"]).to(device)


def short(r):
    return (f"acc {r['ACC_action']:.4f} | F1 {r['F1_action']:.4f} | F1g {r['F1_global']:.4f} | "
            f"dT {r['MAE_deltaT']:.3f} | x {r['MAE_x']:.3f} | y {r['MAE_y']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-path", default="sample_data/test.csv.gz")
    ap.add_argument("--save-path", default="./results/symmetry")
    ap.add_argument("--epochs", type=int, default=None, help="override num_epoch")
    ap.add_argument("--no-pe-fix", action="store_true", help="train with the published positional encoding")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.epochs is not None:
        config["num_epoch"] = args.epochs
    config["eps"] = float(config["eps"])
    features = config["basic_features"] + (config["other_features"] if config["use_other_features"] else [])
    seq_len, bs = config["seq_len"], config["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.no_pe_fix:
        apply_positional_encoding_fix()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    t0 = time.time()
    train_df, min_dict, max_dict = UEID_preprocessing(pd.read_csv(config["train_path"]))
    valid_df, _, _ = UEID_preprocessing(pd.read_csv(config["valid_path"]), min_dict, max_dict)
    test_df, _, _ = UEID_preprocessing(pd.read_csv(args.test_path), min_dict, max_dict)
    train = FastUEID(train_df, seq_len, features, device)
    valid = FastUEID(valid_df, seq_len, features, device)
    test = FastUEID(test_df, seq_len, features, device)
    print(f"  data ready in {time.time() - t0:.0f}s: train {len(train):,} | valid {len(valid):,} | test {len(test):,}")

    mirror = Mirror(features, min_dict, max_dict, config["num_actions"])
    base = build_model(config, len(features), device)
    model = FrameAveraged(base, mirror) if args.arm == "frameavg" else base
    optimizer = torch.optim.Adam(base.parameters(), lr=config["learning_rate"], eps=config["eps"])
    n_params = sum(p.numel() for p in base.parameters())
    print(f"  arm {args.arm} | seed {args.seed} | params {n_params:,} | pe-fix {not args.no_pe_fix}")

    history, best, best_loss, best_epoch, stale = [], None, float("inf"), 0, 0
    for epoch in range(config["num_epoch"]):
        t = time.time()
        model.train()
        tot, comps, nb = 0.0, np.zeros(len(METRICS) - 1), 0
        for X, gt, _ in train.batches(bs, shuffle=True, drop_last=True, generator=gen):
            if args.arm == "augment":
                X, gt = random_mirror(X, gt, mirror)
            optimizer.zero_grad()
            loss, c = cost_function(gt, model(X), min_dict=min_dict, max_dict=max_dict,
                                    device=device, config=config)
            loss.backward()
            optimizer.step()
            tot += loss.item()
            comps += np.array([x.item() for x in c])
            nb += 1
        tr = dict(zip(METRICS, [tot / nb] + list(comps / nb)))
        va = package_metrics(model, valid, min_dict, max_dict, config, batch_size=bs)
        history.append({"epoch": epoch + 1, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"valid_{k}": va[k] for k in METRICS + ["ACC_global", "F1_global"]}})
        improved = va["loss"] < best_loss
        if improved:
            best_loss, best_epoch, stale = va["loss"], epoch + 1, 0
            best = copy.deepcopy(base.state_dict())
        else:
            stale += 1
        print(f"  epoch {epoch + 1:2d} {time.time() - t:5.0f}s | train {tr['loss']:.4f} | "
              f"valid {va['loss']:.4f}{' *' if improved else '  '} | {short(va)}", flush=True)
        if stale >= config["early_stop_patience"]:
            print(f"  early stop: no improvement for {stale} epochs")
            break

    base.load_state_dict(best)
    res = {"arm": args.arm, "seed": args.seed, "best_epoch": best_epoch, "epochs_run": len(history),
           "num_params": n_params, "pe_fix": not args.no_pe_fix, "config": args.config,
           "test": package_metrics(model, test, min_dict, max_dict, config, batch_size=bs),
           "test_mirrored": package_metrics(model, test, min_dict, max_dict, config, batch_size=bs,
                                            mirror_data=mirror),
           "equivariance_gap": equivariance_gap(model, test, mirror, min_dict, max_dict)}
    if args.arm != "frameavg":
        res["test_frameavg_at_inference"] = package_metrics(FrameAveraged(base, mirror), test, min_dict,
                                                            max_dict, config, batch_size=bs)
    print(f"  best epoch {best_epoch} | test: {short(res['test'])}")
    if "test_frameavg_at_inference" in res:
        print(f"  test, frame-averaged at inference: {short(res['test_frameavg_at_inference'])}")
    print("  equivariance gap: " + " | ".join(f"{k} {v:.4g}" for k, v in res["equivariance_gap"].items()))

    # Package-compatible layout, so run_downstream.py can consume the checkpoint.
    run_dir = os.path.join(args.save_path, f"{args.arm}_seed{args.seed}")
    run_1 = os.path.join(run_dir, "run_1")
    os.makedirs(run_1, exist_ok=True)
    # Saved as CPU tensors: the package loads checkpoints without map_location,
    # so one saved from a GPU cannot be opened on a machine that has none.
    torch.save({k: v.cpu() for k, v in best.items()}, os.path.join(run_1, f"_model_{best_epoch}.pth"))
    pd.DataFrame(history).round(4).to_csv(os.path.join(run_1, "_loss.csv"), index=False)
    with open(os.path.join(run_dir, "min_max_dict.json"), "w") as f:
        json.dump({"min_dict": min_dict, "max_dict": max_dict}, f, indent=4)
    hp = {**config, "features": features, "model": "NMSTPP", "best_epoch": best_epoch,
          "num_continous_features": len(features) - 1, "action_embedding_input_len": config["num_actions"],
          "continuous_embedding_input_len": len(features) - 1, "deltaT_output_len": 1,
          "location_output_len": 2, "action_output_len": config["num_actions"],
          "symmetry_arm": args.arm, "seed": args.seed, "pe_fix": not args.no_pe_fix}
    with open(os.path.join(run_1, "hyperparameters.json"), "w") as f:
        json.dump(hp, f, indent=4)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"  written: {run_dir}  ({(time.time() - t0) / 60:.1f} min total)")


if __name__ == "__main__":
    main()
