"""
Run the post-training stages of the OpenSTARLab event pipeline.

Like run_training.py this is a thin wrapper: every stage below is a direct call
into the authors' unmodified API in event/sports/soccer/main_class_soccer/main.py.
No model, inference or application code is reimplemented here.

    inference     -> model.inference(...)                 test-set metrics (Table 6)
    simulation    -> model.inference(..., simulation=True) per-timestep decay (Fig 5)
    applications  -> model.cal_poss_util(...)              Poss-Util / Poss-Util+ (Fig 6)
                     model.cal_HPUS(...)                   HPUS / HPUS+ (Fig 7)

The Fig 8 heat map is deliberately absent: application/heatmap.py requires
start_x_<i>_prob / start_y_<i>_prob columns, which only LEM_inference and
FMS_inference produce. NMSTPP regresses location as two scalars, so the heat map
is not reachable from this model. The paper generates Fig 8 from LEM_3.

Usage:
    python run_downstream.py --run-dir results/full_2epoch/out/train/20260904_102542 --stage inference
"""

import argparse
import glob
import io
import json
import os
import re

import numpy as np
import pandas as pd

from event import Event_Model


def wyscout_team_table(matches_json):
    """Map Wyscout team id -> name, ordered by final league position.

    The UIED conversion keeps Wyscout's numeric team ids, and
    UEID_preprocessing's team_dict only covers the 20 La Liga sides, so a
    Wyscout run carries bare ids like 1651.0 into every plot legend. The
    matches file has what is needed: `label` is "Home - Away, x - y" and
    `teamsData` gives each side, so names fall out of the pairing, and league
    position falls out of the results (3 points a win, 1 a draw, ranked on
    points then goal difference then goals scored).
    """
    with io.open(matches_json, encoding="utf-8") as f:
        matches = json.load(f)

    # The Wyscout files store non-ASCII names double-escaped, so json.load yields
    # the literal six characters "\u00e1" rather than "a-acute". Decoding twice
    # is what turns M\u00e1laga into Malaga.
    def clean(x):
        return x.encode("latin-1", "backslashreplace").decode("unicode_escape") \
            if "\\u" in x else x

    names, pts, gf, ga = {}, {}, {}, {}
    for m in matches:
        label = m["label"].split(",")[0]
        try:
            home_name, away_name = [x.strip() for x in label.split(" - ")]
        except ValueError:
            continue
        side = {v["side"]: (int(tid), int(v["score"]))
                for tid, v in m["teamsData"].items()}
        if "home" not in side or "away" not in side:
            continue
        (hid, hs), (aid, a_s) = side["home"], side["away"]
        names[hid], names[aid] = clean(home_name), clean(away_name)
        for t in (hid, aid):
            pts.setdefault(t, 0); gf.setdefault(t, 0); ga.setdefault(t, 0)
        gf[hid] += hs; ga[hid] += a_s
        gf[aid] += a_s; ga[aid] += hs
        if hs > a_s:
            pts[hid] += 3
        elif a_s > hs:
            pts[aid] += 3
        else:
            pts[hid] += 1; pts[aid] += 1

    order = sorted(names, key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t]))
    match_ids = {m["wyId"] for m in matches if "wyId" in m}
    return names, order, match_ids


def locate(run_dir):
    """Find the artifacts the authors' train() wrote into a run directory."""
    run_1 = os.path.join(run_dir, "run_1")
    models = glob.glob(os.path.join(run_1, "_model_*.pth"))
    if not models:
        raise SystemExit(f"no _model_*.pth under {run_1}")
    # train() writes exactly one checkpoint, named _model_<best_epoch>.pth.
    # Note it does NOT contain the best epoch's weights -- see issue 12 in
    # REPRODUCTION.md -- but it is the only checkpoint the package produces.
    models.sort(key=lambda p: int(re.search(r"_model_(\d+)\.pth$", p).group(1)))
    return {
        "model_path": models[-1],
        "model_config": os.path.join(run_1, "hyperparameters.json"),
        "min_max_dict": os.path.join(run_dir, "min_max_dict.json"),
    }


PRED_COLS_PREFIX = ("action_pred", "delta_T_pred", "start_x_pred", "start_y_pred",
                    "action_", "delta_T_pred_unscaled",
                    "start_x_pred_unscaled", "start_y_pred_unscaled")


def realign_predictions(inference_csv, out_csv):
    """Re-attach each prediction to the event it was actually made for.

    UEID_inference computes end_idx against the frame returned by
    UEID_preprocessing, which has already dropped every game_over/period_over row
    and called reset_index(drop=True). It then writes the prediction back with
    data_raw.loc[end_idx], where data_raw still holds those rows. The two index
    spaces diverge by one position per dropped marker, so a prediction lands on an
    event 1..N rows too early, N being the number of markers before it. Losses are
    unaffected -- they are computed from gt inside the batch loop -- but every
    consumer of inference.csv (poss_util, HPUS, heatmap) reads shifted rows.

    This rebuilds the mapping the drop implies and moves the prediction columns
    onto the correct rows. The authors' code is not modified.
    """
    df = pd.read_csv(inference_csv)
    pred_cols = [c for c in df.columns if c.startswith(PRED_COLS_PREFIX)]

    kept = df.index[~df["action"].isin(["game_over", "period_over"])].to_numpy()
    block = df[pred_cols].iloc[: len(kept)].to_numpy()      # positional, as written

    fixed = df.copy()
    fixed[pred_cols] = float("nan")
    fixed.loc[kept, pred_cols] = block

    moved = int((kept != np.arange(len(kept))).sum())
    print(f"realign: {len(kept)} kept rows, {moved} shifted by up to "
          f"{int((kept - np.arange(len(kept))).max())} events")
    fixed.to_csv(out_csv, index=False)
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="results/<name>/out/train/<timestamp>")
    ap.add_argument("--config", default="configs/train_NMSTPP_reproduction.yaml")
    ap.add_argument("--model", default="NMSTPP")
    ap.add_argument("--test-path", default="sample_data/test.csv.gz")
    ap.add_argument("--stage", default="all",
                    choices=["all", "inference", "simulation", "applications"])
    # Paper section "Event model simulation performance": greedy selection, 26 timesteps.
    ap.add_argument("--max-iter", type=int, default=26)
    ap.add_argument("--random-selection", action="store_true",
                    help="probabilistic instead of the paper's greedy selection")
    ap.add_argument("--matches-json",
                    help="Wyscout matches_<Country>.json, used to label teams by "
                         "name and rank them; without it legends show numeric ids")
    ap.add_argument("--team-ranks", default="1,5,10,15,20",
                    help="league positions to plot in the Poss-Util figures, "
                         "mirroring the paper's Fig 6 (1st/5th/10th/15th/20th)")
    ap.add_argument("--restrict-to-league", action="store_true",
                    help="keep only events from the competition given by "
                         "--matches-json, so the figures describe one league")
    ap.add_argument("--match-id", type=int,
                    help="match to plot in the HPUS figures; default is the "
                         "first match present in the data")
    ap.add_argument("--app-dir", default="applications",
                    help="subdirectory under the run's downstream/ to write the "
                         "application figures into")
    ap.add_argument("--realign", action="store_true",
                    help="correct the row-alignment defect in UEID_inference before "
                         "running the applications (see realign_predictions)")
    args = ap.parse_args()

    art = locate(args.run_dir)
    save_path = os.path.join(args.run_dir, "downstream") + os.sep
    os.makedirs(save_path, exist_ok=True)

    print(f"  checkpoint : {art['model_path']}")
    print(f"  test set   : {args.test_path}")
    print(f"  output     : {save_path}")

    model = Event_Model(args.model, args.config)

    # train_path is passed as the string "None" so UEID_inference skips re-reading the
    # 230MB training csv; the normalisation constants come from min_max_dict.json,
    # which train() already wrote from the training split.
    common = dict(
        model_path=art["model_path"],
        model_config=art["model_config"],
        train_path="None",
        valid_path=args.test_path,
        min_max_dict_path=art["min_max_dict"],
    )

    inference_csv = os.path.join(save_path, "inference.csv")

    if args.stage in ("all", "inference"):
        print("\n=== inference on the test split ===")
        inferenced, loss_df = model.inference(save_path=save_path, **common)
        print(loss_df.to_string(index=False))

    if args.stage in ("all", "simulation"):
        print(f"\n=== simulation (greedy={not args.random_selection}, max_iter={args.max_iter}) ===")
        sim_save = os.path.join(save_path, "simulation") + os.sep
        os.makedirs(sim_save, exist_ok=True)
        _, timestep_eval, es_hota = model.inference(
            save_path=sim_save, simulation=True,
            random_selection=args.random_selection, max_iter=args.max_iter, **common)
        print(timestep_eval.to_string(index=False))

    if args.stage in ("all", "applications"):
        print("\n=== applications ===")
        if not os.path.exists(inference_csv):
            raise SystemExit("run the inference stage first: it writes inference.csv")
        if args.realign:
            data = realign_predictions(
                inference_csv, os.path.join(save_path, "inference_realigned.csv"))
        else:
            data = pd.read_csv(inference_csv)

        # the application plotters concatenate save_path + filename, so it must
        # end in a separator or the files land next to the directory instead of in it
        app_save = os.path.join(save_path, args.app_dir) + os.sep
        os.makedirs(app_save, exist_ok=True)

        # Team labelling. plot_poss_util_dist uses the `team` value itself as the
        # legend entry, so substituting names into the column is what renames the
        # series; `teams` then selects which series are drawn.
        teams = None
        if args.matches_json:
            names, order, league_matches = wyscout_team_table(args.matches_json)
            if args.restrict_to_league:
                before = (len(data), data["match_id"].nunique())
                data = data[data["match_id"].isin(league_matches)]
                print(f"league filter: {before[0]:,} rows / {before[1]} matches "
                      f"-> {len(data):,} rows / {data['match_id'].nunique()} matches")
            present = {int(t) for t in data["team"].dropna().unique()}
            # Rank is the team's true final league position, not its position
            # among those sampled into this split -- the test split holds only a
            # fraction of matches, so some teams may be missing entirely.
            true_rank = {t: i + 1 for i, t in enumerate(order)}
            label = {t: f"{names[t]} ({true_rank[t]})" for t in order}
            wanted = [int(r) for r in args.team_ranks.split(",")]
            teams = [label[t] for t in order
                     if true_rank[t] in wanted and t in present]
            data["team"] = data["team"].map(
                lambda t: label.get(int(t), t) if pd.notna(t) else t)
            print(f"teams: {len(present)} in split, plotting {teams}")

        # Fig 6 - Poss-Util / Poss-Util+ distributions.
        # Without `team`, all teams in the split are drawn on one axis; on the
        # five-league test split that is 98 overlapping curves and a legend that
        # covers the plot. The paper draws five, spread across the table.
        poss_util = model.cal_poss_util(data, save_path=app_save, team=teams)
        poss_util.to_csv(os.path.join(app_save, "poss_util.csv"), index=False)
        print(f"poss_util: {len(poss_util)} possessions -> {app_save}")

        # Fig 7 - HPUS / HPUS+ over one match.
        match_id = args.match_id or int(data["match_id"].dropna().iloc[0])
        for plus in (False, True):
            hpus = model.cal_HPUS(data, save_path=app_save, match_id=match_id, plus=plus)
        hpus.to_csv(os.path.join(app_save, "hpus.csv"), index=False)
        print(f"HPUS: match {match_id}, {len(hpus)} possessions -> {app_save}")

    print("\ndone")


if __name__ == "__main__":
    main()
