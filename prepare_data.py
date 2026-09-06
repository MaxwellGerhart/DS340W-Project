"""
Windows-portable port of the OpenSTARLab PreProcessing example script.

Source: https://github.com/open-starlab/PreProcessing/blob/master/example/NMSTPP_data.py

The original shells out to `wget`, `unzip` and `rm -rf`, none of which exist on
Windows. The download/extract/cleanup steps below use the stdlib instead
(requests + zipfile + os.remove). The preprocessing and train/valid/test split
logic is unchanged from the original.

Downloads the public Wyscout dataset (Pappalardo et al. 2019, CC BY 4.0) and
writes train.csv / valid.csv / test.csv in NMSTPP/UEID format, which is what
event/sports/soccer/models/model_yaml/train_NMSTPP.yaml expects.

Usage:
    python prepare_data.py --data-dir C:/Users/maxwe/ds340w-data
"""

import argparse
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from preprocessing import Event_data

LEAGUES = ["England", "France", "Germany", "Italy", "Spain"]

# The original script points at https://figshare.com/ndownloader/files/<id>/<name>,
# which now returns 403. The ndownloader.figshare.com host still serves these
# files (it redirects to a signed S3 URL), so use that instead. Same file ids.
EVENT_URL = "https://ndownloader.figshare.com/files/14464685"
MATCHES_URL = "https://ndownloader.figshare.com/files/14464622"

EVENT_ZIP = "events.zip"
MATCHES_ZIP = "matches.zip"

# The original keeps only England/France/Italy/Spain/Germany.
DISCARD = [
    "events_European_Championship.json",
    "events_World_Cup.json",
    "matches_European_Championship.json",
    "matches_World_Cup.json",
]

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1


def download(url, output_directory, name):
    """Stream a URL to disk. Replaces the original's wget subprocess call."""
    os.makedirs(output_directory, exist_ok=True)
    filename = os.path.join(output_directory, name)

    # Require a non-empty file: a truncated or 0-byte leftover must not be
    # mistaken for a completed download.
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Already downloaded, skipping: {filename}")
        return filename

    # Do NOT send a browser User-Agent here. figshare answers browser-like
    # clients with an empty "202 Accepted" instead of the "302 Found" redirect
    # to signed S3 storage, which silently yields a 0-byte file. The original
    # script's spoofed Chrome UA triggers exactly that. A plain client works.
    print(f"Downloading {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    if r.status_code == 202 or r.headers.get("content-type", "").startswith("text/html"):
        raise RuntimeError(
            f"figshare did not serve file content (HTTP {r.status_code}, "
            f"{r.headers.get('content-type')}). Expected a redirect to S3."
        )

    total = int(r.headers.get("content-length", 0))
    written = 0
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1e6:.0f} / {total / 1e6:.0f} MB", end="")
    print(f"\nDownloaded to {filename}")
    return filename


def unzip(archive, destination):
    """Replaces the original's `unzip` subprocess call."""
    print(f"Extracting {archive}")
    with zipfile.ZipFile(archive) as z:
        z.extractall(destination)


def cleanup(directory, names):
    """Replaces the original's `rm -rf` subprocess calls."""
    for name in names:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed {path}")


def preprocess_league(job):
    """
    Preprocess one league's event file into UIED format.

    Runs in its own process. The package's own preprocessing() parallelises over
    matches with a ThreadPoolExecutor, which is GIL-bound for pandas work and so
    runs close to serial; putting each league in a separate process is what
    actually uses more than one core.

    Also re-attaches the competition label. UIED_wyscout drops `comp` (it is not
    in the returned column list), but the train/valid/test split needs it to
    split within each competition rather than letting whole leagues land in one
    split.
    """
    league, event_path, matches_path, max_workers = job

    df = Event_data(
        data_provider="wyscout",
        event_path=os.path.join(event_path, f"events_{league}.json"),
        wyscout_matches_path=os.path.join(matches_path, f"matches_{league}.json"),
        preprocess_method="UIED",
        max_workers=max_workers,
    ).preprocessing()

    df["comp"] = league
    return league, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DS340W_DATA", "./data"),
        help="Where to download and write the dataset. Keep this outside OneDrive.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel workers for preprocessing. Raise for faster processing.",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    event_path = os.path.join(data_dir, "event")
    matches_path = os.path.join(data_dir, "matches")
    os.makedirs(data_dir, exist_ok=True)

    print(f"Data directory: {data_dir}\n")

    event_zip = download(EVENT_URL, event_path, EVENT_ZIP)
    matches_zip = download(MATCHES_URL, matches_path, MATCHES_ZIP)

    unzip(event_zip, event_path)
    unzip(matches_zip, matches_path)

    os.remove(event_zip)
    os.remove(matches_zip)
    cleanup(event_path, DISCARD)
    cleanup(matches_path, DISCARD)

    # UIED, not NMSTPP. The authors' example script uses preprocess_method="NMSTPP",
    # which emits a zone-based schema (zone, zone_s, zone_dist2goal, time_diff, ...)
    # that the Event package's training configs cannot consume - they ask for
    # action/delta_T/start_x/start_y plus Period, which only UIED produces.
    print(f"\nPreprocessing {len(LEAGUES)} leagues into UIED format, one process each...")
    jobs = [(lg, event_path, matches_path, args.max_workers) for lg in LEAGUES]

    frames = {}
    with ProcessPoolExecutor(max_workers=len(LEAGUES)) as pool:
        futures = {pool.submit(preprocess_league, j): j[0] for j in jobs}
        for fut in as_completed(futures):
            league = futures[fut]
            df_league = fut.result()[1]
            frames[league] = df_league
            print(f"  done: {league:9s} {len(df_league):>9,} rows")

    wyscout_df = pd.concat([frames[lg] for lg in LEAGUES if lg in frames])
    wyscout_df = wyscout_df.reset_index(drop=True)

    wyscout_df.to_csv(os.path.join(data_dir, "data.csv"), index=False)
    print(f"Wrote data.csv  ({len(wyscout_df):,} rows)")

    # Split by match within each competition, as in the original script.
    train_id, valid_id, test_id = [], [], []
    for comp in np.unique(wyscout_df[["comp"]]):
        temp = wyscout_df[wyscout_df["comp"] == comp]
        id_list = temp.match_id.unique()
        n = temp.match_id.nunique()
        train_id += id_list[0 : round(n * TRAIN_RATIO)].tolist()
        valid_id += id_list[
            round(n * TRAIN_RATIO) : round(n * (TRAIN_RATIO + VALID_RATIO))
        ].tolist()
        test_id += id_list[round(n * (TRAIN_RATIO + VALID_RATIO)) :].tolist()

    for name, ids in (("train", train_id), ("valid", valid_id), ("test", test_id)):
        split = wyscout_df[wyscout_df["match_id"].isin(ids)]
        out = os.path.join(data_dir, f"{name}.csv")
        split.to_csv(out, index=False)
        print(f"Wrote {name}.csv  ({len(split):,} rows, {len(ids)} matches)")

    print("---------------done-----------------")


if __name__ == "__main__":
    main()
