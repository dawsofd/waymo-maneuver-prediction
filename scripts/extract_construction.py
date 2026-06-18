"""
Extract all Construction scenario frames from a WOD-E2E tfrecord shard.
Uses naive JPEG byte-scanning (camera_0 / first JPEG in each record).

Usage:
    python scripts/extract_construction.py
    python scripts/extract_construction.py --shard val_202504211843.tfrecord-00000-of-00093
"""
import argparse
import json
from pathlib import Path

import tensorflow as tf

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_SHARD = "val_202504211843.tfrecord-00000-of-00093"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", default=DEFAULT_SHARD)
    parser.add_argument("--output_dir", default=str(DATA_DIR / "construction_all"))
    parser.add_argument("--cluster", default="Construction",
                        help="Scenario cluster to extract (default: Construction)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "val_sequence_name_to_scenario_cluster.json") as f:
        labels = json.load(f)

    saved = 0
    dataset = tf.data.TFRecordDataset(str(DATA_DIR / args.shard))

    for raw in dataset:
        b = raw.numpy()
        example = tf.train.Example()
        example.ParseFromString(b)
        keys = list(example.features.feature.keys())
        if not keys:
            continue
        seq_id = keys[0].split("-")[0]
        if seq_id not in labels:
            continue
        if labels[seq_id]["scenario_cluster"] != args.cluster:
            continue

        idx = b.find(b"\xff\xd8\xff")
        if idx == -1:
            continue
        end = b.find(b"\xff\xd9", idx) + 2

        fname = output_dir / f"{seq_id}.jpg"
        with open(fname, "wb") as f:
            f.write(b[idx:end])
        saved += 1

    print(f"Saved {saved} {args.cluster} frames to {output_dir}")


if __name__ == "__main__":
    main()
