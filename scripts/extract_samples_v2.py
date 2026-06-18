"""
Extract multiple sample frames per scenario class from a WOD-E2E tfrecord shard.
Uses naive JPEG byte-scanning (camera_0 / first JPEG in each record).

Usage:
    python scripts/extract_samples_v2.py
    python scripts/extract_samples_v2.py --target_per_class 3
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import tensorflow as tf

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_SHARD = "val_202504211843.tfrecord-00000-of-00093"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", default=DEFAULT_SHARD)
    parser.add_argument("--output_dir", default=str(DATA_DIR / "sample_frames_v2"))
    parser.add_argument("--target_per_class", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "val_sequence_name_to_scenario_cluster.json") as f:
        labels = json.load(f)

    saved = defaultdict(int)
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
        cluster = labels[seq_id]["scenario_cluster"]
        if saved[cluster] >= args.target_per_class:
            continue

        idx = b.find(b"\xff\xd8\xff")
        if idx == -1:
            continue
        end = b.find(b"\xff\xd9", idx) + 2

        safe_name = cluster.replace(" ", "_").replace("/", "_")
        fname = output_dir / f"{safe_name}_{saved[cluster]}.jpg"
        with open(fname, "wb") as f:
            f.write(b[idx:end])
        print(f"Saved: {fname}")
        saved[cluster] += 1

    print(f"Done: { {k: v for k, v in saved.items()} }")


if __name__ == "__main__":
    main()
