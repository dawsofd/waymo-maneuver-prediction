"""
Inspect the structure of a single WOD-E2E tfrecord — useful for understanding
the raw data format before using StandardE2E for proper extraction.

Usage:
    python scripts/inspect_one_record.py
    python scripts/inspect_one_record.py --shard val_202504211843.tfrecord-00000-of-00093
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
    parser.add_argument("--n_records", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(DATA_DIR / "val_sequence_name_to_scenario_cluster.json") as f:
        labels = json.load(f)

    dataset = tf.data.TFRecordDataset(str(DATA_DIR / args.shard))

    for i, raw in enumerate(dataset.take(args.n_records)):
        b = raw.numpy()
        example = tf.train.Example()
        example.ParseFromString(b)

        print(f"\n--- Record {i} ---")
        print(f"Total bytes: {len(b)}")
        print(f"First 16 hex: {b[:16].hex()}")

        for key, val in example.features.feature.items():
            print(f"Key: {key}")
            seq_id = key.split("-")[0]
            if seq_id in labels:
                print(f"  -> Scenario: {labels[seq_id]['scenario_cluster']}")
            if val.HasField("bytes_list"):
                b0 = val.bytes_list.value[0] if val.bytes_list.value else b""
                print(f"  bytes_list, len={len(b0)}, first4={b0[:4].hex()}")
            elif val.HasField("int64_list"):
                print(f"  int64_list: {list(val.int64_list.value)[:5]}")
            elif val.HasField("float_list"):
                print(f"  float_list: {list(val.float_list.value)[:5]}")

        # Find all JPEG markers
        pos, jpegs = 0, []
        while True:
            idx = b.find(b"\xff\xd8\xff", pos)
            if idx == -1:
                break
            jpegs.append(idx)
            pos = idx + 1
        print(f"JPEG markers found at positions: {jpegs}")


if __name__ == "__main__":
    main()
