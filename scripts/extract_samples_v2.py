import tensorflow as tf
import json
from collections import defaultdict

with open("val_sequence_name_to_scenario_cluster.json") as f:
    labels = json.load(f)

# Find all Construction sequence IDs
construction_seqs = {k for k, v in labels.items() if v["scenario_cluster"] == "Construction"}
print(f"Total Construction sequences: {len(construction_seqs)}")

# See which ones appear in shard 00000
dataset = tf.data.TFRecordDataset("val_202504211843.tfrecord-00000-of-00093")
found = []
for raw in dataset:
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    keys = list(example.features.feature.keys())
    if not keys:
        continue
    seq_id = keys[0].split("-")[0]
    if seq_id in construction_seqs:
        found.append(seq_id)

print(f"Construction sequences in shard 00000: {len(set(found))}")
print(set(found))