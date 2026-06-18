import tensorflow as tf
import json
import os

with open("val_sequence_name_to_scenario_cluster.json") as f:
    labels = json.load(f)

os.makedirs("construction_all", exist_ok=True)

dataset = tf.data.TFRecordDataset("val_202504211843.tfrecord-00000-of-00093")
saved = 0

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
    if labels[seq_id]["scenario_cluster"] != "Construction":
        continue

    idx = b.find(b'\xff\xd8\xff')
    if idx == -1:
        continue
    end = b.find(b'\xff\xd9', idx) + 2

    fname = f"construction_all/{seq_id}.jpg"
    with open(fname, "wb") as f:
        f.write(b[idx:end])
    saved += 1

print(f"Saved {saved} Construction frames")