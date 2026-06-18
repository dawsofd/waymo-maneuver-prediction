import tensorflow as tf

dataset = tf.data.TFRecordDataset("val_202504211843.tfrecord-00000-of-00093")

for raw in dataset.take(1):
    b = raw.numpy()
    # Extract all 8 JPEGs
    pos = 0
    i = 0
    while True:
        idx = b.find(b'\xff\xd8\xff', pos)
        if idx == -1:
            break
        end = b.find(b'\xff\xd9', idx) + 2
        with open(f"camera_{i}.jpg", "wb") as f:
            f.write(b[idx:end])
        print(f"camera_{i}.jpg: {end-idx} bytes")
        i += 1
        pos = idx + 1