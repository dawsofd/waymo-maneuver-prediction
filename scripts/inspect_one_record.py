import tensorflow as tf

dataset = tf.data.TFRecordDataset("val_202504211843.tfrecord-00000-of-00093")

for raw in dataset.take(1):
    b = raw.numpy()
    # Find all JPEG start markers
    pos = 0
    jpegs = []
    while True:
        idx = b.find(b'\xff\xd8\xff', pos)
        if idx == -1:
            break
        jpegs.append(idx)
        pos = idx + 1
    print(f"JPEG markers found at positions: {jpegs}")
    
    # Save the first one to check
    if jpegs:
        # Find end marker too
        end = b.find(b'\xff\xd9', jpegs[0]) + 2
        with open("test_frame.jpg", "wb") as f:
            f.write(b[jpegs[0]:end])
        print(f"Saved test_frame.jpg ({end - jpegs[0]} bytes)")