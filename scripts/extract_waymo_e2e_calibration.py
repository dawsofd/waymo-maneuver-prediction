#!/usr/bin/env python3
"""Extract ONLY camera calibration (K, T, D) from Waymo E2E TFRecords.

Dependency-light: needs only ``protobuf`` plus a local clone of StandardE2E
(for its vendored .proto definitions). No TensorFlow, no torch, no full
standard_e2e install — runs on macOS or Linux.

Waymo does not ship calibration as a standalone file; it lives inside the
TFRecords in ``frame.context.camera_calibrations`` next to the (large) images.
This reads out ONLY the calibration (images are never touched) and writes a
small JSON keyed by segment_id — a few hundred KB you can commit to the repo,
then join to the existing .npz cache by ``segment_id``.

Per camera it records:
  K : 3x3 intrinsic  [[f_u,0,c_u],[0,f_v,c_v],[0,0,1]]
  T : 4x4 extrinsic (camera -> vehicle), row-major
  D : distortion [k1,k2,p1,p2,k3]  (Brown-Conrady)
  width, height  (native px — rescale K if you resize the image)

It also reports how many DISTINCT calibration sets ("rigs") appear, so you learn
whether a partial download already covers every segment.

Setup
-----
  pip install protobuf
  git clone https://github.com/stepankonev/StandardE2E   # for the vendored protos
  # gs:// inputs additionally need tensorflow-cpu + GCS auth — which you already
  # have, since `python -m standard_e2e.caching.process_source_dataset` runs.

Usage
-----
  # PROBE: stream a single gs:// shard, sample ~200 segments, report rig count.
  # --limit stops early so you don't stream the whole shard just to answer
  # "does calibration vary across segments?".
  python extract_waymo_e2e_calibration.py --repo ./StandardE2E --limit 200 \
      --input 'gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/training.tfrecord-00000-of-*' \
      --output calibration_probe.json

  # FULL: every segment in a split (drop --limit). Works with gs:// or local.
  python extract_waymo_e2e_calibration.py --repo ./StandardE2E \
      --input 'gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/training*' \
      --output calibration_training.json
"""
import argparse
import glob
import hashlib
import json
import os
import struct
import sys
import types

# Waymo CameraName enum (dataset.proto) -> readable name.
CAM_ID_TO_NAME = {
    1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT",
    5: "SIDE_RIGHT", 6: "REAR_LEFT", 7: "REAR", 8: "REAR_RIGHT",
}


def load_proto(repo_root):
    """Return the E2EDFrame proto module.

    Preferred: import it straight from the installed ``standard_e2e`` package
    (works in your processing conda env). Fallback: if the package isn't
    importable, use ``--repo`` (a StandardE2E clone) with a lightweight
    sys.modules shim so only the vendored _pb2 files load (protobuf-only host).
    """
    try:
        from standard_e2e.third_party.waymo_open_dataset.protos import (
            end_to_end_driving_data_pb2 as pb,
        )
        return pb
    except Exception as direct_err:
        if not repo_root:
            raise SystemExit(
                "Could not import the standard_e2e proto. Either run this in the "
                "conda env where standard_e2e is installed, or pass --repo "
                "pointing at a StandardE2E source clone.\n"
                f"(import error: {direct_err})")
        register_light_standard_e2e(repo_root)
        from standard_e2e.third_party.waymo_open_dataset.protos import (
            end_to_end_driving_data_pb2 as pb,
        )
        return pb


def register_light_standard_e2e(repo_root):
    """Import ``standard_e2e.third_party...`` WITHOUT executing the heavy
    top-level ``standard_e2e/__init__.py`` (which imports torch / tensorflow).

    We pre-seed sys.modules with lightweight package stubs whose ``__path__``
    points at the vendored dirs, so only the generated _pb2 files load.
    """
    base = os.path.join(repo_root, "standard_e2e")
    if not os.path.isdir(os.path.join(base, "third_party")):
        raise SystemExit(
            f"--repo does not look like a StandardE2E clone: {repo_root}")
    pkgs = {
        "standard_e2e": base,
        "standard_e2e.third_party": os.path.join(base, "third_party"),
        "standard_e2e.third_party.waymo_open_dataset":
            os.path.join(base, "third_party", "waymo_open_dataset"),
        "standard_e2e.third_party.waymo_open_dataset.protos":
            os.path.join(base, "third_party", "waymo_open_dataset", "protos"),
    }
    for name, path in pkgs.items():
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        mod.__package__ = name
        sys.modules[name] = mod


def tfrecord_iter_local(path):
    """Yield raw record payloads from an UNcompressed local TFRecord file.

    Format per record: <uint64 length><uint32 crc(length)><bytes><uint32 crc>.
    CRCs are skipped (not verified).
    """
    if path.endswith(".gz"):
        raise SystemExit(
            f"{path} looks gzip-compressed; Waymo E2E shards are normally "
            "uncompressed. If yours are gz, decompress first or add GZIP "
            "handling.")
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            (length,) = struct.unpack("<Q", header)
            f.read(4)                 # length CRC (skipped)
            data = f.read(length)
            if len(data) < length:
                break                 # truncated tail
            f.read(4)                 # data CRC (skipped)
            yield data


def list_inputs(pattern):
    """Expand a glob. gs:// patterns use TensorFlow's GCS-aware glob."""
    if pattern.startswith("gs://"):
        import tensorflow as tf  # you already have tensorflow-cpu for StandardE2E
        return sorted(tf.io.gfile.glob(pattern))
    return sorted(glob.glob(os.path.expanduser(pattern)))


def iter_records(paths):
    """Yield raw record bytes. gs:// paths stream via tf.data (same as your
    StandardE2E command); local paths use the pure-python reader."""
    if paths and str(paths[0]).startswith("gs://"):
        import tensorflow as tf
        for raw in tf.data.TFRecordDataset(paths, compression_type=""):
            yield raw.numpy()
    else:
        for p in paths:
            yield from tfrecord_iter_local(p)


def calib_for_frame(frame):
    """{camera_name: {K, T, D, width, height}} for one Waymo frame."""
    out = {}
    for cc in frame.context.camera_calibrations:
        name = CAM_ID_TO_NAME.get(cc.name, f"CAM_{cc.name}")
        it = list(cc.intrinsic)       # [f_u,f_v,c_u,c_v,k1,k2,p1,p2,k3]
        if len(it) < 4:
            continue
        K = [[it[0], 0.0, it[2]], [0.0, it[1], it[3]], [0.0, 0.0, 1.0]]
        T = list(cc.extrinsic.transform)
        T = [T[i:i + 4] for i in range(0, 16, 4)] if len(T) == 16 else T
        out[name] = {
            "K": K, "T": T, "D": [float(x) for x in it[4:9]],
            "width": int(cc.width), "height": int(cc.height),
        }
    return out


def rig_signature(calib):
    return hashlib.sha1(json.dumps(calib, sort_keys=True).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None,
                    help="Optional: path to a StandardE2E clone. Only needed if "
                    "the standard_e2e package is NOT installed in this env.")
    ap.add_argument("--input", required=True, help="Glob for TFRecord shards (quote it).")
    ap.add_argument("--output", required=True, help="Output JSON path.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after this many NEW segments (0 = no limit).")
    args = ap.parse_args()

    wod_e2ed_pb2 = load_proto(args.repo)

    files = list_inputs(args.input)
    if not files:
        raise SystemExit(f"No files matched: {args.input}")
    print(f"Scanning {len(files)} shard(s)...")

    calibrations, rigs, frames_seen = {}, {}, 0
    for raw in iter_records(files):
        frames_seen += 1
        data = wod_e2ed_pb2.E2EDFrame()
        data.ParseFromString(raw)
        segment_id = data.frame.context.name.split("-")[0]
        if segment_id in calibrations:
            continue                  # calibration is constant within a segment
        calib = calib_for_frame(data.frame)
        if not calib:
            continue
        calibrations[segment_id] = calib
        rigs.setdefault(rig_signature(calib), segment_id)
        if len(calibrations) % 100 == 0:
            print(f"  segments={len(calibrations)} rigs={len(rigs)} "
                  f"frames={frames_seen}")
        if args.limit and len(calibrations) >= args.limit:
            print(f"Reached --limit={args.limit}, stopping.")
            break

    payload = {
        "_summary": {
            "num_segments": len(calibrations),
            "num_distinct_rigs": len(rigs),
            "distinct_rig_example_segments": rigs,
            "frames_scanned": frames_seen,
            "camera_name_map": CAM_ID_TO_NAME,
            "intrinsic_layout": "K=[[f_u,0,c_u],[0,f_v,c_v],[0,0,1]]; "
                                "D=[k1,k2,p1,p2,k3]; T=camera->vehicle 4x4",
        },
        "calibrations": calibrations,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {args.output}")
    print(f"  segments      : {len(calibrations)}")
    print(f"  distinct rigs : {len(rigs)}")
    if len(rigs) == 1:
        print("  -> ONE rig for everything: a partial download already covers "
              "all segments; stop pulling shards.")
    elif rigs:
        print("  -> Multiple rigs: calibration varies; keep scanning shards "
              "until every segment is covered.")


if __name__ == "__main__":
    main()