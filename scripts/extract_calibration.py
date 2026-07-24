#!/usr/bin/env python3
"""Sanity-check that the re-cached Waymo E2E .npz files carry camera calibration.

Run in the SAME conda env you process with (needs the standard_e2e package, so
the pickled CameraData objects unpickle natively). Point it at the NEW output.

  python check_calibration.py --dir ./data/processed_calib/waymo_e2e/training

It loads the most recently written .npz, digs out the `cameras` modality, and
prints K / T / D for each camera. If it prints a 3x3 K with sensible fx/fy/cx/cy,
the calibration survived preprocessing and you can let the job finish. If cameras
is still a bare image array (no K), the config didn't take — kill the job.
"""
import argparse
import glob
import os

import numpy as np


def unwrap(o):
    return o.item() if isinstance(o, np.ndarray) and o.dtype == object and o.shape == () else o


def find_cameras(modality_data):
    """modality_data: dict keyed by Modality -> value. Return the cameras value."""
    for k, v in modality_data.items():
        name = getattr(k, "value", str(k)).lower()
        if "camera" in name:            # Modality.CAMERAS -> "cameras"
            return unwrap(v)
    return None


def describe_camera(cam):
    """cam is a CameraData; print its calibration."""
    K = getattr(cam, "K", getattr(cam, "intrinsics", None))
    T = getattr(cam, "T", getattr(cam, "extrinsics", None))
    D = getattr(cam, "D", getattr(cam, "distortion", None))
    img = getattr(cam, "image", None)
    K = np.asarray(K) if K is not None else None
    if K is not None and K.shape == (3, 3):
        print(f"    K = [[{K[0,0]:.1f}, 0, {K[0,2]:.1f}], "
              f"[0, {K[1,1]:.1f}, {K[1,2]:.1f}], [0,0,1]]   (fx,fy,cx,cy)")
        print(f"    T shape={np.asarray(T).shape if T is not None else None}, "
              f"D={None if D is None else np.asarray(D).ravel().tolist()}, "
              f"image={None if img is None else np.asarray(img).shape}")
        return True
    print(f"    NO K found on this camera object (type={type(cam).__name__}). "
          "Calibration did NOT survive.")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Output split dir, e.g. "
                    "./data/processed_calib/waymo_e2e/training")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.dir, "*.npz"))
    if not files:
        raise SystemExit(f"No .npz yet in {args.dir} — wait for a few to appear.")
    p = max(files, key=os.path.getmtime)
    print(f"Inspecting newest file: {os.path.basename(p)}\n")

    z = np.load(p, allow_pickle=True)
    md = unwrap(z["_modality_data"])
    if not isinstance(md, dict):
        raise SystemExit(f"_modality_data is {type(md).__name__}, expected dict.")

    cams = find_cameras(md)
    if cams is None:
        raise SystemExit("No 'cameras' modality found in _modality_data.")

    ok = False
    if isinstance(cams, dict):
        # {CameraDirection: CameraData}  <- what CamerasIdentityAdapter produces
        for direction, cam in cams.items():
            dname = getattr(direction, "value", str(direction))
            print(f"  camera: {dname}")
            ok = describe_camera(unwrap(cam)) or ok
    elif isinstance(cams, np.ndarray):
        print(f"  cameras is a bare ndarray {cams.shape} — this is the OLD pano "
              "output. Calibration did NOT survive; check the config.")
    else:
        print(f"  camera:")
        ok = describe_camera(cams)

    print("\n" + ("PASS: calibration is present — let the job run."
                  if ok else "FAIL: no intrinsics — kill the job and recheck the config."))


if __name__ == "__main__":
    main()