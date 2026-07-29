"""Shared loaders for the pooled 3-class evaluation notebooks.

Single source of truth for the seq_id join and fold assignment, so
`benchmark_pooled_3class.ipynb` and `efficiency_and_tuning.ipynb` can't drift.

Typical use from a notebook in notebooks/:

    from pathlib import Path
    import sys
    cur = Path.cwd(); ROOT = cur.parent if cur.name == 'notebooks' else cur
    sys.path.insert(0, str(ROOT / 'scripts'))
    import bench_common as bc

    ROOT, FD, FOLD_CSV = bc.resolve_paths()
    FOLD, ORDER, y, masks = bc.load_fold(FOLD_CSV)
    tr, va, te = masks['train'], masks['val'], masks['test']
    FAM = bc.assemble_families(FD, ORDER)
    Xof = lambda combo: bc.stack(FAM, combo)
"""
from pathlib import Path
import numpy as np
import pandas as pd

CLASSES = ['straight', 'right-turn', 'left-turn']
RS = 42

# family_name -> (train_arr, train_seqids, val_arr, val_seqids), all under the features dir
SOURCES = {
    'HOG':        ('hog.npy',                     'seq_ids.npy',                     'hog_val.npy',                'seq_ids_val.npy'),
    'HSV':        ('hsv.npy',                     'seq_ids.npy',                     'hsv_val.npy',                'seq_ids_val.npy'),
    'YOLO':       ('yolo.npy',                    'seq_ids.npy',                     'yolo_val.npy',               'seq_ids_val.npy'),
    'Road_v2':    ('road_v2_3class.npy',          'seq_ids_3class.npy',              'road_v2_val.npy',            'seq_ids_val.npy'),
    'VehicleOcc': ('vehicle_occupancy_3class.npy','seq_ids_3class.npy',              'vehicle_occupancy_val.npy',  'seq_ids_val.npy'),
    'Trend+Flow': ('framediff_v2_3class.npy',     'framediff_v2_3class_seq_ids.npy', 'framediff_v2_val.npy',       'seq_ids_val.npy'),
}


def resolve_paths(start=None):
    """Return (project_root, features_dir, fold_csv) whether run from the repo root or notebooks/."""
    cur = Path(start or Path.cwd()).resolve()
    root = cur.parent if cur.name == 'notebooks' else cur
    fd = root / 'data' / 'processed' / 'waymo_e2e' / 'features'
    fold_csv = root / 'data' / 'cnn_2d_seq_fold.csv'
    return root, fd, fold_csv


def load_fold(fold_csv):
    """Load the frozen fold file. Returns (fold_df, order, y, masks) in cnn_row order."""
    fold = pd.read_csv(fold_csv).sort_values('cnn_row').reset_index(drop=True)
    order = fold['seq_id'].astype(str).tolist()
    y = fold['class_label'].to_numpy()
    foldv = fold['fold'].to_numpy()
    masks = {'train': foldv == 'train', 'val': foldv == 'val', 'test': foldv == 'test'}
    return fold, order, y, masks


def _load(fd, name):
    return np.load(Path(fd) / name, allow_pickle=True)


def _two_source(fd, order, tr_arr, tr_sid, va_arr, va_sid):
    A = _load(fd, tr_arr); ts = _load(fd, tr_sid).astype(str)
    B = _load(fd, va_arr); vs = _load(fd, va_sid).astype(str)
    d = {s: A[i] for i, s in enumerate(ts)}
    for i, s in enumerate(vs):
        d[s] = B[i]
    missing = [s for s in order if s not in d]
    if missing:
        raise KeyError(f'{len(missing)} pool seqs have no feature row (e.g. {missing[:2]})')
    return np.stack([np.asarray(d[s], np.float32) for s in order])


def assemble_families(fd, order, sources=SOURCES, include_learned=True, verbose=True):
    """Assemble every available feature family into a dict, laid out in `order` (by seq_id).

    Two-source families (train array + *_val array) come from `sources`. Learned single-array
    families (CNN_2D.npy in cnn_row order; av_embedding.npy keyed by its own seq_ids) load if present.
    """
    fd = Path(fd)
    fam = {}
    for name, s in sources.items():
        try:
            fam[name] = _two_source(fd, order, *s)
            if verbose:
                print(f'  {name:12s}{fam[name].shape}')
        except FileNotFoundError:
            if verbose:
                print(f'  {name:12s} skipped (missing file)')
    if include_learned:
        if (fd / 'CNN_2D.npy').exists():
            A = _load(fd, 'CNN_2D.npy')
            if len(A) == len(order):
                fam['CNN'] = np.asarray(A, np.float32)
                if verbose:
                    print(f'  {"CNN":12s}{fam["CNN"].shape}')
            elif verbose:
                print(f'  CNN skipped (rows {len(A)} != pool {len(order)})')
        if (fd / 'av_embedding.npy').exists():
            A = _load(fd, 'av_embedding.npy')
            s = _load(fd, 'av_embedding_seq_ids.npy').astype(str)
            dd = {sid: A[i] for i, sid in enumerate(s)}
            if all(sid in dd for sid in order):
                fam['AV'] = np.stack([np.asarray(dd[sid], np.float32) for sid in order])
                if verbose:
                    print(f'  {"AV":12s}{fam["AV"].shape}')
    return fam


def stack(fam, combo):
    """Concatenate the named families (a list) into one feature matrix."""
    return np.concatenate([fam[f] for f in combo], axis=1)