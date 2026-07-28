#!/usr/bin/env python3
"""
Generate the CNN_2D split CSVs that notebooks/CNN_2D.ipynb reads, directly from the
frozen canonical fold assignment (data/cnn_2d_seq_fold.csv).

Why this instead of re-running CNN_2D_resplit.ipynb:
  - It guarantees CNN_2D.ipynb trains on the exact same train/val/test folds every other
    feature family uses (the frozen fold file), rather than re-deriving a stochastic split.
  - Because rows are written in cnn_row order, the CNN_2D.npy the notebook exports
    (train, then val, then test) aligns row-for-row to cnn_2d_seq_ids.npy with NO edit to
    Matt's notebook.

Outputs (to data/processed/waymo_e2e/CNN_2D/):
  df_train.csv, df_val.csv, df_test.csv   (columns: target_frame_path, class_label)
  y_train.csv,  y_val.csv,  y_test.csv    (column:  class_label)

Run from the repo root:
    python scripts/gen_cnn2d_split_csvs.py
then Run All on notebooks/CNN_2D.ipynb.
"""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FOLD_CSV = REPO / 'data' / 'cnn_2d_seq_fold.csv'
OUT_DIR  = REPO / 'data' / 'processed' / 'waymo_e2e' / 'CNN_2D'
SPLITS   = ['train', 'val', 'test']


def main():
    if not FOLD_CSV.exists():
        raise FileNotFoundError(f'Fold file not found: {FOLD_CSV}')

    fold = pd.read_csv(FOLD_CSV).sort_values('cnn_row').reset_index(drop=True)
    need = {'cnn_row', 'seq_id', 'fold', 'class_label', 'target_frame_path'}
    missing = need - set(fold.columns)
    if missing:
        raise ValueError(f'{FOLD_CSV.name} is missing columns: {sorted(missing)}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        sub = fold[fold['fold'] == split]                 # already in cnn_row order
        df_path = OUT_DIR / f'df_{split}.csv'
        y_path  = OUT_DIR / f'y_{split}.csv'
        sub[['target_frame_path', 'class_label']].to_csv(df_path, index=False)
        sub[['class_label']].to_csv(y_path, index=False)
        counts = sub['class_label'].value_counts().reindex(
            ['straight', 'right-turn', 'left-turn']).to_dict()
        print(f'{split:5s}: {len(sub):4d} rows -> {df_path.name}, {y_path.name}   {counts}')

    print(f'\nWrote 6 CSVs to {OUT_DIR}')
    print('Row order == cnn_row order in cnn_2d_seq_fold.csv, so the exported CNN_2D.npy '
          '(train+val+test) aligns to cnn_2d_seq_ids.npy.')


if __name__ == '__main__':
    main()
