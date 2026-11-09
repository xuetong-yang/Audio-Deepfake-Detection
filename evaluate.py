import sys
import pickle
import pandas as pd
import numpy as np

# --- numpy module alias fix (same as in dataloader.py) ---
try:
    import numpy.core as _np_core
    import numpy.core.numeric as _np_numeric
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.numeric", _np_numeric)
except Exception:
    pass

from scripts.evaluation import calculate_eer

def safe_pickle_load(filepath: str):
    try:
        return pd.read_pickle(filepath)
    except Exception:
        with open(filepath, "rb") as f:
            try:
                return pickle.load(f)
            except Exception:
                f.seek(0)
                return pickle.load(f, encoding="latin1")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python eval_safe.py <prediction.pkl> <labels.pkl>")

    pred_path = sys.argv[1]
    lab_path = sys.argv[2]

    pred_df = safe_pickle_load(pred_path)
    lab_df = safe_pickle_load(lab_path)

    # merge by uttid to align
    df = pred_df.merge(lab_df, on="uttid", how="inner")
    if len(df) != len(lab_df):
        print(f"[warn] merged rows={len(df)} labels rows={len(lab_df)} (uttid mismatch?)")

    scores = df["predictions"].astype(float).to_numpy()
    labels = df["label"].astype(int).to_numpy()

    eer, thr = calculate_eer(scores, labels)
    print(f"EER={eer:.6f}  thr={thr:.6f}")
