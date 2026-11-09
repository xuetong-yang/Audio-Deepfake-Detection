import os
import sys
import pickle
import pandas as pd
import numpy as np

# --- numpy pickle compatibility shim (same as in dataloader.py) ---
try:
    import numpy.core as _np_core
    import numpy.core.numeric as _np_numeric
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.numeric", _np_numeric)
except Exception:
    pass

BASE = r"data"  # run inside project/


def safe_pickle_load(filepath: str):
    """
    Robust loader for .pkl saved in slightly different numpy/pandas environments.
    Tries pandas.read_pickle first; falls back to pickle.load with broader exception handling.
    """
    try:
        return pd.read_pickle(filepath)
    except Exception:
        with open(filepath, "rb") as f:
            try:
                return pickle.load(f)
            except Exception:
                f.seek(0)
                return pickle.load(f, encoding="latin1")


def load_df(path):
    df = safe_pickle_load(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} is not a DataFrame, got {type(df)}")
    return df


def check_split(name, feat_path, lab_path=None):
    print(f"\n===== {name} =====")
    fdf = load_df(feat_path)

    assert "uttid" in fdf.columns, f"{feat_path} missing 'uttid'"
    assert "features" in fdf.columns, f"{feat_path} missing 'features'"

    n_feat = len(fdf)
    uttids_feat = fdf["uttid"].astype(str)
    dup_feat = uttids_feat.duplicated().sum()

    print(f"features rows: {n_feat}")
    print(f"features uttid duplicates: {dup_feat}")

    # Feature shape sanity (sample)
    shapes = []
    for i in np.random.choice(n_feat, size=min(10, n_feat), replace=False):
        x = fdf.iloc[int(i)]["features"]
        shp = tuple(x.shape) if hasattr(x, "shape") else None
        shapes.append(shp)
    print(f"sample feature shapes (up to 10): {shapes}")

    if lab_path is None:
        return set(uttids_feat.tolist()), None

    ldf = load_df(lab_path)
    assert "uttid" in ldf.columns, f"{lab_path} missing 'uttid'"
    assert "label" in ldf.columns, f"{lab_path} missing 'label'"

    n_lab = len(ldf)
    uttids_lab = ldf["uttid"].astype(str)
    dup_lab = uttids_lab.duplicated().sum()

    print(f"labels rows: {n_lab}")
    print(f"labels uttid duplicates: {dup_lab}")

    unique_labels = sorted(ldf["label"].unique().tolist())
    print(f"label unique values: {unique_labels}")

    set_feat = set(uttids_feat.tolist())
    set_lab = set(uttids_lab.tolist())

    missing_in_lab = set_feat - set_lab
    extra_in_lab = set_lab - set_feat

    print(f"uttids in features but missing in labels: {len(missing_in_lab)}")
    print(f"uttids in labels but missing in features: {len(extra_in_lab)}")

    if missing_in_lab:
        print("  examples missing_in_labels:", list(sorted(missing_in_lab))[:10])
    if extra_in_lab:
        print("  examples extra_in_labels:", list(sorted(extra_in_lab))[:10])

    if dup_lab > 0:
        dups = uttids_lab[uttids_lab.duplicated(keep=False)]
        print("  WARNING: duplicated label uttids examples:", dups.unique()[:10].tolist())

    return set_feat, ldf


def label_distribution_smoke_test(train_labels_df, dev_labels_df):
    def ratio(df):
        y = df["label"].astype(int).values
        p1 = (y == 1).mean()
        p0 = (y == 0).mean()
        return p0, p1

    tr0, tr1 = ratio(train_labels_df)
    dv0, dv1 = ratio(dev_labels_df)

    print("\n===== Label distribution smoke test =====")
    print(f"train: p(label=0)={tr0:.4f}, p(label=1)={tr1:.4f}")
    print(f"dev  : p(label=0)={dv0:.4f}, p(label=1)={dv1:.4f}")

    if max(tr0, tr1) > 0.95:
        print("  WARNING: train labels extremely imbalanced (>95%).")
    if max(dv0, dv1) > 0.95:
        print("  WARNING: dev labels extremely imbalanced (>95%).")


def main():
    train_feat = os.path.join(BASE, "train", "features.pkl")
    train_lab  = os.path.join(BASE, "train", "labels.pkl")
    dev_feat   = os.path.join(BASE, "dev", "features.pkl")
    dev_lab    = os.path.join(BASE, "dev", "labels.pkl")
    test_feat  = os.path.join(BASE, "test1", "features.pkl")

    train_set, train_ldf = check_split("train", train_feat, train_lab)
    dev_set, dev_ldf     = check_split("dev", dev_feat, dev_lab)
    test_set, _          = check_split("test1", test_feat, lab_path=None)

    print("\n===== Cross-split uttid overlap =====")
    print(f"train ∩ dev   : {len(train_set & dev_set)}")
    print(f"train ∩ test1 : {len(train_set & test_set)}")
    print(f"dev   ∩ test1 : {len(dev_set & test_set)}")

    if len(train_set & dev_set) > 0:
        print("  WARNING: train/dev overlap examples:", list(sorted(train_set & dev_set))[:20])
    if len(train_set & test_set) > 0:
        print("  WARNING: train/test overlap examples:", list(sorted(train_set & test_set))[:20])
    if len(dev_set & test_set) > 0:
        print("  WARNING: dev/test overlap examples:", list(sorted(dev_set & test_set))[:20])

    label_distribution_smoke_test(train_ldf, dev_ldf)

    print("\n===== Summary =====")
    print("Believability checks:")
    print("  - labels uttid duplicates == 0")
    print("  - for train/dev: feature uttids == label uttids (missing/extra == 0)")
    print("  - train/dev/test overlaps == 0")
    print("  - labels not extremely imbalanced")


if __name__ == "__main__":
    main()
