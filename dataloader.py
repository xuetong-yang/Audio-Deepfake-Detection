"""
Data loading for audio deepfake detection.

Project spec: each 'features' is a torch.Tensor of shape [180, 321] (D=180, T=321).
We return batches as:
  features: [B, 180, 321] → [B, D, T]
  label:    [B] (float32 0/1) for train/dev
  uttid:    list[str]
"""

import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np

try:
    import numpy.core as _np_core
    import numpy.core.numeric as _np_numeric
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.numeric", _np_numeric)
except Exception:
    pass

def safe_pickle_load(filepath: str):
    """
    Robust loader for .pkl saved in slightly different numpy/pandas environments.
    Tries pandas.read_pickle first; falls back to pickle.load for broader exception types.
    """
    try:
        return pd.read_pickle(filepath)
    except Exception as e:
        # fallback
        with open(filepath, "rb") as f:
            try:
                return pickle.load(f)
            except Exception:
                # try latin1 as last resort (old python2 pickles, etc.)
                f.seek(0)
                return pickle.load(f, encoding="latin1")


class AudioFeaturesDataset(Dataset):
    def __init__(self, features_path, labels_path=None):  # load everything into memory once
        self.features_df = safe_pickle_load(features_path)  # expects columns: uttid, features

        self.label_map = None  # dict {uttid: label}, used to look up the label of each sample
        if labels_path is not None:
            labels_df = safe_pickle_load(labels_path)       # expects columns: uttid, label
            self.label_map = dict(zip(labels_df["uttid"].tolist(), labels_df["label"].tolist()))

        # Cache columns for faster __getitem__
        self.uttids = self.features_df["uttid"].tolist()
        self.features = self.features_df["features"].tolist()

    def __len__(self): # the length of the dataset
        return len(self.uttids) 

    def __getitem__(self, idx: int):  # return one sample in a fixed, validated shape
        uttid = self.uttids[idx] 
        feat = self.features[idx]  # torch.Tensor expected, shape [180, 321]

        if not torch.is_tensor(feat):
            feat = torch.tensor(feat, dtype=torch.float32)
        else:
            feat = feat.float()

        # Enforce 2D tensor
        if feat.dim() != 2:
            raise ValueError(f"features must be 2D, got shape {tuple(feat.shape)} for uttid={uttid}")

        # We expect [D, T] = (180, 321). If stored as [T, D] = (321, 180), fix it.
        if feat.shape == (321, 180):
            feat = feat.transpose(0, 1)

        if feat.shape != (180, 321):
            raise ValueError(f"Expected features shape (180,321) as [D,T], got {tuple(feat.shape)} for uttid={uttid}")
        
        item = {"uttid": uttid, "features": feat}

        if self.label_map is not None:
            if uttid not in self.label_map:
                raise KeyError(f"uttid {uttid} not found in labels.pkl")
            item["label"] = torch.tensor(float(self.label_map[uttid]), dtype=torch.float32)
            # Store the label as a float32 tensor, the format BCEWithLogitsLoss expects.
            # turn uttid to float32 for BCEWithLogitsLoss, and then turn 0.0/1.0 into a PyTorch tensor
            # collate_fn later stacks the labels of a batch into a [B] vector.

        return item


def collate_fn(batch):  # combine a list of sample dicts into one batch dict
    # DEBUG: find None entries
    for i, b in enumerate(batch):
        if b is None:
            raise RuntimeError(f"collate_fn received None at index {i} in batch. "
                               f"Your Dataset.__getitem__ returned None for some sample.")
    # Since D and T is fixed, we can stack directly.
    uttids = [b["uttid"] for b in batch]
    feats = torch.stack([b["features"] for b in batch], dim=0)  
    # [B, 180, 321]
    # torch.stack requires all tensors to have the same shape.

    out = {"uttid": uttids, "features": feats}

    if "label" in batch[0]:  # labels exist for train/dev only
        labels = torch.stack([b["label"] for b in batch], dim=0).view(-1)  # [B]
        out["label"] = labels

    return out

# Wrap the datasets in DataLoaders
def get_dataloaders(train_features_path, train_labels_path,
                    dev_features_path, dev_labels_path,
                    batch_size=32, num_workers=0): # here batchsize=32 is default
    train_ds = AudioFeaturesDataset(train_features_path, train_labels_path)
    dev_ds = AudioFeaturesDataset(dev_features_path, dev_labels_path)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, # shuffle for training
        num_workers=num_workers, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, # no shuffle for dev
        num_workers=num_workers, collate_fn=collate_fn
    )
    return train_loader, dev_loader


def get_test_dataloader(test_features_path, batch_size=32, num_workers=0):
    test_ds = AudioFeaturesDataset(test_features_path, labels_path=None)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    return test_loader

