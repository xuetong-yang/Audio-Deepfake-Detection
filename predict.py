import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm

from dataloader import get_test_dataloader
from model import ConvAudioModel, RNNAudioModel

# Load a trained checkpoint, run a forward pass over the test features,
# and write the scores to prediction.pkl.

# Rebuild the same architecture that was trained, then load its weights
def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_type = ckpt.get("model_type", "conv") 

    # The architecture must match the checkpoint for load_state_dict to work
    if model_type == "conv":
        model = ConvAudioModel(feature_dim=180)
    elif model_type == "rnn":
        model = RNNAudioModel(feature_dim=180)
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_type


@torch.no_grad()
def predict(model, test_loader, device: torch.device):
    uttids_all = []
    preds_all = []

    for batch in tqdm(test_loader, desc="predict"):
        x = batch["features"].to(device)     # [B, D, T] = [B,180,321]
        uttids = batch["uttid"]              # list[str]
        logits = model(x).view(-1)           # [B]
        uttids_all.extend(uttids)
        preds_all.extend(logits.cpu().tolist())

    return uttids_all, preds_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_features", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="prediction.pkl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    model, model_type = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint model_type={model_type} from {args.checkpoint}")

    test_loader = get_test_dataloader(args.test_features, batch_size=args.batch_size)

    uttids, preds = predict(model, test_loader, device)

    df = pd.DataFrame({"uttid": uttids, "predictions": preds})
    df.to_pickle(args.output)
    print(f"Saved {len(df)} predictions to {args.output}")
    print(df.head())


if __name__ == "__main__":
    main()
