import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from tqdm import tqdm

from dataloader import get_dataloaders, get_test_dataloader
from model import ConvAudioModel, RNNAudioModel

# Use the official project evaluation script
from scripts.evaluation import calculate_eer

def spec_augment_batch(
    x: torch.Tensor,
    time_mask_param: int = 30,     # max width of time mask (in frames)
    freq_mask_param: int = 18,     # max width of feature mask (in dims)
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
    p: float = 0.9,                # probability to apply augmentation
) -> torch.Tensor:
    """
    SpecAugment-style masking for features shaped [B, D, T].
    - Time masking masks along T dimension.
    - Freq/feature masking masks along D dimension.
    """
    if (not x.is_floating_point()) or random.random() > p:
        return x

    B, D, T = x.shape
    out = x.clone()

    # Time masks (mask contiguous time steps)
    for _ in range(num_time_masks):
        if T <= 1:
            break
        w = random.randint(0, min(time_mask_param, T - 1))
        if w == 0:
            continue
        t0 = random.randint(0, T - w)
        out[:, :, t0:t0 + w] = 0.0

    # Feature masks (mask contiguous feature dims)
    for _ in range(num_freq_masks):
        if D <= 1:
            break
        w = random.randint(0, min(freq_mask_param, D - 1))
        if w == 0:
            continue
        d0 = random.randint(0, D - w)
        out[:, d0:d0 + w, :] = 0.0

    return out

# Compute EER on the dev set
def evaluate(model, dev_loader, device):
    model.eval()  # dropout off, BatchNorm uses running statistics: keeps EER stable
    # Collect scores and labels of all samples, then compute EER once
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="eval", leave=False):
            x = batch["features"].to(device)          # [B, D, T] = [B, 180, 321]
            y = batch["label"].to(device).view(-1)    # [B]; view(-1) avoids [B, 1] vs [B] mismatches

            logits = model(x).view(-1)                # [B] (logits)
            all_scores.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    scores = torch.cat(all_scores, dim=0).numpy()  # [N_dev]
    labels = torch.cat(all_labels, dim=0).numpy()

    eer, threshold = calculate_eer(scores, labels)

    model.train()  # switch back for the next epoch
    return float(eer), float(threshold)

# Run inference and write prediction.pkl
def predict(model, test_loader, device, output_path="prediction.pkl"):
    model.eval()
    uttids_all = []
    preds_all = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="predict", leave=False):
            uttids = batch["uttid"]                   # list[str]
            x = batch["features"].to(device)          # [B, D, T] = [B, 180, 321]
            logits = model(x).view(-1)                # [B] logits
            preds_all.extend(logits.detach().cpu().tolist())
            uttids_all.extend(list(uttids))

    pd.DataFrame({"uttid": uttids_all, "predictions": preds_all}).to_pickle(output_path)
    print(f"Saved predictions to {output_path}")


def train_epoch(model, train_loader, criterion, optimizer, device, log_every=20):
    model.train()
    running = 0.0  # sum of batch losses, used for the epoch average

    for step, batch in enumerate(tqdm(train_loader, desc="train", leave=False), start=1):
        x = batch["features"].to(device)              # [B, D, T] = [B, 180, 321]
        x = spec_augment_batch(x, time_mask_param=30, freq_mask_param=18, num_time_masks=2, num_freq_masks=2, p=0.9)
        y = batch["label"].to(device).view(-1)        # [B]
        
        optimizer.zero_grad()
        logits = model(x).view(-1)      # forward
        loss = criterion(logits, y)     # compute loss
        loss.backward()                 # backward
        optimizer.step()                # update parameters

        running += loss.item()
        if step % log_every == 0:
            print(f"  step {step:4d}/{len(train_loader)}  loss={loss.item():.4f}")

    return running / max(1, len(train_loader))


def main():  # all training settings are command-line arguments, so experiments are easy to compare
    parser = argparse.ArgumentParser(description="Audio deepfake detection training")

    parser.add_argument("--train_features", type=str, required=True)
    parser.add_argument("--train_labels", type=str, required=True)
    parser.add_argument("--dev_features", type=str, required=True)
    parser.add_argument("--dev_labels", type=str, required=True)

    parser.add_argument("--test_features", type=str, default=None, help="optional, for prediction.pkl generation")
    parser.add_argument("--prediction_out", type=str, default="prediction.pkl")

    parser.add_argument("--model", type=str, default="conv", choices=["conv", "rnn"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)  # L2 regularization
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--patience", type=int, default=10)  # early stopping: epochs without dev EER improvement

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Data
    train_loader, dev_loader = get_dataloaders(
        args.train_features, args.train_labels,
        args.dev_features, args.dev_labels,
        batch_size=args.batch_size
    )

    # Model (feature_dim is D=180, not T=321)
    if args.model == "conv":
        model = ConvAudioModel(feature_dim=180).to(device)
    else:
        model = RNNAudioModel(feature_dim=180).to(device)

    # Loss / Optim
    criterion = nn.BCEWithLogitsLoss() # sigmoid + BCE
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    # Halve the learning rate when dev EER has not improved for 5 epochs.
    # Learn fast with a large lr first, then fine-tune with a smaller one.


    # Training loop
    best_eer = float("inf")
    best_epoch = -1
    patience_counter = 0
    best_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        dev_eer, dev_thr = evaluate(model, dev_loader, device)

        print(f"train_loss={train_loss:.4f}  dev_eer={dev_eer:.4f}  (thr={dev_thr:.4f})")
        scheduler.step(dev_eer)
        # Each epoch: train, evaluate on dev, then let dev EER drive the scheduler

        # Save the checkpoint whenever dev EER improves
        if dev_eer < best_eer:
            best_eer = dev_eer
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": args.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_eer": best_eer,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path} (EER={best_eer:.4f})")
        # Early stopping: stop after `patience` epochs without a new best EER
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs. Best epoch={best_epoch}")
                break

    print(f"\nTraining done. Best dev EER={best_eer:.4f} at epoch={best_epoch}")

    # Prediction (optional)
    if args.test_features is not None:
        print("\nRunning prediction on test set...")
        ckpt = torch.load(best_path, map_location=device)  # reload the best checkpoint before inference
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        test_loader = get_test_dataloader(args.test_features, batch_size=args.batch_size)
        predict(model, test_loader, device, output_path=args.prediction_out)


if __name__ == "__main__":
    main()
