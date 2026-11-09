# Model Description

## Data
The inputs are pre-extracted features. 
Each utterance feature has tensor shape: [D, T] = [180, 321]
I use a fixed time length T=321 as given in the project setup.
Data augmentation: time masking and feature masking during training to improve robustness to unseen attacks.

## Model
This is a deepfake audio detection model for binary classification.
I use 1D CNN modeling along time. 
CNN expects input as [B, C, L], so feature dimension D=180 is treated as C(channels) and time frame=321 is treated as sequence length.
The network consists of three Conv1d blocks followed by global average pooling over time. Each block contains Conv1d, BatchNorm1d, ReLU and MaxPool1d. So that I can extract more abstract features layer by layer.
After pooling, a small MLP head maps 256 → 128 → 1 to produce the final logit. I choose logits because they are raw scores and run stable with BCEWithLogitsLoss.
( I also tested BiLSTM baseline, but it performed poorly in several rounds of test, so I did not use it)

## Loss
Loss: BCEWithLogitsLoss.

## Others
Optimizer: Adam with lr = 1e-3 and weight_decay = 5e-4, batch size = 8.
Early stopping: I monitor dev set EER and apply early stopping (patience=10), saving the best checkpoint by dev EER.
Regularization: Includes dropout = 0.3 in the MLP head and weight decay.
Inference: The trained model is applied to the test features in evaluation mode to produce one logit per utterance, which is then used for submission.
By turning the training configuration into command line arguments, I can quickly compare hyperparameter settings.
