import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from src.DKTplus.model import DKTplus
from src.DKTplus.dataset import DKTplus_dataset, DKTplus_collate

# Data path
PROCESSED_DATA_PATH = "./data/ASSISTments/ASSISTments_processed_data.json"
DKT_MODEL_PATH = "./models/DKTplus.pt"

# Model Hyperparameters
BATCH_SIZE = 64
EPOCHS = 25
NUM_C = 44
LEARNING_RATE = 0.001
EMB_SIZE = 128
HIDDEN_SIZE = 256
DROPOUT = 0.3
LAMBDA_R, LAMBDA_W1, LAMBDA_W2 = 0.01, 0.003, 0.005

# Device Selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DKTplus custom loss function
def DKTplus_loss(model, y_pred, q_curr, q_next, r_curr, r_next, sequence_mask):

    # Prediction Loss Calcuation
    y_pred_for_q_next = torch.gather(y_pred, 2, q_next.unsqueeze(2)).squeeze(-1)
    prediction_loss = nn.functional.binary_cross_entropy_with_logits(
        torch.masked_select(y_pred_for_q_next, sequence_mask),
        torch.masked_select(r_next, sequence_mask).float(),
        reduction="mean",
    )

    # Consistency Loss Calculation
    y_pred_for_q_curr = torch.gather(y_pred, 2, q_curr.unsqueeze(2)).squeeze(-1)
    r_curr_float = r_curr.float()  # Check it out
    consistency_loss = nn.functional.binary_cross_entropy_with_logits(
        torch.masked_select(y_pred_for_q_curr, sequence_mask),
        torch.masked_select(r_curr_float, sequence_mask),
    )

    # Smoothness Loss Calculation (L! + L2)
    waviness = torch.norm(y_pred[:, 1:] - y_pred[:, :-1], p=2, dim=-1)  # (B, T-1)
    waviness_masked_L1 = torch.masked_select(waviness, sequence_mask[:, 1:])
    smoothness_loss_L1 = waviness_masked_L1.mean() / model.num_c

    waviness_masked_L2 = torch.masked_select(waviness**2, sequence_mask[:, 1:])
    smoothness_loss_L2 = waviness_masked_L2.mean() / model.num_c

    loss = (
        prediction_loss
        + model.lambda_r * consistency_loss
        + model.lambda_w1 * smoothness_loss_L1
        + model.lambda_w2 * smoothness_loss_L2
    )

    return loss


# Data loading
with open(PROCESSED_DATA_PATH, "r") as f:
    data = json.load(f)

q_sequences, r_sequences, t_sequences = (
    data["q_sequences"],
    data["r_sequences"],
    data["t_sequences"],
)

q_train, q_val, r_train, r_val, t_train, t_val = train_test_split(
    q_sequences, r_sequences, t_sequences, test_size=0.2, random_state=7
)

train_dataset = DKTplus_dataset(q_train, r_train, t_train)
val_dataset = DKTplus_dataset(q_val, r_val, t_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=DKTplus_collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=DKTplus_collate)


# Model initailization
model = DKTplus(NUM_C, EMB_SIZE, HIDDEN_SIZE, LAMBDA_R, LAMBDA_W1, LAMBDA_W2, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting Model Training")
# Training loop
for epochs in range(EPOCHS):

    model.train()
    train_preds, train_targets = [], []

    for batch in train_loader:
        if batch[0] == None:
            continue
        q, r, t, q_next, r_next, sequence_mask = [item.to(DEVICE) for item in batch]

        optimizer.zero_grad()

        y_pred = model(q, r, t)

        loss = DKTplus_loss(model, y_pred, q, q_next, r, r_next, sequence_mask)

        loss.backward()
        optimizer.step()

        pred_for_q_next = torch.gather(y_pred, 2, q_next.unsqueeze(2)).squeeze(-1)
        train_preds.append(torch.masked_select(pred_for_q_next, sequence_mask).detach().cpu().numpy())
        train_targets.append(torch.masked_select(r_next, sequence_mask).detach().cpu().numpy())

    train_preds = np.concatenate(train_preds)
    train_targets = np.concatenate(train_targets)

    train_probs = torch.sigmoid(torch.tensor(train_preds)).numpy()

    train_acc = accuracy_score(train_targets.astype(int), (train_probs > 0.5).astype(int))

    # Evaluation
    model.eval()
    val_preds, val_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            if batch[0] == None:
                continue
            q, r, t, q_next, r_next, sequence_mask = [item.to(DEVICE) for item in batch]

            y_pred = model(q, r, t)

            pred_for_q_next = torch.gather(y_pred, 2, q_next.unsqueeze(2)).squeeze(-1)
            val_preds.append(torch.masked_select(pred_for_q_next, sequence_mask).detach().cpu().numpy())
            val_targets.append(torch.masked_select(r_next, sequence_mask).detach().cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        val_probs = torch.sigmoid(torch.tensor(val_preds)).numpy()

        val_auc = roc_auc_score(val_targets.astype(int), val_probs)
        val_acc = accuracy_score(val_targets.astype(int), (val_probs > 0.5).astype(int))

        print(
            f"Epoch {epochs+1}/{EPOCHS} -> Train acc: {train_acc:.4f}, Validation acc: {val_acc:.4f}, Validation AUC: {val_auc:.4f}"
        )


# Saving the trained model
model.eval()

dummy_q = torch.randint(0, model.num_c + 1, (2, 5)).to(DEVICE)
dummy_r = torch.randint(0, 2, (2, 5)).to(DEVICE)
dummy_t = torch.rand(2, 5).to

traced_model = torch.jit.trace(model, (dummy_q, dummy_r, dummy_t))
traced_model.save(DKT_MODEL_PATH)
