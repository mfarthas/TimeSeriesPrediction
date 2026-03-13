import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = "./synthetic"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

F = 8
HIDDEN1 = 64
HIDDEN2 = 32
DROPOUT = 0.3
BATCH_SIZE = 512
MAX_EPOCHS = 100
LR = 1e-3
PATIENCE = 7
VAL_SPLIT = 0.1


class IncidentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(F, HIDDEN1, batch_first=True)
        self.drop1 = nn.Dropout(DROPOUT)
        self.lstm2 = nn.LSTM(HIDDEN1, HIDDEN2, batch_first=True)
        self.drop2 = nn.Dropout(DROPOUT)
        self.output = nn.Linear(HIDDEN2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out)
        return self.output(out[:, -1, :]).squeeze(1)


def load():
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")
    return X_train, y_train, X_test, y_test


def normalize(X_train, X_test):
    N_tr, W, Fc = X_train.shape
    N_te = X_test.shape[0]
    scaler = StandardScaler()
    X_train = (
        scaler.fit_transform(X_train.reshape(-1, Fc))
        .reshape(N_tr, W, Fc)
        .astype(np.float32)
    )
    X_test = (
        scaler.transform(X_test.reshape(-1, Fc))
        .reshape(N_te, W, Fc)
        .astype(np.float32)
    )
    return X_train, X_test


def make_dataloaders(X_train, y_train):
    N = len(X_train)
    val_size = int(N * VAL_SPLIT)
    idx = np.random.permutation(N)
    val_idx, tr_idx = idx[:val_size], idx[val_size:]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train[tr_idx]),
            torch.tensor(y_train[tr_idx].astype(np.float32)),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train[val_idx]),
            torch.tensor(y_train[val_idx].astype(np.float32)),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, val_loader


def train(model, train_loader, val_loader, pos_weight):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )

    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                val_loss += criterion(model(X_b), y_b).item() * len(X_b)
        val_loss /= len(val_loader.dataset)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:03d}/{MAX_EPOCHS}"
            f"  train={train_loss:.4f}"
            f"  val={val_loss:.4f}"
            f"  lr={lr:.2e}"
            f"  ({time.time() - t0:.1f}s)"
        )
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def evaluate(model, X_test, y_test):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_test)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    logits = np.concatenate([
        model(X.to(DEVICE)).cpu().detach().numpy() for (X,) in loader
    ])
    y_prob = 1 / (1 + np.exp(-logits))
    y_pred = (y_prob >= 0.5).astype(int)

    print(classification_report(
        y_test, y_pred, target_names=["normal", "incident"]
    ))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"Confusion matrix:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"PR-AUC:   {average_precision_score(y_test, y_prob):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")

    np.save(f"{DATA_DIR}/y_prob_lstm.npy", y_prob)
    torch.save(model.state_dict(), f"{DATA_DIR}/lstm.pt")


def main():
    print(f"Device: {DEVICE}")
    X_train, y_train, X_test, y_test = load()
    X_train, X_test = normalize(X_train, X_test)

    train_loader, val_loader = make_dataloaders(X_train, y_train)
    pos_weight = torch.tensor(
        (y_train == 0).sum() / (y_train == 1).sum(), dtype=torch.float32
    )
    print(f"pos_weight: {pos_weight.item():.2f}")

    model = IncidentLSTM().to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train(model, train_loader, val_loader, pos_weight)
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
