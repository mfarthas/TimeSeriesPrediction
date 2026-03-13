import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


T = 100_000
F = 8
W = 100
H = 50
DRIFT_LEAD = W
INCIDENT_LEN = 30
MIN_GAP = 300
SEED = 42
TARGET_PR_AUC = 0.9     
OUTPUT_DIR = "./synthetic"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = [
    (0.22, 0.26, 0.18),
    (0.24, 0.26, 0.15),
    (0.25, 0.27, 0.12),
    (0.26, 0.28, 0.10),
    (0.27, 0.29, 0.08),
    (0.28, 0.30, 0.06),
    (0.29, 0.31, 0.05),
    (0.30, 0.32, 0.04),
]


def sample_incidents(rng):
    times = []
    t = DRIFT_LEAD
    while t < T - INCIDENT_LEN:
        if rng.random() < 0.6:
            times.append(t)
            t += INCIDENT_LEN + DRIFT_LEAD + MIN_GAP
        else:
            t += MIN_GAP // 4
    return times


def generate_signal(incident_times, rng, normal_noise, incident_noise,
                    normal_shared):
    signal = np.zeros((T, F), dtype=np.float32)
    labels = np.zeros(T, dtype=np.int8)

    for t_inc in incident_times:
        labels[t_inc:t_inc + INCIDENT_LEN] = 1

    for t in range(T):
        if labels[t] == 1:
            shared = 0.0
            noise_scale = incident_noise
        else:
            in_precursor = False
            progress = 0.0
            for t_inc in incident_times:
                if t_inc - DRIFT_LEAD <= t < t_inc:
                    progress = (t - (t_inc - DRIFT_LEAD)) / DRIFT_LEAD
                    in_precursor = True
                    break
            if in_precursor:
                shared = float(rng.normal(0, normal_shared * (1 - progress)))
                noise_scale = (
                    normal_noise + (incident_noise - normal_noise) * progress
                )
            else:
                shared = float(rng.normal(0, normal_shared))
                noise_scale = normal_noise

        signal[t] = (
            shared + rng.normal(0, noise_scale, size=F).astype(np.float32)
        )

    return signal, labels


def make_windows(signal, labels):
    n = signal.shape[0] - W - H + 1
    X = np.zeros((n, W, F), dtype=np.float32)
    y = np.zeros(n, dtype=np.int8)
    for i in range(n):
        X[i] = signal[i:i + W]
        y[i] = 1 if labels[i + W:i + W + H].any() else 0
    return X, y


def normalize(X_train, X_test):
    N_tr, W_, Fc = X_train.shape
    N_te = X_test.shape[0]
    scaler = StandardScaler()
    X_train = (
        scaler.fit_transform(X_train.reshape(-1, Fc))
        .reshape(N_tr, W_, Fc)
        .astype(np.float32)
    )
    X_test = (
        scaler.transform(X_test.reshape(-1, Fc))
        .reshape(N_te, W_, Fc)
        .astype(np.float32)
    )
    return X_train, X_test


class IncidentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(F, 64, batch_first=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.drop2 = nn.Dropout(0.3)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out)
        return self.output(out[:, -1, :]).squeeze(1)


def train_lstm(X_train, y_train):
    N = len(X_train)
    val_size = int(N * 0.1)
    idx = np.random.permutation(N)
    val_idx, tr_idx = idx[:val_size], idx[val_size:]

    pos_weight = torch.tensor(
        (y_train == 0).sum() / (y_train == 1).sum(), dtype=torch.float32
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train[tr_idx]),
            torch.tensor(y_train[tr_idx].astype(np.float32)),
        ),
        batch_size=512,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train[val_idx]),
            torch.tensor(y_train[val_idx].astype(np.float32)),
        ),
        batch_size=512,
        shuffle=False,
    )

    model = IncidentLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )

    best_val = float("inf")
    best_state = None
    patience = 0

    for _ in range(50):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                val_loss += criterion(model(X_b), y_b).item() * len(X_b)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= 7:
                break

    model.load_state_dict(best_state)
    return model


def infer(model, X_test):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_test)),
        batch_size=512,
        shuffle=False,
    )
    logits = np.concatenate([
        model(X.to(DEVICE)).cpu().detach().numpy() for (X,) in loader
    ])
    return 1 / (1 + np.exp(-logits))


def save(X_tr_s, y_train, X_te_s, y_test, y_prob, model,
         incident_times, normal_noise, incident_noise, normal_shared):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_tr_s)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_te_s)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)
    np.save(f"{OUTPUT_DIR}/y_prob_lstm.npy", y_prob)
    np.save(f"{OUTPUT_DIR}/y_true.npy", y_test)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/lstm.pt")
    meta = {
        "T": T, "F": F, "W": W, "H": H,
        "normal_noise": normal_noise,
        "incident_noise": incident_noise,
        "normal_shared": normal_shared,
        "drift_lead": DRIFT_LEAD,
        "incident_len": INCIDENT_LEN,
        "n_incidents": len(incident_times),
    }
    with open(f"{OUTPUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    print(f"Device: {DEVICE}")
    print(f"Searching for first config with PR-AUC < {TARGET_PR_AUC}\n")

    for normal_noise, incident_noise, normal_shared in CONFIGS:
        rng = np.random.default_rng(SEED)
        incident_times = sample_incidents(rng)
        signal, labels = generate_signal(
            incident_times, rng, normal_noise, incident_noise, normal_shared
        )

        split = int(T * 0.7)
        X_train, y_train = make_windows(signal[:split], labels[:split])
        X_test, y_test = make_windows(signal[split:], labels[split:])
        X_train, X_test = normalize(X_train, X_test)

        model = train_lstm(X_train, y_train)
        y_prob = infer(model, X_test)
        pr_auc = average_precision_score(y_test, y_prob)

        print(
            f"normal_noise={normal_noise}"
            f"  incident_noise={incident_noise}"
            f"  shared={normal_shared}"
            f"  pos={100 * y_train.mean():.1f}%"
            f"  PR-AUC={pr_auc:.4f}"
        )

        if pr_auc < TARGET_PR_AUC:
            print("\n  --> Target reached. Saving.")
            save(
                X_train, y_train, X_test, y_test, y_prob, model,
                incident_times, normal_noise, incident_noise, normal_shared,
            )
            break
    else:
        print("\nNo config reached target. Extend CONFIGS with harder values.")


if __name__ == "__main__":
    main()
