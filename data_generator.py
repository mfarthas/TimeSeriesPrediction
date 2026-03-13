import argparse
import json
import os

import numpy as np


DEFAULTS = {
    "T": 100_000,
    "F": 8,
    "W": 100,
    "H": 50,
    "normal_noise": 0.24,
    "incident_noise": 0.26,
    "normal_shared": 0.15,
}

DRIFT_LEAD = DEFAULTS["W"]
INCIDENT_LEN = 30
MIN_GAP = 300
SEED = 42
OUTPUT_DIR = "./synthetic"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--T", type=int, default=DEFAULTS["T"])
    parser.add_argument("--F", type=int, default=DEFAULTS["F"])
    parser.add_argument("--W", type=int, default=DEFAULTS["W"])
    parser.add_argument("--H", type=int, default=DEFAULTS["H"])
    parser.add_argument("--normal-noise", type=float,
                        default=DEFAULTS["normal_noise"])
    parser.add_argument("--incident-noise", type=float,
                        default=DEFAULTS["incident_noise"])
    parser.add_argument("--normal-shared", type=float,
                        default=DEFAULTS["normal_shared"])
    return parser.parse_args()


def sample_incidents(rng, T, drift_lead, incident_len, min_gap):
    times = []
    t = drift_lead
    while t < T - incident_len:
        if rng.random() < 0.6:
            times.append(t)
            t += incident_len + drift_lead + min_gap
        else:
            t += min_gap // 4
    return times


def generate_signal(incident_times, rng, T, F, normal_noise, incident_noise,
                    normal_shared, drift_lead, incident_len):
    signal = np.zeros((T, F), dtype=np.float32)
    labels = np.zeros(T, dtype=np.int8)

    for t_inc in incident_times:
        labels[t_inc:t_inc + incident_len] = 1

    for t in range(T):
        if labels[t] == 1:
            shared = 0.0
            noise_scale = incident_noise
        else:
            in_precursor = False
            progress = 0.0
            for t_inc in incident_times:
                if t_inc - drift_lead <= t < t_inc:
                    progress = (t - (t_inc - drift_lead)) / drift_lead
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


def make_windows(signal, labels, W, H, F):
    n = signal.shape[0] - W - H + 1
    X = np.zeros((n, W, F), dtype=np.float32)
    y = np.zeros(n, dtype=np.int8)
    for i in range(n):
        X[i] = signal[i:i + W]
        y[i] = 1 if labels[i + W:i + W + H].any() else 0
    return X, y


def main():
    args = parse_args()
    T = args.T
    F = args.F
    W = args.W
    H = args.H
    normal_noise = args.normal_noise
    incident_noise = args.incident_noise
    normal_shared = args.normal_shared
    drift_lead = W

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    incident_times = sample_incidents(
        rng, T, drift_lead, INCIDENT_LEN, MIN_GAP
    )
    signal, labels = generate_signal(
        incident_times, rng, T, F, normal_noise, incident_noise,
        normal_shared, drift_lead, INCIDENT_LEN,
    )

    split = int(T * 0.7)
    X_train, y_train = make_windows(signal[:split], labels[:split], W, H, F)
    X_test, y_test = make_windows(signal[split:], labels[split:], W, H, F)

    print(f"Incidents: {len(incident_times)}")
    print(
        f"Train: {X_train.shape}  "
        f"positives: {y_train.sum()} ({100 * y_train.mean():.1f}%)"
    )
    print(
        f"Test:  {X_test.shape}  "
        f"positives: {y_test.sum()} ({100 * y_test.mean():.1f}%)"
    )

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    meta = {
        "T": T, "F": F, "W": W, "H": H,
        "normal_noise": normal_noise,
        "incident_noise": incident_noise,
        "normal_shared": normal_shared,
        "drift_lead": drift_lead,
        "incident_len": INCIDENT_LEN,
        "n_incidents": len(incident_times),
        "incident_times": incident_times,
    }
    with open(f"{OUTPUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
