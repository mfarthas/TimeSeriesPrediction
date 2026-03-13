import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


DATA_DIR = "./synthetic"


def load():
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")
    return X_train, y_train, X_test, y_test


def flatten(X):
    N, W, F = X.shape
    return X.reshape(N, W * F)


def main():
    X_train, y_train, X_test, y_test = load()

    X_train_flat = flatten(X_train)
    X_test_flat = flatten(X_test)

    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
    )
    print("Training ...")
    model.fit(X_train_flat, y_train)

    y_pred = model.predict(X_test_flat)
    y_prob = model.predict_proba(X_test_flat)[:, 1]

    print(classification_report(
        y_test, y_pred, target_names=["normal", "incident"]
    ))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"Confusion matrix:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"PR-AUC:   {average_precision_score(y_test, y_prob):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")

    np.save(f"{DATA_DIR}/y_prob_baseline.npy", y_prob)
    np.save(f"{DATA_DIR}/y_true.npy", y_test)


if __name__ == "__main__":
    main()
