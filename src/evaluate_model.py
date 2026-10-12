"""Evaluate the saved HireSafe model on the same held-out split used in training."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MODEL_PATH, VECTORIZER_PATH
from train_model import load_cleaned_dataset, split_features_and_labels


def load_saved_artifacts():
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Saved model files were not found. Train the model first:\n"
            "  python src/train_model.py"
        )
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


def evaluate() -> dict[str, float]:
    frame = load_cleaned_dataset()
    _, x_test, _, y_test = split_features_and_labels(frame)
    vectorizer, model = load_saved_artifacts()

    # Transform the test split with the already-fitted training vectorizer.
    x_test_tfidf = vectorizer.transform(x_test.astype(str))
    y_pred = model.predict(x_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    print("HireSafe evaluation (held-out test set)")
    print("-" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print()
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(matrix)
    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate (0)", "Fraudulent (1)"],
            zero_division=0,
        )
    )
    print(
        "Fraud-detection note: accuracy can look high because most postings are legitimate. "
        "Precision is the share of predicted fakes that are actually fake "
        "(false alarms hurt when a real job is blocked). "
        "Recall is the share of actual fakes that the model catches "
        "(missed scams are the cost of a false negative)."
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "test_rows": int(len(y_test)),
        "fraudulent_in_test": int((y_test == 1).sum()),
    }


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)
