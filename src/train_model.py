"""Train a Linear SVM on TF-IDF features. TF-IDF is fit on training data only."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    CLEANED_DATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TFIDF_PARAMS,
    VECTORIZER_PATH,
)
from preprocessing import run_preprocessing


def load_cleaned_dataset() -> pd.DataFrame:
    """Load cleaned_jobs.csv, creating it first if needed."""
    if not CLEANED_DATA_PATH.exists():
        print("Cleaned dataset not found. Running preprocessing first...")
        return run_preprocessing()

    frame = pd.read_csv(CLEANED_DATA_PATH)
    if "cleaned_text" not in frame.columns or TARGET_COLUMN not in frame.columns:
        raise ValueError("cleaned_jobs.csv is missing cleaned_text or fraudulent columns.")
    frame = frame.dropna(subset=["cleaned_text", TARGET_COLUMN])
    return frame


def split_features_and_labels(frame: pd.DataFrame):
    """Create a stratified train/test split before vectorization."""
    x = frame["cleaned_text"].astype(str)
    y = frame[TARGET_COLUMN].astype(int)
    return train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(**TFIDF_PARAMS)


def train_linear_svm(x_train, y_train) -> LinearSVC:
    """LinearSVC is a linear SVM that scales well to sparse TF-IDF matrices."""
    classifier = LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000,
    )
    classifier.fit(x_train, y_train)
    return classifier


def save_artifacts(vectorizer: TfidfVectorizer, model: LinearSVC) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved classifier to {MODEL_PATH}")


def train() -> None:
    frame = load_cleaned_dataset()
    x_train, x_test, y_train, y_test = split_features_and_labels(frame)

    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print("Fitting TF-IDF on the training split only...")

    vectorizer = build_vectorizer()
    # NOTE: text_length is computed in preprocessing and saved to cleaned_jobs.csv
    # but is intentionally not used here. TF-IDF operates on the cleaned_text column only.
    # text_length is preserved in the CSV for potential future use as an additional feature.
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    print("Training Linear SVM...")
    model = train_linear_svm(x_train_tfidf, y_train)
    save_artifacts(vectorizer, model)

    train_accuracy = model.score(x_train_tfidf, y_train)
    test_accuracy = model.score(x_test_tfidf, y_test)
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Held-out accuracy: {test_accuracy:.4f}")

    # Print class distribution in training data
    counts = y_train.value_counts().sort_index()
    total = len(y_train)
    legit_count = counts.get(0, 0)
    fraud_count = counts.get(1, 0)
    print("\nClass distribution in training data:")
    print(f"  Legitimate (0): {legit_count} ({legit_count / total:.2%})")
    print(f"  Fraudulent  (1): {fraud_count} ({fraud_count / total:.2%})")

    # Print full classification report on the held-out test set
    print("\nClassification Report (held-out test set):")
    print(
        classification_report(
            y_test,
            model.predict(x_test_tfidf),
            target_names=["Legitimate", "Fraudulent"],
        )
    )
    print("Run evaluate_model.py to also view the confusion matrix.")


if __name__ == "__main__":
    try:
        train()
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)
