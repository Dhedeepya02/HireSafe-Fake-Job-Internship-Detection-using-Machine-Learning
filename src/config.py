"""Shared project paths and training constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "job_postings.csv"
CLEANED_DATA_PATH = DATA_DIR / "processed" / "cleaned_jobs.csv"

MODELS_DIR = PROJECT_ROOT / "models"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODELS_DIR / "fake_job_classifier.pkl"

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]
TARGET_COLUMN = "fraudulent"

TEST_SIZE = 0.20
RANDOM_STATE = 42

# TF-IDF is fit only on the training split to avoid data leakage.
# NOTE: text_length is computed during preprocessing but not used as a TF-IDF feature.
# It is saved in cleaned_jobs.csv for potential future use alongside TF-IDF features.
TFIDF_PARAMS = {
    "max_features": 8000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.90,
}

# Public mirrors of the EMSCAD / Kaggle fake job posting dataset.
DATASET_URLS = [
    "https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction/resolve/main/fake_job_postings.csv",
    "https://huggingface.co/datasets/haja17/real-or-fake-fake-jobposting-prediction/resolve/main/fake_job_postings.csv",
]
