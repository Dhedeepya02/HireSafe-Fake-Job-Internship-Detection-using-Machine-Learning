"""Text cleaning and dataset preparation for HireSafe."""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    CLEANED_DATA_PATH,
    DATASET_URLS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEXT_COLUMNS,
)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

_NLTK_READY = False
_LEMMATIZER = None
_STOPWORDS = None


def ensure_nltk_resources() -> None:
    """Download required NLTK resources once, then cache local tools."""
    global _NLTK_READY, _LEMMATIZER, _STOPWORDS

    if _NLTK_READY:
        return

    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)

    _LEMMATIZER = WordNetLemmatizer()
    _STOPWORDS = set(stopwords.words("english"))
    _NLTK_READY = True


def clean_text(text: object) -> str:
    """Lowercase, strip URLs/HTML/punctuation, tokenize, drop stopwords, lemmatize."""
    ensure_nltk_resources()
    from nltk.tokenize import word_tokenize

    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    cleaned = str(text).lower()
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = HTML_PATTERN.sub(" ", cleaned)
    cleaned = NON_ALPHANUMERIC_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    if not cleaned:
        return ""

    tokens = word_tokenize(cleaned)
    lemmatized = [
        _LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in _STOPWORDS and len(token) > 1
    ]
    return " ".join(lemmatized)


def combine_text_fields(row: pd.Series) -> str:
    """Join the useful posting fields into a single raw text string."""
    parts = []
    for column in TEXT_COLUMNS:
        value = row.get(column, "")
        if pd.notna(value) and str(value).strip():
            parts.append(str(value))
    return " ".join(parts)


def validate_dataset(frame: pd.DataFrame) -> None:
    """Confirm the dataset has the expected target column and values."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"The dataset must contain a '{TARGET_COLUMN}' column "
            "(0 = legitimate, 1 = fraudulent)."
        )

    unique_labels = set(pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").dropna().unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Column '{TARGET_COLUMN}' must contain only 0 (legitimate) and 1 (fraudulent)."
        )


def download_public_dataset(destination: Path) -> bool:
    """Try known public mirrors of the EMSCAD fake-job dataset."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    for url in DATASET_URLS:
        try:
            print(f"Trying dataset download: {url}")
            urllib.request.urlretrieve(url, destination)
            if destination.exists() and destination.stat().st_size > 10_000:
                print(f"Downloaded dataset to {destination}")
                return True
            if destination.exists():
                destination.unlink()
        except Exception as exc:
            print(f"Download failed for {url}: {exc}")
            if destination.exists():
                destination.unlink()
    return False


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load job_postings.csv, attempting a public download if the file is missing."""
    data_path = path or RAW_DATA_PATH
    if not data_path.exists():
        downloaded = download_public_dataset(data_path)
        if not downloaded:
            raise FileNotFoundError(
                "Could not find or download a real job-posting dataset.\n"
                f"Place a CSV with a '{TARGET_COLUMN}' column at:\n"
                f"  {RAW_DATA_PATH}\n\n"
                "Recommended public dataset (EMSCAD / Kaggle):\n"
                "  https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction\n"
                "Download fake_job_postings.csv, rename it to job_postings.csv, and put it in data/raw/."
            )

    frame = pd.read_csv(data_path)
    validate_dataset(frame)
    return frame


def preprocess_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, combine text fields, and create cleaned_text."""
    processed = frame.copy()
    processed[TARGET_COLUMN] = pd.to_numeric(processed[TARGET_COLUMN], errors="coerce")
    processed = processed.dropna(subset=[TARGET_COLUMN])
    processed[TARGET_COLUMN] = processed[TARGET_COLUMN].astype(int)

    for column in TEXT_COLUMNS:
        if column not in processed.columns:
            processed[column] = ""
        processed[column] = processed[column].fillna("")

    processed["combined_text"] = processed.apply(combine_text_fields, axis=1)
    processed["cleaned_text"] = processed["combined_text"].apply(clean_text)
    processed = processed[processed["cleaned_text"].str.strip().astype(bool)].copy()
    processed["text_length"] = processed["cleaned_text"].str.split().str.len()
    return processed


def save_cleaned_dataset(frame: pd.DataFrame, path: Path | None = None) -> Path:
    """Save the processed dataset used by training and evaluation."""
    output_path = path or CLEANED_DATA_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keep_columns = [column for column in TEXT_COLUMNS if column in frame.columns]
    export_columns = keep_columns + [
        TARGET_COLUMN,
        "combined_text",
        "cleaned_text",
        "text_length",
    ]
    frame[export_columns].to_csv(output_path, index=False)
    return output_path


def run_preprocessing() -> pd.DataFrame:
    """End-to-end cleaning pipeline used by the training scripts and notebook."""
    raw = load_raw_dataset()
    cleaned = preprocess_dataframe(raw)
    output_path = save_cleaned_dataset(cleaned)
    print(f"Saved cleaned dataset to {output_path}")
    print(f"Rows after cleaning: {len(cleaned)}")
    print(cleaned[TARGET_COLUMN].value_counts().rename({0: "legitimate", 1: "fraudulent"}))
    return cleaned


if __name__ == "__main__":
    try:
        run_preprocessing()
    except Exception as exc:
        print(f"Preprocessing failed: {exc}", file=sys.stderr)
        sys.exit(1)
