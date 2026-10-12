"""Score a new job posting with the saved TF-IDF vectorizer and SVM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MODEL_PATH, VECTORIZER_PATH
from preprocessing import clean_text, combine_text_fields


def load_saved_artifacts():
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Saved model files were not found. Train the model first:\n"
            "  python src/train_model.py"
        )
    return joblib.load(VECTORIZER_PATH), joblib.load(MODEL_PATH)


def posting_to_cleaned_text(
    title: str = "",
    company_profile: str = "",
    description: str = "",
    requirements: str = "",
    benefits: str = "",
) -> str:
    row = {
        "title": title,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
    }
    return clean_text(combine_text_fields(row))


def score_to_boundary_distance(decision_score: float) -> float:
    """Map |decision_function| into a 0-1 display value representing normalized SVM
    margin distance. This is NOT a calibrated probability."""
    return abs(decision_score) / (1.0 + abs(decision_score))


def predict_posting(
    title: str = "",
    company_profile: str = "",
    description: str = "",
    requirements: str = "",
    benefits: str = "",
    threshold: float = 0.0,
) -> dict:
    cleaned_text = posting_to_cleaned_text(
        title=title,
        company_profile=company_profile,
        description=description,
        requirements=requirements,
        benefits=benefits,
    )
    if not cleaned_text:
        raise ValueError("Please provide at least one non-empty posting field.")
    
    # Warn if text is too short for reliable prediction
    word_count = len(cleaned_text.split())
    if word_count < 10:
        raise ValueError(
            f"Input text is too short ({word_count} words). "
            "Please provide more detailed job information (title, company, description, requirements, benefits) "
            "for accurate classification. The model needs sufficient context to identify patterns."
        )
    
    # Check if input contains meaningful words (not just random characters)
    # After NLTK cleaning, valid words should be alphabetic and not just single letters
    meaningful_words = [w for w in cleaned_text.split() if len(w) > 2]
    if len(meaningful_words) < 8:
        raise ValueError(
            f"Input does not contain enough meaningful words ({len(meaningful_words)} meaningful words found). "
            "Please provide actual job posting content with real words describing the position, company, "
            "requirements, and benefits. Random characters or gibberish cannot be classified."
        )
    
    # Check for common job-related vocabulary to ensure content is relevant
    job_related_keywords = {
        'job', 'work', 'position', 'role', 'company', 'team', 'skill', 'experience',
        'required', 'responsibility', 'salary', 'benefit', 'remote', 'full', 'part',
        'engineer', 'developer', 'manager', 'analyst', 'specialist', 'coordinator',
        'software', 'data', 'marketing', 'sales', 'support', 'service', 'product',
        'year', 'degree', 'education', 'python', 'java', 'javascript', 'sql', 'cloud',
        'manage', 'develop', 'design', 'create', 'build', 'lead', 'support'
    }
    
    words_lower = set(word.lower() for word in cleaned_text.split())
    job_keywords_found = len(words_lower.intersection(job_related_keywords))
    
    if job_keywords_found < 3:
        raise ValueError(
            f"Input does not appear to be a job posting (found only {job_keywords_found} job-related keywords). "
            "Please provide actual job posting content with terms like: job, work, position, company, "
            "experience, skills, responsibilities, salary, benefits, etc."
        )

    vectorizer, model = load_saved_artifacts()
    features = vectorizer.transform([cleaned_text])
    decision_score = float(model.decision_function(features)[0])
    prediction = 1 if decision_score >= threshold else 0

    label = "Potential Fake Job" if prediction == 1 else "Legitimate Job"
    ui_label = "Potentially Fraudulent" if prediction == 1 else "Likely Legitimate"
    return {
        "prediction": prediction,
        "label": label,
        "ui_label": ui_label,
        "decision_score": decision_score,
        "boundary_distance_pct": score_to_boundary_distance(decision_score),
        "threshold_used": threshold,
        "cleaned_text": cleaned_text,
    }


def print_prediction(result: dict) -> None:
    print(result["label"])
    print(f"SVM decision score: {result['decision_score']:.4f}")
    print(f"SVM boundary distance: {result['boundary_distance_pct']:.2%}")
    print(
        "The SVM decision score is the signed distance from the decision boundary. "
        "It is not a calibrated probability."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether a job posting looks fraudulent.")
    parser.add_argument("--title", default="")
    parser.add_argument("--company", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--requirements", default="")
    parser.add_argument("--benefits", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        result = predict_posting(
            title=args.title,
            company_profile=args.company,
            description=args.description,
            requirements=args.requirements,
            benefits=args.benefits,
        )
        print_prediction(result)
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        sys.exit(1)
