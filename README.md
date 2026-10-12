# HireSafe – Fake Job & Internship Detection using Machine Learning

HireSafe is an NLP system that classifies a job or internship posting as **legitimate** or **potentially fraudulent**. It combines NLTK text cleaning, TF-IDF features, and a linear Support Vector Machine (SVM).

The model is a screening aid. It does **not** claim 100% accuracy and should not be the only check before applying for a role or publishing a posting.

## Features

- Combine title, company profile, description, requirements, and benefits into one text field
- NLTK preprocessing: lowercase, URL/HTML cleanup, tokenization, stopword removal, and lemmatization
- Stratified train/test split **before** TF-IDF fitting (no leakage from the test set)
- Linear SVM (`LinearSVC`) with class weighting for the rare fraudulent class
- Metrics beyond accuracy: precision, recall, F1-score, confusion matrix, and classification report
- Command-line prediction and a Streamlit screening UI

## Tech stack

- Python
- Pandas, NumPy
- NLTK
- Scikit-learn (TF-IDF, LinearSVC)
- Joblib
- Streamlit
- Matplotlib, Seaborn (exploratory plots in the notebook)

## Architecture

```text
Job Posting
      ↓
Data Cleaning
      ↓
NLTK Text Preprocessing
      ↓
Combine Text Fields
      ↓
Train/Test Split
      ↓
TF-IDF Feature Extraction  (fit on training data only)
      ↓
SVM Classification
      ↓
Model Evaluation
      ↓
Save Model
      ↓
New Job Posting
      ↓
Prediction
      ↓
Legitimate / Potentially Fraudulent
```

## Machine learning workflow

1. Load a labeled CSV whose target column is `fraudulent` (`0` = legitimate, `1` = fraudulent).
2. Fill missing text fields, concatenate useful columns, and create `cleaned_text`.
3. Split into train and test sets with stratification so both splits keep the same fraud rate.
4. Fit `TfidfVectorizer` on **training** documents only, then transform train and test.
5. Train `LinearSVC` with `class_weight="balanced"`.
6. Evaluate on the held-out test split and save the vectorizer and classifier with Joblib.
7. At inference time, apply the **same** cleaning functions and the saved artifacts. The model is not retrained by the app.

## Project structure

```text
HireSafe/
├── data/
│   ├── raw/
│   │   └── job_postings.csv
│   └── processed/
│       └── cleaned_jobs.csv
├── notebooks/
│   └── HireSafe_Model_Development.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── fake_job_classifier.pkl
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset information

HireSafe expects `data/raw/job_postings.csv` with a `fraudulent` column.

Recommended public dataset: **Employment Scam Aegean Dataset (EMSCAD)**, distributed on Kaggle as [Real or Fake: Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).

Typical columns include `title`, `company_profile`, `description`, `requirements`, `benefits`, and `fraudulent`.

If `job_postings.csv` is missing, preprocessing will try a few public mirrors. If those fail, download the Kaggle file yourself, rename it to `job_postings.csv`, and place it in `data/raw/`. Do not invent labels.

## Installation

From the project root (`HireSafe/`):

```bash
py -3.14 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## NLTK setup

The first preprocessing run downloads these NLTK resources if they are not already present:

- `punkt` / `punkt_tab`
- `stopwords`
- `wordnet`
- `omw-1.4`

You can also download them once in a Python shell:

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## Model training

```bash
python src/preprocessing.py
python src/train_model.py
```

`train_model.py` will run preprocessing automatically if `data/processed/cleaned_jobs.csv` does not exist.

Saved files:

- `models/tfidf_vectorizer.pkl`
- `models/fake_job_classifier.pkl`

## Evaluation

```bash
python src/evaluate_model.py
```

Accuracy alone is a weak fraud metric because most postings are legitimate. **Precision** answers: of the postings flagged as fake, how many really were fake? **Recall** answers: of all actual fake postings, how many did the model catch?

Results from a local run on a held-out stratified test set of 3,576 postings (173 fraudulent):

| Metric | Value (fraudulent class) |
| --- | --- |
| Accuracy | 0.9860 |
| Precision | 0.8555 |
| Recall | 0.8555 |
| F1-score | 0.8555 |

Confusion matrix: 3,378 true negatives, 25 false positives, 25 false negatives, 148 true positives. These numbers come from `evaluate_model.py`; they are not claimed to generalize to every future scam.

## Streamlit application

```bash
streamlit run app.py
```

Enter a job title, company information, description, requirements, and benefits, then click **Detect Job**. The UI shows **Likely Legitimate** or **Potentially Fraudulent**, plus an SVM decision score. That score is **not** a calibrated probability.

## Example prediction

```bash
python src/predict.py --title "Software Engineer Intern" --company "Established product company with a careers page and named team" --description "Paid internship working with the backend team on Python APIs. Apply through the company careers portal." --requirements "Python, Git, currently enrolled in a CS program" --benefits "Stipend, mentorship, certificate"
```

A second check with a posting that asks for an upfront fee, a personal bank transfer, or WhatsApp-only hiring is useful for seeing the opposite class.

## Limitations

- Language and scams change; a model trained on historical EMSCAD-style ads will miss new patterns.
- SVM decision scores are not probabilities.
- Very short postings give the model little text to work with.
- The dataset is imbalanced; a high accuracy number can hide missed fraud.
- This tool is not legal advice, background screening, or a guarantee that a role is safe.

## Future improvements

- Calibrated probabilities (for example Platt scaling or a logistic model)
- Additional models (logistic regression, gradient boosting) and a comparison table
- Character n-grams and metadata features (missing company logo, salary range, telecommuting)
- Threshold tuning to trade precision vs recall
- Multilingual cleaning and non-English postings
- Human-in-the-loop review for high-risk scores
