"""Streamlit UI for HireSafe fake job and internship detection."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict import predict_posting  # noqa: E402


st.set_page_config(page_title="HireSafe", page_icon="🛡️", layout="centered")

st.title("HireSafe")
st.subheader("Fake Job & Internship Detection")
st.write(
    "Paste a job or internship posting and HireSafe will classify it as likely "
    "legitimate or potentially fraudulent using TF-IDF features and a linear SVM."
)

with st.expander("⚙️ Detection sensitivity", expanded=False):
    threshold = st.slider(
        "Decision threshold",
        min_value=-1.0,
        max_value=1.0,
        value=-0.3,
        step=0.05,
        help=(
            "Lower = catches more fake jobs (fewer missed frauds, more false alarms). "
            "Higher = only flags obvious fakes (fewer false alarms, more missed frauds). "
            "Default -0.3 is recommended for fraud detection."
        ),
    )

with st.form("job_form"):
    title = st.text_input("Job title")
    company_profile = st.text_area("Company information")
    description = st.text_area("Job description", height=180)
    requirements = st.text_area("Requirements")
    benefits = st.text_area("Benefits")
    submitted = st.form_submit_button("Detect Job")

if submitted:
    try:
        with st.spinner("Analyzing job posting..."):
            result = predict_posting(
                title=title,
                company_profile=company_profile,
                description=description,
                requirements=requirements,
                benefits=benefits,
                threshold=threshold,
            )
        if result["prediction"] == 1:
            st.error(result["ui_label"])
        else:
            st.success(result["ui_label"])

        st.caption("Prediction generated using TF-IDF features and an SVM classifier.")
        st.info(
            "This score is the model's signed distance from the decision boundary, "
            "not a calibrated probability. HireSafe can miss sophisticated scams and "
            "can flag unusual but real postings. Use it as a screening aid, not as a "
            "guarantee."
        )
        st.write(
            f"SVM decision score: `{result['decision_score']:.4f}`  \n"
            f"SVM boundary distance (not a probability): `{result['boundary_distance_pct']:.2%}`"
        )
    except FileNotFoundError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.warning(str(exc))
        st.info(
            "💡 **Tip**: For accurate fraud detection, provide detailed information including:\n"
            "- Full job title (e.g., Senior Software Engineer)\n"
            "- Company background/profile\n"
            "- Detailed job description with responsibilities\n"
            "- Specific requirements and skills needed\n"
            "- Benefits and compensation details\n\n"
            "The model analyzes word patterns and context to identify suspicious job postings. "
            "Use real job-related terms like: job, work, position, company, experience, skills, "
            "responsibilities, salary, benefits, etc."
        )
    except Exception as exc:
        st.error(f"Could not generate a prediction: {exc}")
