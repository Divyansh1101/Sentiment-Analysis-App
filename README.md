# Streamlit Sentiment Analysis App

A full-featured Sentiment Analysis web application built with **Streamlit**.

## Features

- Dataset upload (CSV) with `review`, `sentiment`, `date` columns.
- Manual review text input.
- Synthetic data generation if no file is uploaded.
- Preprocessing: cleaning, tokenization, stopword removal, language detection.
- Sentiment classification using TF-IDF + Logistic Regression (or a lightweight fallback).
- Aspect-based sentiment detection (food, service, price) via keyword matching.
- Emotion detection via `text2emotion`.
- Visualizations: word clouds, sentiment distribution, time-series charts.
- Explainability: top contributing features per class (when model is trained).
- CSV and PDF report generation.
- Robust error handling and helpful messages.

## Quickstart (local)

1. Clone or download the project and install dependencies:

```bash
python -m venv venv
source venv/bin/activate    # on Windows use `venv\\Scripts\\activate`
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run streamlit_app.py
```

3. Upload a CSV with columns `review`, `sentiment`, `date` or use the synthetic dataset.

## Files

- `streamlit_app.py` — main Streamlit app.
- `requirements.txt` — Python dependencies.
- Example synthetic data is generated automatically by the app.

## Notes

- The app downloads required NLTK data the first time it runs.
- For production deployment (Streamlit Cloud), ensure you add a `packages.txt` or proper deployment config if necessary.

Enjoy! 🎉

## Improvements added

- GitHub Actions CI workflow to run basic tests (`.github/workflows/ci.yml`).
- Improved aspect extraction using noun-phrase chunking and VADER sentiment scoring for aspect-level sentiment.
- Updated requirements to include `vaderSentiment`.

