# Sentiment Analysis App - Streamlit Ready

This repository contains a Streamlit Sentiment Analysis application designed to be stable and deployable to Streamlit Cloud.

See `streamlit_app.py` for the main app.

## Deploying to Streamlit Cloud

1. Push to GitHub.
2. On https://share.streamlit.io create a new app, connect repo and set main file to `streamlit_app.py`.
3. Streamlit will install from `requirements.txt`.

## Docker

Build and run locally:

```bash
docker build -t sentiment-app:latest .
docker run -p 8501:8501 sentiment-app:latest
```
