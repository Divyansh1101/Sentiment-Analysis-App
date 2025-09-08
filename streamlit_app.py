
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud
import nltk, re, os, io
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
import text2emotion as te
from langdetect import detect
import base64
import shap

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(layout="wide", page_title="Sentiment Analysis App", page_icon="📝")
nltk_downloaded = False
def download_nltk():
    global nltk_downloaded
    if nltk_downloaded:
        return
    try:
        nltk.data.find("tokenizers/punkt")
    except:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except:
        nltk.download("wordnet")
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except:
        nltk.download("averaged_perceptron_tagger")
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except:
        nltk.download("vader_lexicon")
    nltk_downloaded = True

download_nltk()
STOPWORDS = set(stopwords.words("english"))
vader = SentimentIntensityAnalyzer()

def simple_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_remove_stopwords(text, lang='en'):
    tokens = word_tokenize(text)
    if lang.startswith("en"):
        tokens = [t for t in tokens if t.lower() not in STOPWORDS and len(t)>1]
    else:
        tokens = [t for t in tokens if len(t)>1]
    return tokens

def generate_synthetic_data(n=200):
    reviews=[]
    labels=[]
    dates=[]
    sample_pos=["The food was amazing and the service was excellent.","Loved the ambience and the friendly staff!","Best meal I've had in weeks, will come again."]
    sample_neg=["Food was cold and tasteless, very disappointed.","Service was slow and the waiter was rude.","Will not recommend this place to anyone."]
    sample_neu=["The restaurant is located downtown. It opens at 10am.","I visited yesterday. Prices are average.","It was an OK experience, nothing special."]
    for i in range(n):
        choice = np.random.choice([0,1,2], p=[0.45,0.35,0.20])
        if choice==0:
            text=np.random.choice(sample_pos); labels.append("Positive")
        elif choice==1:
            text=np.random.choice(sample_neg); labels.append("Negative")
        else:
            text=np.random.choice(sample_neu); labels.append("Neutral")
        reviews.append(text)
        dates.append((datetime.now() - pd.to_timedelta(np.random.randint(0,365), unit='d')).strftime("%Y-%m-%d"))
    return pd.DataFrame({"review":reviews,"sentiment":labels,"date":dates})

# minimal UI to allow tests and functions to be imported
st.title("Sentiment Analysis App (Stable Build)")
st.sidebar.header("Settings")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
manual_review = st.text_area("Manual review")

# helpers for explainability and bootstrap
from sklearn.utils import resample
def compute_shap_for_text(pipeline, X_train_sample, text, top_k=15):
    try:
        vec = pipeline.named_steps['tfidfvectorizer']
        lr = pipeline.named_steps['logisticregression']
    except:
        return {}
    try:
        Xb = vec.transform(X_train_sample[:200])
        explainer = shap.LinearExplainer(lr, Xb, feature_dependence="independent")
        x_vec = vec.transform([text])
        shap_values = explainer.shap_values(x_vec)
        feature_names = vec.get_feature_names_out()
        res = {}
        for cls_idx, cls in enumerate(lr.classes_):
            sv = shap_values[cls_idx].ravel()
            top_idx = np.argsort(np.abs(sv))[-top_k:][::-1]
            res[cls] = [(feature_names[i], float(sv[i])) for i in top_idx]
        return res
    except Exception:
        return {}

def bootstrap_proba_ci(pipeline_factory, X_train, y_train, text, n_iter=30, alpha=0.05):
    probs=[]
    for i in range(n_iter):
        try:
            Xb, yb = resample(X_train, y_train)
            p = pipeline_factory()
            p.fit(Xb, yb)
            probs.append(p.predict_proba([text])[0])
        except:
            continue
    if len(probs)==0:
        return None,None,None
    probs = np.array(probs)
    lower = np.percentile(probs, 100*alpha/2.0, axis=0)
    upper = np.percentile(probs, 100*(1-alpha/2.0), axis=0)
    mean = probs.mean(axis=0)
    return mean, lower, upper

# expose functions for tests
__all__ = ["simple_clean","tokenize_and_remove_stopwords","generate_synthetic_data","compute_shap_for_text","bootstrap_proba_ci"]
