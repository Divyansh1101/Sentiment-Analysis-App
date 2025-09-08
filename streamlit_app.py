
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

# Additional imports for improved aspect sentiment
from nltk import pos_tag, word_tokenize, RegexpParser
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Helper utilities ---
st.set_page_config(layout="wide", page_title="Sentiment Analysis App", page_icon="📝")

@st.cache_data
def download_nltk():
    datasets = ["punkt","stopwords","wordnet","averaged_perceptron_tagger","vader_lexicon"]
    for d in datasets:
        try:
            nltk.data.find(d)
        except:
            try:
                nltk.download(d)
            except:
                pass
    return True

download_nltk()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))

def simple_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def tokenize_and_remove_stopwords(text, lang='en'):
    tokens = word_tokenize(text)
    if lang.startswith("en"):
        tokens = [t for t in tokens if t.lower() not in STOPWORDS and len(t)>1]
    else:
        tokens = [t for t in tokens if len(t)>1]
    return tokens

def generate_synthetic_data(n=200):
    reviews = []
    labels = []
    dates = []
    sample_pos = [
        "The food was amazing and the service was excellent.",
        "Loved the ambience and the friendly staff!",
        "Best meal I've had in weeks, will come again."
    ]
    sample_neg = [
        "Food was cold and tasteless, very disappointed.",
        "Service was slow and the waiter was rude.",
        "Will not recommend this place to anyone."
    ]
    sample_neu = [
        "The restaurant is located downtown. It opens at 10am.",
        "I visited yesterday. Prices are average.",
        "It was an OK experience, nothing special."
    ]
    for i in range(n):
        choice = np.random.choice([0,1,2], p=[0.45,0.35,0.20])
        if choice==0:
            text = np.random.choice(sample_pos)
            labels.append("Positive")
        elif choice==1:
            text = np.random.choice(sample_neg)
            labels.append("Negative")
        else:
            text = np.random.choice(sample_neu)
            labels.append("Neutral")
        reviews.append(text)
        dates.append((datetime.now() - pd.to_timedelta(np.random.randint(0,365), unit='d')).strftime("%Y-%m-%d"))
    return pd.DataFrame({"review":reviews, "sentiment":labels, "date":dates})

# Initialize VADER
vader = SentimentIntensityAnalyzer()

# --- UI ---
st.title("📝 Streamlit Sentiment Analysis — Improved")
st.markdown("This improved version adds: GitHub Actions CI, better aspect-extraction (noun-phrase chunking), and aspect-level sentiment with VADER.")

with st.sidebar:
    st.header("Settings & Controls")
    retrain = st.checkbox("Retrain model on uploaded data (if available)", value=True)
    test_size = st.slider("Test set size (%)", 10, 50, 20)
    random_state = st.number_input("Random state", 0, 9999, 42)
    proba_threshold = st.slider("Probability threshold for neutral", 0.1, 0.5, 0.25)
    st.markdown("**Filters**")
    filter_sentiment = st.multiselect("Filter by sentiment", options=["Positive","Negative","Neutral"], default=["Positive","Negative","Neutral"])
    keyword_filter = st.text_input("Keyword filter (comma separated)")
    date_from = st.date_input("From", value=None)
    date_to = st.date_input("To", value=None)

# File uploader & text input
uploaded_file = st.file_uploader("Upload CSV with columns: review, sentiment, date (optional)", type=["csv"])
manual_review = st.text_area("Or paste a single review here to analyze", height=120)
use_synthetic = st.checkbox("Use synthetic example dataset if no file is uploaded", value=True)

# Load dataset
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        df = None
else:
    df = None

if df is None and use_synthetic:
    st.info("Generating synthetic dataset (example) — you can replace it by uploading a CSV.")
    df = generate_synthetic_data(300)

# Validate required columns
if df is not None:
    if "review" not in df.columns:
        st.warning("Uploaded dataframe does not contain 'review' column. Trying best-effort to find text columns.")
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if len(text_cols)>0:
            df = df.rename(columns={text_cols[0]:"review"})
            st.info(f"Renamed column {text_cols[0]} to 'review'.")
        else:
            st.error("No textual column found. Switching to synthetic data.")
            df = generate_synthetic_data(300)

    if "sentiment" not in df.columns:
        st.info("No 'sentiment' column found — labels will be created from model predictions (unlabeled).")
        df["sentiment"] = None
    if "date" not in df.columns:
        df["date"] = pd.NaT

    # preview
    st.subheader("Dataset preview")
    st.dataframe(df.head(50))

# Preprocessing
if df is not None:
    st.sidebar.subheader("Preprocessing options")
    lowercase = st.sidebar.checkbox("Lowercase text", value=True)
    clean_text = st.sidebar.checkbox("Clean text (remove urls/punct)", value=True)
    detect_lang = st.sidebar.checkbox("Detect language", value=True)

    df["raw_review"] = df["review"].astype(str)
    def preprocess_row(x):
        txt = x
        if lowercase:
            txt = txt.lower()
        if clean_text:
            txt = simple_clean(txt)
        lang = detect_language(x) if detect_lang else "unknown"
        tokens = tokenize_and_remove_stopwords(txt, lang=lang)
        return " ".join(tokens), lang

    preproc = df["raw_review"].apply(preprocess_row)
    df["clean_review"] = preproc.apply(lambda x: x[0])
    df["lang"] = preproc.apply(lambda x: x[1])

    st.write("Preprocessing done. Language samples:")
    st.dataframe(df[["raw_review","clean_review","lang"]].head(50))

# Model training
model = None
vectorizer = None
pipeline = None
trainable = retrain and (df is not None) and (df["sentiment"].notna().sum()>0)

if trainable:
    labeled = df[df["sentiment"].notna()].copy()
    labeled = labeled[labeled["sentiment"].isin(["Positive","Negative","Neutral"])]
    if labeled.shape[0] < 20:
        st.warning("Too few labeled samples for reliable training. Consider uploading labeled data or using more synthetic data.")
    else:
        X = labeled["clean_review"].values
        y = labeled["sentiment"].values
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size/100.0, random_state=random_state)
        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=8000)
        model = LogisticRegression(max_iter=1000)
        pipeline = make_pipeline(vectorizer, model)
        pipeline.fit(X_train, y_train)
        ypred = pipeline.predict(X_test)
        yproba = pipeline.predict_proba(X_test)
        st.subheader("Model evaluation (on test split)")
        st.text(classification_report(y_test, ypred, digits=3))
        cm = confusion_matrix(y_test, ypred, labels=["Positive","Neutral","Negative"])
        fig, ax = plt.subplots(figsize=(4,3))
        ax.imshow(cm, interpolation='nearest')
        ax.set_title("Confusion matrix")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(["Positive","Neutral","Negative"], rotation=45)
        ax.set_yticklabels(["Positive","Neutral","Negative"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j,i,cm[i,j],ha="center",va="center", color="white")
        st.pyplot(fig)
else:
    st.info("Model training skipped (no labeled data or retrain disabled). A lightweight demo classifier will be used for predictions.")

# Prediction function (works even without training)
def predict_texts(texts):
    results = []
    if pipeline is not None:
        proba = pipeline.predict_proba(texts)
        preds = pipeline.predict(texts)
        for i,t in enumerate(texts):
            p = proba[i]
            pred = preds[i]
            conf = max(p)
            results.append({"review":t, "pred":pred, "proba":p, "conf":conf})
    else:
        # simple rule-based fallback using keywords
        pos_k = ["good","great","excellent","love","best","amazing","friendly","delicious"]
        neg_k = ["bad","worst","cold","slow","rude","disappoint","terrible"]
        for t in texts:
            lc = t.lower()
            score = 0
            for w in pos_k:
                if w in lc: score += 1
            for w in neg_k:
                if w in lc: score -= 1
            if abs(score) <= 0:
                pred = "Neutral"
                conf = 0.5
                proba = np.array([0.4,0.2,0.4])
            elif score>0:
                pred = "Positive"; conf = min(0.9, 0.5 + 0.1*score); proba = np.array([conf, 1-conf-conf*0.1, 1-conf])
            else:
                pred = "Negative"; conf = min(0.9, 0.5 + 0.1*(-score)); proba = np.array([1-conf, 1-conf*0.1, conf])
            results.append({"review":t, "pred":pred, "proba":proba, "conf":conf})
    return results

# Aspect extraction using noun-phrase chunking
grammar = r"""
    NP: {<JJ>*<NN.*>+}   # adjective(s) + noun(s)
        {<NN.*>+}        # noun(s)
"""
chunker = RegexpParser(grammar)
def extract_noun_chunks(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    tree = chunker.parse(tags)
    chunks = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        chunk = " ".join([w for w,pos in subtree.leaves()])
        chunks.append(chunk.lower())
    return list(dict.fromkeys(chunks))  # dedupe preserving order

def aspect_sentiment_vader(text, aspects):
    res = {}
    for a in aspects:
        if a in text:
            # get sentence(s) mentioning aspect
            sentences = re.split(r'[.?!;]', text)
            relevant = [s for s in sentences if a in s]
            joined = ". ".join(relevant) if relevant else text
            score = vader.polarity_scores(joined)
            comp = score['compound']
            if comp >= 0.05:
                res[a] = "Positive"
            elif comp <= -0.05:
                res[a] = "Negative"
            else:
                res[a] = "Neutral"
        else:
            res[a] = "Not mentioned"
    return res

# Single manual review analysis
st.header("Analyze a single review")
if manual_review.strip()!="":
    review_text = manual_review.strip()
    clean = simple_clean(review_text)
    lang = detect_language(review_text)
    tokens = tokenize_and_remove_stopwords(clean, lang)
    st.write("Detected language:", lang)
    st.write("Cleaned / tokenized text:", " ".join(tokens))
    emo = te.get_emotion(review_text)
    st.write("Detected emotions:", emo)
    pred = predict_texts([clean])[0]
    st.write("Predicted sentiment:", pred["pred"])
    st.write("Confidence:", pred["conf"])
    # extract aspects
    noun_chunks = extract_noun_chunks(review_text)
    st.write("Noun-phrase aspects detected (approx):", noun_chunks[:10])
    aspect_sent = aspect_sentiment_vader(review_text.lower(), noun_chunks[:10])
    st.write("Aspect-level sentiment (VADER):", aspect_sent)
    if pipeline is not None:
        # feature contributions
        try:
            vec = pipeline.named_steps['tfidfvectorizer']
            model_lr = pipeline.named_steps['logisticregression']
            x_vec = vec.transform([clean])
            coef = model_lr.coef_
            classes = model_lr.classes_
            top_feats = {}
            for idx,cls in enumerate(classes):
                coefs = coef[idx]
                topn = np.argsort(coefs)[-10:]
                features = [vec.get_feature_names_out()[i] for i in topn][::-1]
                top_feats[cls] = features
            st.write("Top contributing features per class (approx):")
            st.write(top_feats)
        except Exception as e:
            st.info("Explainability limited: " + str(e))
    # emotion wordcloud
    wc = WordCloud(width=400,height=200).generate(clean)
    fig,ax = plt.subplots(figsize=(6,3))
    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
    st.pyplot(fig)

# Bulk predictions for dataframe
st.header("Bulk predictions & exploration")
if df is not None:
    # apply predictions
    df["pred_sentiment"] = None
    preds_all = predict_texts(df["clean_review"].astype(str).tolist())
    df["pred_sentiment"] = [p["pred"] for p in preds_all]
    df["pred_confidence"] = [p["conf"] for p in preds_all]

    # apply filters
    df_filtered = df.copy()
    if filter_sentiment:
        df_filtered = df_filtered[df_filtered["pred_sentiment"].isin(filter_sentiment)]
    if keyword_filter.strip()!="":
        kws = [k.strip().lower() for k in keyword_filter.split(",") if k.strip()!=""]
        df_filtered = df_filtered[df_filtered["clean_review"].str.contains("|".join(kws))]
    if date_from is not None:
        try:
            df_filtered = df_filtered[pd.to_datetime(df_filtered["date"]) >= pd.to_datetime(date_from)]
        except:
            pass
    if date_to is not None:
        try:
            df_filtered = df_filtered[pd.to_datetime(df_filtered["date"]) <= pd.to_datetime(date_to)]
        except:
            pass

    st.subheader("Filtered dataset preview")
    st.dataframe(df_filtered.head(200))

    # Sentiment distribution
    st.subheader("Sentiment distribution")
    dist = df_filtered["pred_sentiment"].value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0)
    fig,ax = plt.subplots()
    dist.plot(kind="bar", ax=ax)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Wordclouds per sentiment
    st.subheader("Wordclouds per sentiment")
    cols = st.columns(3)
    for i, s in enumerate(["Positive","Negative","Neutral"]):
        with cols[i]:
            texts = " ".join(df_filtered[df_filtered["pred_sentiment"]==s]["clean_review"].astype(str).tolist())
            if len(texts.strip())==0:
                st.write(s, " — no data")
                continue
            wc = WordCloud(width=400,height=300).generate(texts)
            fig,ax = plt.subplots(figsize=(4,3))
            ax.imshow(wc); ax.axis("off")
            st.pyplot(fig)

    # Time series (counts by date)
    st.subheader("Time series of sentiments")
    try:
        ts = df_filtered.copy()
        ts["date2"] = pd.to_datetime(ts["date"], errors="coerce")
        ts = ts.dropna(subset=["date2"])
        ts_group = ts.groupby([pd.Grouper(key="date2", freq="W"), "pred_sentiment"]).size().unstack(fill_value=0)
        fig,ax = plt.subplots(figsize=(8,3))
        ts_group.plot(ax=ax)
        ax.set_ylabel("Counts per week")
        st.pyplot(fig)
    except Exception as e:
        st.info("Could not create time-series: " + str(e))

    # Improved Aspect-based sentiment: extract noun-chunks and use VADER on sentences mentioning aspect
    st.subheader("Aspect-based sentiment (noun-phrases + VADER)")
    def detect_aspects_row(row):
        text = row["raw_review"]
        chunks = extract_noun_chunks(text)
        aspect_scores = aspect_sentiment_vader(text.lower(), chunks[:8])
        return aspect_scores
    aspect_df = df_filtered.copy()
    aspect_df["aspect_sentiments"] = aspect_df.apply(detect_aspects_row, axis=1)
    st.write("Sample aspect detection (first 30 rows)")
    st.json(aspect_df[["raw_review","aspect_sentiments"]].head(30).to_dict(orient="records"))

    # Emotion analysis (text2emotion)
    st.subheader("Emotion detection (joy, anger, sadness, fear, surprise)")
    emo_sample = aspect_df["raw_review"].astype(str).apply(lambda x: te.get_emotion(x))
    st.dataframe(pd.DataFrame(emo_sample.tolist()).sum().to_frame("counts"))

    # Explainability: show top features if model present
    st.subheader("Explainability & top features")
    if pipeline is not None:
        try:
            vec = pipeline.named_steps['tfidfvectorizer']
            model_lr = pipeline.named_steps['logisticregression']
            feature_names = vec.get_feature_names_out()
            coefs = model_lr.coef_
            classes = model_lr.classes_
            explanation = {}
            for idx,cls in enumerate(classes):
                topn = np.argsort(coefs[idx])[-20:][::-1]
                explanation[cls] = [feature_names[i] for i in topn]
            st.write(explanation)
        except Exception as e:
            st.info("Could not extract features: " + str(e))
    else:
        st.info("Train a model to enable detailed explainability.")

    # Feedback on predictions
    st.subheader("Feedback on model predictions")
    if st.button("Mark first filtered row as corrected (simulate feedback)"):
        st.success("Feedback recorded (demo). In a production app you would append this to training data and retrain.")

    # Generate reports (CSV + PDF)
    st.subheader("Save & export")
    if st.button("Download filtered CSV"):
        to_download = df_filtered.copy()
        csv = to_download.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_report.csv">Download CSV file</a>'
        st.markdown(href, unsafe_allow_html=True)

    if st.button("Download PDF report"):
        try:
            buffer = io.BytesIO()
            with PdfPages(buffer) as pdf:
                fig1, ax1 = plt.subplots()
                dist.plot(kind="bar", ax=ax1); ax1.set_title("Sentiment distribution")
                pdf.savefig(fig1); plt.close(fig1)

                if 'ts_group' in locals():
                    fig2, ax2 = plt.subplots(figsize=(8,3))
                    ts_group.plot(ax=ax2); ax2.set_title("Time series (weekly)")
                    pdf.savefig(fig2); plt.close(fig2)

                fig3, ax3 = plt.subplots(figsize=(6,3))
                ax3.text(0.01, 0.5, f"Total reviews: {len(df_filtered)}\\nPositive: {int(dist.get('Positive',0))}\\nNeutral: {int(dist.get('Neutral',0))}\\nNegative: {int(dist.get('Negative',0))}", fontsize=12)
                ax3.axis('off')
                pdf.savefig(fig3); plt.close(fig3)

            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="sentiment_report.pdf">Download PDF report</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error("Could not generate PDF: " + str(e))
