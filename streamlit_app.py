import streamlit as st
st.set_page_config(
    layout="wide", 
    page_title="Sentiment Analysis App", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import datetime
import nltk
import re
import text2emotion as te
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# NLTK dependencies
@st.cache_resource
def download_nltk_resources():
    for r in ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger', 'vader_lexicon']:
        nltk.download(r, quiet=True)
download_nltk_resources()

@st.cache_resource
def nlp_objects():
    return SentimentIntensityAnalyzer(), set(stopwords.words("english"))
vader, STOPWORDS = nlp_objects()

### Helper functions

def simple_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def analyze_sentiment_vader(text):
    try:
        return vader.polarity_scores(text)
    except Exception:
        return {'pos':0,'neu':1,'neg':0,'compound':0}

def analyze_emotion_text2emotion(text):
    try:
        return te.get_emotion(text)
    except Exception:
        return {}

def get_sentiment_label(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def create_sample_csv():
    df = pd.DataFrame({
        "review": [
            "I love this product!", "Absolutely terrible experience.", "It was okay, nothing special.",
            "Great value for the price.", "Not impressed with the quality.", "Customer service was excellent.",
            "Shipping took too long.", "Item arrived damaged.", "Very happy with my purchase!",
            "I would not recommend this.", "Works as expected.", "Disappointed overall.", "Better than I hoped.",
            "Faulty item received.", "Five stars!", "Will buy again.", "Too expensive.", "Good but not great.",
            "Poor customer support.", "Quick delivery!", "Amazing experience.", "Lacks some features.",
            "Packaging was nice.", "Terrible taste.", "Happy with the results."
        ],
        "date": pd.date_range("2024-09-01", periods=25).strftime("%Y-%m-%d"),
        "sentiment": [
            "Positive", "Negative", "Neutral", "Positive", "Negative", "Positive",
            "Negative", "Negative", "Positive", "Negative", "Positive", "Negative", "Positive",
            "Negative", "Positive", "Positive", "Negative", "Neutral", "Negative",
            "Positive", "Positive", "Neutral", "Positive", "Negative", "Positive"
        ]
    })
    return df

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig

def plot_roc(y_true, y_prob, class_names):
    fig, ax = plt.subplots(figsize=(8,5))
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    for i, label in enumerate(class_names):
        if len(np.unique(y_true_bin[:, i])) < 2: continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_score = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC={auc_score:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.legend(); ax.set_title('ROC Curves')
    plt.tight_layout()
    return fig

def generate_pdf(results_df, metrics_str):
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis('off')
        tbl = ax.table(cellText=results_df.values, colLabels=results_df.columns,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.2, 1.2)
        ax.set_title("Sentiment Analysis Results", fontweight='bold', pad=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        # Metrics
        fig, ax = plt.subplots(figsize=(8.5, 4))
        ax.axis('off')
        ax.text(0.01, 0.98, "Statistical Summary", fontsize=14, fontweight="bold", va="top")
        for i, line in enumerate(metrics_str.splitlines()):
            ax.text(0.01, 0.95-0.075*i, line, fontsize=12, va="top")
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    return pdf_buffer.getvalue()

## MAIN APP

def main():
    st.title("📊 Sentiment Analysis App with Statistical Reporting")
    st.sidebar.header("⚙️ Settings")
    analysis_mode = st.sidebar.radio("Choose Analysis Mode:", ["📁 File Upload", "✏️ Manual Text Input"])

    # Sample CSV download
    st.sidebar.markdown("---")
    st.sidebar.markdown("📂 Download a sample CSV for testing:")
    sample_df = create_sample_csv()
    st.sidebar.download_button(
        label="⬇️ Download Sample CSV",
        data=sample_df.to_csv(index=False).encode(),
        file_name="sample_reviews.csv",
        mime="text/csv"
    )

    if analysis_mode == "📁 File Upload":
        st.header("📁 File Upload Analysis")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
            st.dataframe(df.head(10), use_container_width=True)
            text_cols = df.select_dtypes(include="object").columns.tolist()
            if text_cols:
                selected_column = st.selectbox("Select text column for analysis", text_cols)
                gt_col = st.selectbox(
                    "Select ground truth (optional, for stats)", ["None"] + text_cols
                )
                if st.button("🚀 Run Analysis", type="primary"):
                    preds, probs, compounds, emotions = [], [], [], []
                    for text in df[selected_column].astype(str):
                        text_clean = simple_clean(text)
                        vs = analyze_sentiment_vader(text_clean)
                        label = get_sentiment_label(vs['compound'])
                        preds.append(label)
                        compounds.append(vs['compound'])
                        pos, neu, neg = vs['pos'], vs['neu'], vs['neg']
                        tot = pos + neu + neg + 1e-8
                        probs.append([neg/tot, neu/tot, pos/tot])
                        emotions.append(analyze_emotion_text2emotion(text_clean))

                    results = df.copy()
                    results["Predicted_Sentiment"] = preds
                    results["VADER_Compound"] = compounds

                    st.subheader("📋 Analysis Results")
                    st.dataframe(results.head(25), use_container_width=True)

                    metrics_str = ""
                    # If ground truth exists, do evaluation
                    if gt_col != "None":
                        y_true = [x.capitalize() for x in df[gt_col].astype(str)]
                        y_pred = preds
                        # Ensure both have common mapping
                        classes = sorted(list(set(y_true) | set(y_pred)))
                        le = LabelEncoder().fit(classes)
                        yt = le.transform(y_true)
                        yp = le.transform(y_pred)
                        ypr = np.array(probs)
                        # Metrics
                        accuracy = accuracy_score(yt, yp)
                        precision = precision_score(yt, yp, average="macro", zero_division=0)
                        recall = recall_score(yt, yp, average="macro", zero_division=0)
                        f1 = f1_score(yt, yp, average="macro", zero_division=0)
                        kappa = cohen_kappa_score(yt, yp)
                        metrics_str=(
                            f"Accuracy: {accuracy:.3f}\n"
                            f"Precision (macro): {precision:.3f}\n"
                            f"Recall (macro): {recall:.3f}\n"
                            f"F1 (macro): {f1:.3f}\n"
                            f"Cohen's Kappa: {kappa:.3f}\n"
                        )
                        st.info(metrics_str)
                        # Confusion matrix
                        st.pyplot(plot_confusion_matrix(yt, yp, le.classes_))
                        # ROC-AUC
                        st.pyplot(plot_roc(yt, ypr, le.classes_))
                        # PDF download with metrics
                        pdf_bytes = generate_pdf(results, metrics_str)
                        st.download_button(
                            "📊 Download PDF Report",
                            data=pdf_bytes,
                            file_name="sentiment_report.pdf",
                            mime="application/pdf"
                        )

                    # CSV download, always available
                    st.download_button(
                        "📄 Download CSV Results",
                        data=results.to_csv(index=False).encode(),
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )

    else:
        st.header("✏️ Manual Text Input")
        manual_text = st.text_area("Enter your text here:")
        if st.button("🔍 Analyze Text"):
            v = analyze_sentiment_vader(simple_clean(manual_text))
            e = analyze_emotion_text2emotion(simple_clean(manual_text))
            st.write("**Sentiment Scores:**")
            st.json(v)
            st.write("**Detected Emotion:**")
            st.json(e)

    st.markdown("---")
    st.caption("Built using Python 3.10 | Streamlit")

if __name__ == "__main__":
    main()
