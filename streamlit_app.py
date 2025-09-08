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
import traceback

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="Sentiment Analysis App", 
    page_icon="📝",                # Correct Memo emoji!
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# NLTK downloads
nltk_downloaded = False
def download_nltk():
    global nltk_downloaded
    if nltk_downloaded:
        return
    try:
        nltk.data.find("tokenizers/punkt")
    except:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except:
        nltk.download("averaged_perceptron_tagger", quiet=True)
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except:
        nltk.download("vader_lexicon", quiet=True)
    nltk_downloaded = True

# Download required NLTK data
with st.spinner("Initializing NLP components..."):
    download_nltk()

STOPWORDS = set(stopwords.words("english"))
vader = SentimentIntensityAnalyzer()

def simple_clean(text):
    """Clean text data"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_remove_stopwords(text, lang='en'):
    """Tokenize and remove stopwords"""
    tokens = word_tokenize(text)
    if lang.startswith("en"):
        tokens = [t for t in tokens if t.lower() not in STOPWORDS and len(t) > 1]
    else:
        tokens = [t for t in tokens if len(t) > 1]
    return tokens

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    try:
        scores = vader.polarity_scores(text)
        return scores
    except Exception as e:
        st.error(f"Error in VADER analysis: {e}")
        return None

def analyze_emotion_text2emotion(text):
    """Analyze emotions using text2emotion"""
    try:
        emotions = te.get_emotion(text)
        return emotions
    except Exception as e:
        st.error(f"Error in emotion analysis: {e}")
        return None

def process_csv_file(uploaded_file):
    """Process uploaded CSV file"""
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)

        # Display basic info about the dataset
        st.subheader("📋 Dataset Overview")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Data Types", len(df.dtypes.unique()))

        # Show first few rows
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Let user select text column for analysis
        text_columns = df.select_dtypes(include=['object']).columns.tolist()

        if text_columns:
            st.subheader("🎯 Select Text Column for Analysis")
            selected_column = st.selectbox(
                "Choose the column containing text data:", 
                text_columns,
                help="Select the column that contains the text you want to analyze"
            )

            if st.button("🚀 Start Sentiment Analysis", type="primary"):
                with st.spinner("Analyzing sentiments... This may take a moment."):
                    results = perform_sentiment_analysis(df, selected_column)
                    st.session_state.processed_data = df
                    st.session_state.analysis_results = results

        else:
            st.error("❌ No text columns found in the dataset. Please ensure your CSV contains text data.")

    except Exception as e:
        st.error(f"❌ Error processing CSV file: {e}")
        st.error("Please make sure your file is a valid CSV format.")
        st.code(traceback.format_exc())

def perform_sentiment_analysis(df, text_column):
    """Perform comprehensive sentiment analysis"""
    results = {
        'vader_scores': [],
        'emotions': [],
        'sentiments': [],
        'processed_texts': []
    }

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_rows = len(df)

    for i, text in enumerate(df[text_column].astype(str)):
        # Update progress
        progress = (i + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing {i+1}/{total_rows} texts...")

        # Clean text
        cleaned_text = simple_clean(text)
        results['processed_texts'].append(cleaned_text)

        # VADER sentiment analysis
        vader_scores = analyze_sentiment_vader(cleaned_text)
        results['vader_scores'].append(vader_scores)

        # Determine overall sentiment
        if vader_scores:
            compound = vader_scores['compound']
            if compound >= 0.05:
                sentiment = 'Positive'
            elif compound <= -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
        else:
            sentiment = 'Unknown'
        results['sentiments'].append(sentiment)

        # Emotion analysis
        emotions = analyze_emotion_text2emotion(cleaned_text)
        results['emotions'].append(emotions)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    return results

def display_analysis_results(df, results):
    """Display comprehensive analysis results"""
    st.success("✅ Analysis completed successfully!")

    # Create results dataframe
    results_df = df.copy()
    results_df['Processed_Text'] = results['processed_texts']
    results_df['Sentiment'] = results['sentiments']

    # Add VADER scores
    if results['vader_scores'] and results['vader_scores'][0] is not None:
        results_df['VADER_Compound'] = [score['compound'] if score else 0 for score in results['vader_scores']]
        results_df['VADER_Positive'] = [score['pos'] if score else 0 for score in results['vader_scores']]
        results_df['VADER_Negative'] = [score['neg'] if score else 0 for score in results['vader_scores']]
        results_df['VADER_Neutral'] = [score['neu'] if score else 0 for score in results['vader_scores']]

    # Add emotion scores
    if results['emotions'] and results['emotions'][0] is not None:
        emotion_keys = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
        for key in emotion_keys:
            results_df[f'Emotion_{key}'] = [emotions.get(key, 0) if emotions else 0 for emotions in results['emotions']]

    # Display results
    st.subheader("📈 Analysis Results")
    st.dataframe(results_df, use_container_width=True)

    # Sentiment distribution
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = pd.Series(results['sentiments']).value_counts()

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'red', 'gray'])
        ax.set_title('Sentiment Distribution')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        # Pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['green', 'red', 'gray'])
        ax.set_title('Sentiment Distribution (%)')
        ax.set_ylabel('')
        st.pyplot(fig)

    # Summary statistics
    st.subheader("📊 Summary Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Positive Texts", sentiment_counts.get('Positive', 0))
    with col2:
        st.metric("Negative Texts", sentiment_counts.get('Negative', 0))
    with col3:
        st.metric("Neutral Texts", sentiment_counts.get('Neutral', 0))

    # Download processed data
    st.subheader("📥 Download Results")
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Analysis Results (CSV)",
        data=csv,
        file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def analyze_manual_text(text):
    """Analyze manually entered text"""
    if not text.strip():
        st.warning("Please enter some text to analyze.")
        return

    with st.spinner("Analyzing your text..."):
        # Clean text
        cleaned_text = simple_clean(text)

        # VADER analysis
        vader_scores = analyze_sentiment_vader(cleaned_text)

        # Emotion analysis
        emotions = analyze_emotion_text2emotion(cleaned_text)

        # Display results
        st.subheader("🔬 Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**VADER Sentiment Scores:**")
            if vader_scores:
                st.json(vader_scores)

                # Overall sentiment
                compound = vader_scores['compound']
                if compound >= 0.05:
                    sentiment = '😊 Positive'
                    st.success(f"Overall Sentiment: {sentiment}")
                elif compound <= -0.05:
                    sentiment = '😞 Negative'
                    st.error(f"Overall Sentiment: {sentiment}")
                else:
                    sentiment = '😐 Neutral'
                    st.info(f"Overall Sentiment: {sentiment}")

        with col2:
            st.write("**Emotion Analysis:**")
            if emotions:
                st.json(emotions)

                # Dominant emotion
                if emotions:
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                    st.info(f"Dominant Emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.2f})")

# Main app interface
def main():
    st.title("📝 Sentiment Analysis App (Stable Build)")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Mode:",
        ["📤 File Upload", "✏️ Manual Text Input"],
        help="Select how you want to input text for analysis"
    )

    if analysis_mode == "📤 File Upload":
        st.header("📤 File Upload Analysis")
        st.markdown("Upload a CSV file containing text data for sentiment analysis.")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file with text data. Maximum file size: 200MB"
        )

        if uploaded_file is not None:
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

            # File details
            file_details = {
                "Filename": uploaded_file.name,
                "File Size": f"{uploaded_file.size} bytes",
                "File Type": uploaded_file.type
            }
            st.json(file_details)

            # Process the file
            process_csv_file(uploaded_file)

            # Display results if analysis was performed
            if st.session_state.analysis_results is not None and st.session_state.processed_data is not None:
                display_analysis_results(st.session_state.processed_data, st.session_state.analysis_results)

    else:  # Manual text input
        st.header("✏️ Manual Text Analysis")
        st.markdown("Enter text manually for quick sentiment analysis.")

        manual_text = st.text_area(
            "Enter your text here:",
            height=150,
            placeholder="Type or paste your text here for sentiment analysis...",
            help="Enter any text you want to analyze for sentiment and emotions"
        )

        if st.button("🔬 Analyze Text", type="primary"):
            if manual_text.strip():
                analyze_manual_text(manual_text)
            else:
                st.warning("Please enter some text to analyze.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>💡 <strong>Tip:</strong> For best results, ensure your CSV file has clear column headers and text data.</p>
            <p>Built with ❤️ using Streamlit | Powered by VADER & Text2Emotion</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 
