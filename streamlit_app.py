import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud
import nltk
import re
import os
import io
from io import BytesIO
from datetime import datetime
import traceback
import warnings
import base64

warnings.filterwarnings('ignore')

# Advanced Statistical Analysis Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    cohen_kappa_score, brier_score_loss
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.calibration import calibration_curve
from scipy import stats
import text2emotion as te
from langdetect import detect
import shap

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# NLTK downloads with caching
@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger', 'vader_lexicon']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

download_nltk_resources()

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="Advanced Sentiment Analysis App", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'statistical_results' not in st.session_state:
    st.session_state.statistical_results = None

# Initialize VADER and stopwords
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except:
        return set()

vader = load_vader()
STOPWORDS = load_stopwords()

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
    try:
        tokens = word_tokenize(text)
        if lang.startswith("en") and STOPWORDS:
            tokens = [t for t in tokens if t.lower() not in STOPWORDS and len(t) > 1]
        else:
            tokens = [t for t in tokens if len(t) > 1]
        return tokens
    except:
        return text.split()

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    try:
        scores = vader.polarity_scores(text)
        return scores
    except Exception as e:
        return {'pos': 0, 'neu': 1, 'neg': 0, 'compound': 0}

def analyze_emotion_text2emotion(text):
    """Analyze emotions using text2emotion"""
    try:
        emotions = te.get_emotion(text)
        return emotions
    except Exception:
        return {}

def get_sentiment_label(compound_score):
    """Convert VADER compound score to sentiment label"""
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def create_sample_csv():
    """Create sample CSV data for testing"""
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

# Statistical Analysis Functions
def calculate_classification_metrics(y_true, y_pred):
    """Calculate comprehensive classification metrics"""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Generate and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    return fig, cm

def plot_roc_curves(y_true, y_proba, class_names):
    """Plot ROC curves for multi-class classification"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    try:
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        auc_scores = {}
        
        for i in range(len(class_names)):
            if len(np.unique(y_true_bin[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                auc_score = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                auc_scores[class_names[i]] = auc_score
                ax.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc_score:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - Multi-Class Classification')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig, auc_scores
    except Exception as e:
        st.error(f"Error plotting ROC curves: {e}")
        return None, {}

def perform_cross_validation(X, y, cv_folds=5):
    """Perform cross-validation with statistical analysis"""
    try:
        pipeline = make_pipeline(TfidfVectorizer(max_features=1000), LogisticRegression(random_state=42))
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')
        
        results = {
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'cv_scores': cv_scores,
            'confidence_interval_95': (
                np.mean(cv_scores) - 1.96 * np.std(cv_scores) / np.sqrt(cv_folds),
                np.mean(cv_scores) + 1.96 * np.std(cv_scores) / np.sqrt(cv_folds)
            )
        }
        return results
    except Exception as e:
        st.error(f"Error in cross-validation: {e}")
        return None

def comprehensive_statistical_analysis(y_true, y_pred, y_proba, class_names, texts=None):
    """Perform comprehensive statistical analysis"""
    results = {}
    
    try:
        # Basic metrics
        results['metrics'] = calculate_classification_metrics(y_true, y_pred)
        
        # Confusion matrix
        fig_cm, cm = plot_confusion_matrix(y_true, y_pred, class_names)
        results['confusion_matrix'] = {'figure': fig_cm, 'matrix': cm}
        
        # ROC curves
        if len(class_names) >= 2:
            fig_roc, auc_scores = plot_roc_curves(y_true, y_proba, class_names)
            if fig_roc:
                results['roc_analysis'] = {'figure': fig_roc, 'auc_scores': auc_scores}
        
        # Cross-validation
        if texts and len(texts) >= 10:
            cv_results = perform_cross_validation(texts[:len(y_true)], y_true)
            if cv_results:
                results['cross_validation'] = cv_results
        
        # Distribution analysis
        max_proba = np.max(y_proba, axis=1)
        results['distribution'] = {
            'confidence_mean': np.mean(max_proba),
            'confidence_std': np.std(max_proba),
            'error_rate': np.mean(y_pred != y_true)
        }
        
    except Exception as e:
        st.error(f"Error in statistical analysis: {e}")
        results['error'] = str(e)
    
    return results

def generate_comprehensive_pdf(results_df, statistical_results=None, sentiment_counts=None):
    """Generate comprehensive PDF report"""
    pdf_buffer = BytesIO()
    
    try:
        with PdfPages(pdf_buffer) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            ax.text(0.5, 0.95, 'Sentiment Analysis Report', 
                    fontsize=24, fontweight='bold', ha='center', va='top')
            ax.text(0.5, 0.88, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                    fontsize=12, ha='center', va='top')
            
            # Dataset overview
            ax.text(0.1, 0.75, 'Dataset Overview:', fontsize=16, fontweight='bold')
            ax.text(0.1, 0.70, f'• Total Records: {len(results_df)}', fontsize=12)
            ax.text(0.1, 0.66, f'• Columns: {len(results_df.columns)}', fontsize=12)
            
            # Sentiment distribution
            if sentiment_counts is not None:
                ax.text(0.1, 0.58, 'Sentiment Distribution:', fontsize=16, fontweight='bold')
                y_pos = 0.53
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / len(results_df)) * 100
                    ax.text(0.1, y_pos, f'• {sentiment}: {count} ({percentage:.1f}%)', fontsize=12)
                    y_pos -= 0.04
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Data table
            if len(results_df) > 0:
                display_df = results_df.head(20).copy()
                
                # Truncate long text for better display
                for col in display_df.select_dtypes(include=['object']).columns:
                    display_df[col] = display_df[col].astype(str).apply(
                        lambda x: x[:50] + '...' if len(str(x)) > 50 else str(x)
                    )
                
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                
                table = ax.table(
                    cellText=display_df.values,
                    colLabels=display_df.columns,
                    loc='center',
                    cellLoc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.2, 1.5)
                
                ax.set_title('Sentiment Analysis Results (First 20 Records)', 
                            fontsize=16, fontweight='bold', pad=20)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Statistical summary
            if statistical_results and 'metrics' in statistical_results:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                
                ax.text(0.5, 0.95, 'Statistical Analysis Summary', 
                       fontsize=20, fontweight='bold', ha='center')
                
                metrics = statistical_results['metrics']
                y_pos = 0.85
                
                ax.text(0.1, y_pos, 'Classification Metrics:', fontsize=16, fontweight='bold')
                y_pos -= 0.05
                
                for metric, value in metrics.items():
                    ax.text(0.15, y_pos, f'{metric.replace("_", " ").title()}: {value:.4f}', fontsize=12)
                    y_pos -= 0.04
                
                # Cross-validation results
                if 'cross_validation' in statistical_results:
                    y_pos -= 0.02
                    cv_info = statistical_results['cross_validation']
                    ax.text(0.1, y_pos, 'Cross-Validation Results:', fontsize=16, fontweight='bold')
                    y_pos -= 0.05
                    ax.text(0.15, y_pos, f'Mean Accuracy: {cv_info["mean_accuracy"]:.4f} ± {cv_info["std_accuracy"]:.4f}', fontsize=12)
                    y_pos -= 0.04
                    ci = cv_info["confidence_interval_95"]
                    ax.text(0.15, y_pos, f'95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]', fontsize=12)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Add statistical plots
            if statistical_results:
                if 'confusion_matrix' in statistical_results:
                    pdf.savefig(statistical_results['confusion_matrix']['figure'], bbox_inches='tight')
                
                if 'roc_analysis' in statistical_results:
                    pdf.savefig(statistical_results['roc_analysis']['figure'], bbox_inches='tight')
        
        return pdf_buffer.getvalue()
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return b""

def perform_sentiment_analysis(df, text_column, ground_truth_column=None):
    """Perform comprehensive sentiment analysis"""
    results = {
        'vader_scores': [],
        'emotions': [],
        'sentiments': [],
        'processed_texts': [],
        'sentiment_probabilities': [],
        'continuous_scores': []
    }

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_rows = len(df)

    # Analyze each text
    for i, text in enumerate(df[text_column].astype(str)):
        progress = (i + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing {i+1}/{total_rows} texts...")

        cleaned_text = simple_clean(text)
        results['processed_texts'].append(cleaned_text)

        # VADER analysis
        vader_scores = analyze_sentiment_vader(cleaned_text)
        results['vader_scores'].append(vader_scores)

        if vader_scores:
            compound = vader_scores['compound']
            sentiment = get_sentiment_label(compound)
            results['continuous_scores'].append(compound)
            
            # Convert to probabilities
            pos_prob = max(0, vader_scores['pos'])
            neg_prob = max(0, vader_scores['neg'])
            neu_prob = max(0, vader_scores['neu'])
            total = pos_prob + neg_prob + neu_prob + 1e-6
            
            probs = [neg_prob/total, neu_prob/total, pos_prob/total]
            results['sentiment_probabilities'].append(probs)
        else:
            sentiment = 'Neutral'
            results['continuous_scores'].append(0.0)
            results['sentiment_probabilities'].append([0.33, 0.34, 0.33])

        results['sentiments'].append(sentiment)

        # Emotion analysis
        emotions = analyze_emotion_text2emotion(cleaned_text)
        results['emotions'].append(emotions)

    progress_bar.empty()
    status_text.empty()

    # Statistical analysis if ground truth is provided
    statistical_results = None
    if ground_truth_column is not None:
        try:
            y_true = df[ground_truth_column].astype(str)
            y_pred = results['sentiments']
            
            # Map to consistent labels
            label_mapping = {'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'}
            y_true = [label_mapping.get(label.lower(), label) for label in y_true]
            
            # Filter to common labels
            common_labels = list(set(y_true) & set(y_pred))
            if len(common_labels) >= 2:
                valid_indices = [i for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)) 
                               if true_label in common_labels and pred_label in common_labels]
                
                if len(valid_indices) > 5:
                    y_true_filtered = [y_true[i] for i in valid_indices]
                    y_pred_filtered = [y_pred[i] for i in valid_indices]
                    y_proba_filtered = [results['sentiment_probabilities'][i] for i in valid_indices]
                    texts_filtered = [results['processed_texts'][i] for i in valid_indices]
                    
                    # Encode labels
                    le = LabelEncoder()
                    le.fit(common_labels)
                    
                    y_true_encoded = le.transform(y_true_filtered)
                    y_pred_encoded = le.transform(y_pred_filtered)
                    y_proba_array = np.array(y_proba_filtered)
                    
                    # Perform statistical analysis
                    statistical_results = comprehensive_statistical_analysis(
                        y_true_encoded, y_pred_encoded, y_proba_array, 
                        le.classes_, texts_filtered
                    )
        except Exception as e:
            st.error(f"Error in statistical analysis: {e}")

    return results, statistical_results

def process_csv_file(uploaded_file):
    """Process uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📋 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Data Types", len(df.dtypes.unique()))

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        text_columns = df.select_dtypes(include=['object']).columns.tolist()

        if text_columns:
            st.subheader("🎯 Select Text Column for Analysis")
            selected_column = st.selectbox(
                "Choose the column containing text data:", 
                text_columns,
                help="Select the column that contains the text you want to analyze"
            )

            # Ground truth column selection
            potential_gt_columns = [col for col in df.columns if 'sentiment' in col.lower() or 'label' in col.lower()]
            
            ground_truth_column = None
            if potential_gt_columns:
                ground_truth_column = st.selectbox(
                    "Select ground truth column (optional - for statistical validation):",
                    ["None"] + potential_gt_columns
                )
                if ground_truth_column == "None":
                    ground_truth_column = None

            if st.button("🚀 Start Comprehensive Analysis", type="primary"):
                with st.spinner("Performing comprehensive analysis..."):
                    analysis_results, statistical_results = perform_sentiment_analysis(df, selected_column, ground_truth_column)
                    st.session_state.processed_data = df
                    st.session_state.analysis_results = analysis_results
                    st.session_state.statistical_results = statistical_results

        else:
            st.error("❌ No text columns found in the dataset. Please ensure your CSV contains text data.")

    except Exception as e:
        st.error(f"❌ Error processing CSV file: {e}")
        st.code(traceback.format_exc())

def display_analysis_results(df, analysis_results, statistical_results=None):
    """Display comprehensive analysis results"""
    st.success("✅ Analysis completed successfully!")

    # Create results dataframe
    results_df = df.copy()
    results_df['Processed_Text'] = analysis_results['processed_texts']
    results_df['Predicted_Sentiment'] = analysis_results['sentiments']

    # Add VADER scores
    if analysis_results['vader_scores'] and analysis_results['vader_scores'][0] is not None:
        results_df['VADER_Compound'] = [score['compound'] if score else 0 for score in analysis_results['vader_scores']]
        results_df['VADER_Positive'] = [score['pos'] if score else 0 for score in analysis_results['vader_scores']]
        results_df['VADER_Negative'] = [score['neg'] if score else 0 for score in analysis_results['vader_scores']]
        results_df['VADER_Neutral'] = [score['neu'] if score else 0 for score in analysis_results['vader_scores']]

    # Add emotion scores
    if analysis_results['emotions'] and analysis_results['emotions'][0]:
        emotion_keys = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
        for key in emotion_keys:
            results_df[f'Emotion_{key}'] = [emotions.get(key, 0) if emotions else 0 for emotions in analysis_results['emotions']]

    # Display results
    st.subheader("📈 Analysis Results")
    st.dataframe(results_df, use_container_width=True)

    # Sentiment distribution
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = pd.Series(analysis_results['sentiments']).value_counts()

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

    # Statistical Analysis Results
    if statistical_results and 'error' not in statistical_results:
        st.subheader("📈 Advanced Statistical Analysis")
        
        tabs = st.tabs([
            "Classification Metrics", "Confusion Matrix", "ROC Analysis", 
            "Cross-Validation", "Distribution Analysis"
        ])
        
        with tabs[0]:
            if 'metrics' in statistical_results:
                metrics = statistical_results['metrics']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col2:
                    st.metric("Precision (Macro)", f"{metrics['precision_macro']:.3f}")
                with col3:
                    st.metric("Recall (Macro)", f"{metrics['recall_macro']:.3f}")
                with col4:
                    st.metric("F1-Score (Macro)", f"{metrics['f1_macro']:.3f}")
                
                st.metric("Cohen's Kappa", f"{metrics['cohen_kappa']:.3f}")
        
        with tabs[1]:
            if 'confusion_matrix' in statistical_results:
                st.pyplot(statistical_results['confusion_matrix']['figure'])
        
        with tabs[2]:
            if 'roc_analysis' in statistical_results:
                st.pyplot(statistical_results['roc_analysis']['figure'])
                st.write("**AUC Scores by Class:**")
                for class_name, auc in statistical_results['roc_analysis']['auc_scores'].items():
                    st.write(f"- {class_name}: {auc:.3f}")
        
        with tabs[3]:
            if 'cross_validation' in statistical_results:
                cv = statistical_results['cross_validation']
                st.metric("Cross-Validation Accuracy", f"{cv['mean_accuracy']:.3f} ± {cv['std_accuracy']:.3f}")
                ci = cv['confidence_interval_95']
                st.write(f"**95% Confidence Interval**: [{ci[0]:.3f}, {ci[1]:.3f}]")
        
        with tabs[4]:
            if 'distribution' in statistical_results:
                dist = statistical_results['distribution']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Confidence", f"{dist['confidence_mean']:.3f}")
                with col2:
                    st.metric("Error Rate", f"{dist['error_rate']:.3f}")

    # Download options
    st.subheader("📥 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV Results",
            data=csv_data,
            file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        pdf_data = generate_comprehensive_pdf(results_df, statistical_results, sentiment_counts)
        if pdf_data:
            st.download_button(
                label="📊 Download PDF Report",
                data=pdf_data,
                file_name=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
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
                sentiment = get_sentiment_label(compound)
                
                if sentiment == 'Positive':
                    st.success(f"Overall Sentiment: 😊 {sentiment}")
                elif sentiment == 'Negative':
                    st.error(f"Overall Sentiment: 😞 {sentiment}")
                else:
                    st.info(f"Overall Sentiment: 😐 {sentiment}")

        with col2:
            st.write("**Emotion Analysis:**")
            if emotions:
                st.json(emotions)

                # Dominant emotion
                if emotions:
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                    st.info(f"Dominant Emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.2f})")

def main():
    st.title("📊 Advanced Sentiment Analysis App")
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Features Include:
    - **Comprehensive Analysis**: VADER sentiment analysis + emotion detection
    - **Statistical Evaluation**: Confusion matrix, ROC curves, cross-validation
    - **Performance Metrics**: Precision, Recall, F1-Score, Cohen's Kappa  
    - **Visual Reports**: Charts, graphs, and downloadable PDF reports
    - **Sample Data**: Download test CSV with 25 sample reviews
    """)

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Mode:",
        ["📤 File Upload", "✏️ Manual Text Input"],
        help="Select how you want to input text for analysis"
    )

    # Sample CSV download
    st.sidebar.markdown("---")
    st.sidebar.markdown("📂 **Download Sample CSV for Testing:**")
    sample_df = create_sample_csv()
    sample_csv = sample_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="⬇️ Download Sample CSV",
        data=sample_csv,
        file_name="sample_reviews.csv",
        mime="text/csv",
        help="Download a sample CSV file with 25 reviews for testing the app"
    )
    st.sidebar.markdown("---")

    if analysis_mode == "📤 File Upload":
        st.header("📤 File Upload Analysis")
        st.markdown("Upload a CSV file containing text data for comprehensive sentiment analysis.")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file with text data. Include a ground truth column for statistical validation."
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
            if (st.session_state.analysis_results is not None and 
                st.session_state.processed_data is not None):
                display_analysis_results(
                    st.session_state.processed_data, 
                    st.session_state.analysis_results,
                    st.session_state.statistical_results
                )

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
            <p>📊 <strong>Advanced Features:</strong> Statistical validation, confusion matrix, ROC analysis, cross-validation, and comprehensive PDF reports</p>
            <p>Built with ❤️ using Streamlit | Powered by VADER, Text2Emotion & Scikit-learn</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
