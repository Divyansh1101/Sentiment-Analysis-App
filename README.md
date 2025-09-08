# 📝 Sentiment Analysis App

A **Streamlit-based NLP web application** that performs **sentiment analysis** on user-input text using multiple techniques such as **VADER, Text2Emotion, and ML models**. The app provides interactive visualizations including **word clouds, polarity distributions, and SHAP explainability plots**.  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
![Last Commit](https://img.shields.io/github/last-commit/Divyansh1101/Sentiment-Analysis-App?style=flat-square)  

---

## ✨ Features

- 🧠 **Multi-Model Sentiment Analysis** – Supports **VADER** for polarity detection and **Text2Emotion** for emotion recognition.  
- 🧪 **Data Preprocessing** – Tokenization, stopword removal, and vectorization for clean input.  
- 📊 **Interactive Visualizations** – Generate **word clouds, sentiment distributions, and bar charts**.  
- ⚡ **Explainability with SHAP** – Understand how different words contribute to predictions.  
- 🌍 **Language Detection** – Auto-detect language of input text before analysis.  
- 🎨 **Streamlit UI** – Clean, theme-configurable web app interface.  

---

## 📂 Repository Structure

```
.
├── .streamlit/               # Streamlit theme & config (config.toml)
├── local/                    # Dev requirements (requirements-dev.txt)
├── tests/                    # Unit tests (test_preprocessing.py)
├── example_dataset.csv        # Sample dataset for testing
├── requirements.txt           # App dependencies
├── runtime.txt                # Runtime Python version
├── Dockerfile                 # Containerization setup
├── README.md                  # Project documentation
└── app.py / main.py (expected)# Main Streamlit app file
```

---

## 🚀 Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Divyansh1101/Sentiment-Analysis-App.git
cd Sentiment-Analysis-App
```

### 2️⃣ Create a virtual environment & install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🧪 Testing

Run unit tests with:
```bash
pytest tests/
```

---

## 📊 Example Outputs

- ✅ **Sentiment classification** (Positive / Negative / Neutral)  
- 🎭 **Emotion breakdown** (Happy, Angry, Fear, Surprise, Sad)  
- ☁️ **Word cloud** visualization of input text  
- 🔍 **SHAP explainability** plots  

---

## 🛠️ Tech Stack

- **Python 3.8+**  
- **Streamlit** – Interactive web app  
- **scikit-learn** – ML preprocessing & models  
- **NLTK / VADER** – Lexicon-based sentiment analysis  
- **Text2Emotion** – Emotion recognition  
- **Matplotlib / WordCloud** – Visualizations  
- **SHAP** – Model explainability  

---

## 🚀 Deployment

### 🔹 Streamlit Cloud
1. Push this repository to GitHub.  
2. Go to [Streamlit Cloud](https://share.streamlit.io/).  
3. Select this repo and deploy – Streamlit will use `requirements.txt` and `runtime.txt` automatically.  

### 🔹 Docker
Build and run the app inside a Docker container:
```bash
# Build image
docker build -t sentiment-app .

# Run container
docker run -p 8501:8501 sentiment-app
```
Access the app at: [http://localhost:8501](http://localhost:8501)  

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).  
