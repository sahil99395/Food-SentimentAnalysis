# 🍔 Zomato / Swiggy Review Sentiment & Aspect Extraction System

## 📌 Project Overview

This project is an NLP-based Sentiment Analysis and Aspect Extraction system built using Indian food delivery reviews from platforms like Zomato and Swiggy.

The system analyzes customer reviews and predicts:

* Overall Sentiment

  * Positive
  * Negative
  * Neutral

* Review Aspects

  * Delivery
  * Food Quality
  * Packaging
  * Pricing

The project also supports Hinglish review preprocessing, making it more suitable for real-world Indian review datasets.

---

# 🚀 Features

✅ Sentiment Analysis using Machine Learning
✅ Aspect Extraction from customer reviews
✅ Hinglish-aware preprocessing
✅ TF-IDF Feature Engineering
✅ Logistic Regression Classification
✅ Interactive Streamlit Web App
✅ Real-time Review Prediction
✅ Deployed Online

---

# 🧠 Technologies Used

| Technology          | Purpose                  |
| ------------------- | ------------------------ |
| Python              | Core Programming         |
| Pandas              | Data Handling            |
| NumPy               | Numerical Operations     |
| NLTK                | NLP Preprocessing        |
| Scikit-learn        | Machine Learning         |
| TF-IDF              | Text Vectorization       |
| Logistic Regression | Sentiment Classification |
| Streamlit           | Web Application          |
| Pickle              | Model Serialization      |
| Git & GitHub        | Version Control          |

---

# 📂 Project Structure

```bash
FoodSentiment-Analysis/
│
├── app/
│   └── app.py
│
├── data/
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_aspect_extraction.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Machine Learning Pipeline

## 1. Data Collection

Food delivery reviews dataset collected from Kaggle.

## 2. Text Preprocessing

The preprocessing pipeline includes:

* Lowercasing
* Special character removal
* Stopword removal
* Tokenization
* Hinglish normalization

Example:

```text
Original Review:
"Pizza mast tha but delivery bakwas 😭"

Cleaned Review:
"pizza good delivery bad"
```

---

# 📊 Feature Engineering

TF-IDF Vectorization was used to convert review text into numerical vectors.

Formula:

```text
TF-IDF = Term Frequency × Inverse Document Frequency
```

---

# 🤖 Model Used

## Logistic Regression

The sentiment classification model was trained using:

* TF-IDF Features
* Logistic Regression
* Balanced Class Weights

---

# 🔍 Aspect Extraction

Rule-based aspect extraction was implemented using keyword mapping.

Example:

| Keyword   | Aspect    |
| --------- | --------- |
| late      | Delivery  |
| tasty     | Food      |
| expensive | Price     |
| packaging | Packaging |

Example Output:

| Review                                  | Aspect         |
| --------------------------------------- | -------------- |
| Delivery was late but pizza was amazing | Delivery, Food |

---

# 🌐 Streamlit Web Application

The project includes a deployed Streamlit application where users can:

* Enter food delivery reviews
* Predict sentiment
* Detect review aspects
* Analyze Hinglish reviews

---

# 📈 Sample Predictions

| Review                              | Prediction |
| ----------------------------------- | ---------- |
| Food was amazing                    | Positive   |
| Delivery was terrible               | Negative   |
| Pizza mast tha but packaging bakwas | Mixed      |

---

# 🛠️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/FoodSentiment-Analysis.git
```

## Move into Project Folder

```bash
cd FoodSentiment-Analysis
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
streamlit run app/app.py
```

---

# 📌 Future Improvements

* BERT / Transformer-based sentiment analysis
* Aspect-wise sentiment prediction
* Real-time dashboard analytics
* Advanced multilingual NLP
* FastAPI deployment
* Spring Boot + Python ML integration
* Real-time review scraping pipeline

---

---

# 🌍 Live Demo

https://food-sentimentanalysis-sahil.streamlit.app/


---

# 👩‍💻 Author

Sahil Kumar

# ⭐ If You Like This Project

Give this repository a star on GitHub ⭐
