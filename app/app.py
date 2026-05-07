import streamlit as st
import pandas as pd
import pickle
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ---------------------------------------------------
# DOWNLOAD NLTK RESOURCES
# ---------------------------------------------------

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ---------------------------------------------------
# LOAD TRAINED MODEL + TFIDF VECTORIZER
# ---------------------------------------------------

model = pickle.load(open("models/sentiment_model.pkl", "rb"))

vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

# ---------------------------------------------------
# HINGLISH NORMALIZATION DICTIONARY
# ---------------------------------------------------

hinglish_dict = {
    "mast": "good",
    "bakwas": "bad",
    "bekar": "bad",
    "acha": "good",
    "bahut": "very",
    "badiya": "good",
    "faltu": "bad"
}

# ---------------------------------------------------
# ASPECT KEYWORDS
# ---------------------------------------------------

aspects = {

    "Delivery": [
        "delivery",
        "late",
        "rider",
        "slow",
        "fast"
    ],

    "Food": [
        "food",
        "pizza",
        "burger",
        "taste",
        "tasty",
        "delicious",
        "cold",
        "paneer",
        "biryani"
    ],

    "Packaging": [
        "packaging",
        "packed",
        "spill",
        "box"
    ],

    "Price": [
        "price",
        "expensive",
        "cheap",
        "cost"
    ]
}

# ---------------------------------------------------
# TEXT PREPROCESSING FUNCTION
# ---------------------------------------------------

def preprocess_text(text):

    text = str(text)

    # lowercase
    text = text.lower()

    # remove special characters/emojis
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # tokenization
    tokens = word_tokenize(text)

    # stopwords
    stop_words = set(stopwords.words('english'))

    tokens = [word for word in tokens if word not in stop_words]

    # hinglish normalization
    tokens = [hinglish_dict.get(word, word) for word in tokens]

    # join back
    cleaned_text = " ".join(tokens)

    return cleaned_text

# ---------------------------------------------------
# ASPECT EXTRACTION FUNCTION
# ---------------------------------------------------

def extract_aspects(review):

    detected_aspects = []

    words = review.lower().split()

    for aspect, keywords in aspects.items():

        for keyword in keywords:

            if keyword in words:

                detected_aspects.append(aspect)

                break

    return detected_aspects

# ---------------------------------------------------
# SENTIMENT PREDICTION FUNCTION
# ---------------------------------------------------

def predict_sentiment(review):

    cleaned_review = preprocess_text(review)

    vectorized_review = vectorizer.transform([cleaned_review])

    prediction = model.predict(vectorized_review)

    return prediction[0]

# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------

st.set_page_config(
    page_title="Food Review Sentiment Analysis",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Zomato / Swiggy Review Sentiment Analysis")

st.write(
    """
    Analyze Indian food delivery reviews using NLP and Machine Learning.
    
    Features:
    - Sentiment Analysis
    - Aspect Extraction
    - Hinglish Support
    """
)

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

review = st.text_area(
    "Enter your review",
    height=150
)

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

if st.button("Analyze Review"):

    if review.strip() == "":

        st.warning("Please enter a review.")

    else:

        # sentiment prediction
        sentiment = predict_sentiment(review)

        # aspect extraction
        detected_aspects = extract_aspects(review)

        # cleaned review
        cleaned = preprocess_text(review)

        # ---------------------------------------------------
        # OUTPUT
        # ---------------------------------------------------

        st.subheader("Results")

        st.success(f"Predicted Sentiment: {sentiment}")

        st.write("### Detected Aspects")

        if detected_aspects:

            for aspect in detected_aspects:

                st.write(f"• {aspect}")

        else:

            st.write("No aspects detected.")

        st.write("### Cleaned Review")

        st.code(cleaned)

# ---------------------------------------------------
# SAMPLE REVIEWS
# ---------------------------------------------------

st.write("## Sample Reviews")

sample_reviews = [
    "Delivery was very late but pizza was amazing",
    "Packaging was bad and food was cold",
    "Paneer biryani mast tha",
    "Food was delicious and delivery was fast",
    "Very expensive but taste was good"
]

for sample in sample_reviews:

    st.write(f"• {sample}")