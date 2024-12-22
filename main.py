import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.models import load_model
import cv2
import os
from PIL import Image

# Load Data for Music
music_data = {
    'song': [
        "Walking on Sunshine", "Feeling Good", "Mad World", "Rain on Me",
        "We Will Rock You", "You Are My Sunshine", "The Sound of Silence", "Beautiful Day","Kissik","Hukum"
    ],
    'artist': [
        "Katrina and the Waves", "Michael Bublé", "Tears for Fears", "Lady Gaga & Ariana Grande",
        "Queen", "Johnny Cash", "Simon & Garfunkel", "U2","Dsp","Anirudh"
    ],
    'mood': ['Happy', 'Happy', 'Sad', 'Sad', 'Energetic','Happy', 'Sad', 'Happy','Energetic','Energetic']
}
music_df = pd.DataFrame(music_data)

# Functions for Mood Detection
@st.cache_resource
def load_emotion_model():
    # Update the path here to the correct file location
    model_path = r"C:\Users\Tapas\Downloads\projectmoodmusic\emotion_model.h5"
    if not os.path.exists(model_path):
        st.error("The emotion model file is missing. Please ensure it exists at the specified path.")
        st.stop()
    return load_model(model_path)

@st.cache_data
def train_text_mood_model():
    text_data = {
        'lyrics': [
            "I'm walking on sunshine, whoa-oh!", "And I'm feeling good, yeah, nothing can bring me down",
            "I'm so tired of being here, suppressed by all my childish fears",
            "Why does it always rain on me? Is it because I lied when I was seventeen?",
            "We will, we will rock you!", "You are my sunshine, my only sunshine",
            "Hello darkness, my old friend, I've come to talk with you again",
            "It's a beautiful day, don't let it get away",
            "I'm so sad today, can't shake this feeling", "I'm scared, I don't know what's coming",
            "I can't believe I made it through the storm, I'm alive and stronger",
            "I'm full of energy, let's go!"
        ],
        'mood': ['Happy', 'Happy', 'Sad', 'Sad', 'Energetic', 'Happy', 'Sad', 'Happy', 'Sad', 'Fear', 'Energetic', 'Energetic']
    }
    df = pd.DataFrame(text_data)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['lyrics'])
    y = df['mood']

    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer

def detect_text_mood(user_input, model, vectorizer):
    user_input_vectorized = vectorizer.transform([user_input])
    predicted_mood = model.predict(user_input_vectorized)[0]
    prediction_probs = model.predict_proba(user_input_vectorized)
    st.write(f"Prediction probabilities: {prediction_probs}")
    return predicted_mood

def detect_image_mood_from_image(uploaded_image, emotion_model):
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion_to_mood = {
        'Angry': 'Energetic', 'Disgust': 'Sad', 'Fear': 'Sad', 'Happy': 'Happy',
        'Sad': 'Sad', 'Surprise': 'Happy', 'Neutral': 'Happy'
    }
    image = uploaded_image.convert("RGB")
    img_array = np.array(image)
    gray_frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (48, 48))
    img = resized_frame.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    try:
        predictions = emotion_model.predict(img)
        if predictions.ndim == 2 and predictions.shape[1] == len(emotion_labels):
            predicted_emotion = emotion_labels[np.argmax(predictions)]
        else:
            raise ValueError("Prediction output shape is incorrect.")
    except Exception as e:
        st.error(f"Error during emotion detection: {str(e)}")
        predicted_emotion = 'Neutral'
    return emotion_to_mood.get(predicted_emotion, 'Neutral')

st.set_page_config(page_title="Mood-Based Music Recommendation", page_icon="🎵", layout="wide")
st.title("🎵 Mood-Based Music Recommendation")

st.sidebar.title("Input Options")
st.sidebar.markdown("<p style='font-size:18px;color:blue;'>Choose how you'd like to provide your mood:</p>", unsafe_allow_html=True)

input_choice = st.sidebar.radio("Input Method", ["Text", "Upload Image"])

emotion_model = load_emotion_model()
text_mood_model, vectorizer = train_text_mood_model()

if input_choice == "Text":
    st.markdown("<h3 style='color:green;'>Describe how you feel:</h3>", unsafe_allow_html=True)
    user_text = st.text_area("Enter your feelings:")
    if st.button("Analyze Mood and Recommend Music"):
        if user_text.strip():
            mood = detect_text_mood(user_text, text_mood_model, vectorizer)
            st.markdown(f"<h3 style='color:blue;'>Detected Mood: {mood}</h3>", unsafe_allow_html=True)
            st.subheader("Recommended Songs:")
            recommendations = music_df[music_df['mood'] == mood]
            for idx, row in recommendations.iterrows():
                st.write(f"- {row['song']} by {row['artist']}")
        else:
            st.warning("Please enter some text to analyze.")
elif input_choice == "Upload Image":
    st.markdown("<h3 style='color:purple;'>Upload an Image to Analyze Mood:</h3>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze Mood and Recommend Music"):
            mood = detect_image_mood_from_image(image, emotion_model)
            st.markdown(f"<h3 style='color:blue;'>Detected Mood: {mood}</h3>", unsafe_allow_html=True)
            st.subheader("Recommended Songs:")
            recommendations = music_df[music_df['mood'] == mood]
            for idx, row in recommendations.iterrows():
                st.write(f"- {row['song']} by {row['artist']}")
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='font-size:16px; text-align:center;'>Built with ❤ by P Tapasvi</p>", unsafe_allow_html=True)
