import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import librosa
from keras.models import load_model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import joblib

# Load the trained LSTM model
model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40, 1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])
model.load_weights('modelEA.h5')  # Load model weights

pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

# Define emotion labels
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

emotions_emoji_dict = {
    "anger": "😠", 
    "disgust": "🤮", 
    "fear": "😨😱", 
    "happy": "🤗", 
    "joy": "😂", 
    "neutral": "😐", 
    "sad": "😔",
    "sadness": "😔", 
    "shame": "😳", 
    "surprise": "😮"
}

# Function to extract MFCC features from audio file
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Function to preprocess input audio for model prediction
def preprocess_audio(audio_file):
    mfcc = extract_mfcc(audio_file)
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
    return mfcc

# Function to predict emotion from audio file
def predict_emotion(audio_file):
    processed_audio = preprocess_audio(audio_file)
    prediction = model.predict(processed_audio)
    emotion_label = np.argmax(prediction)
    return emotion_labels.get(emotion_label, "Unknown")

def predict_text_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Emotion Detection")
    st.subheader("Text and Audio Emotion Detection")

    # Text Emotion Detection
    with st.form(key='text_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit Text')

    if submit_text:
        prediction_text = predict_text_emotion(raw_text)
        emoji_icon_text = emotions_emoji_dict.get(prediction_text, "😐")
        st.write("{}:{}".format(prediction_text, emoji_icon_text))

    # Audio Emotion Detection
    with st.form(key='audio_form'):
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
        submit_audio = st.form_submit_button(label='Submit Audio')

    if submit_audio and audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        emotion_label_audio = predict_emotion(audio_file)
        st.write("Predicted Emotion from Audio:", emotion_label_audio)

if __name__ == '__main__':
    main()
