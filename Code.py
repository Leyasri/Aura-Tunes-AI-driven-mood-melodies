import streamlit as st
import cv2
from deepface import DeepFace
import sounddevice as sd
import numpy as np
import librosa
import subprocess

# 🎵 Mood-to-Playlist Mapping (Updated with fear)
mood_to_playlist = {
    "happy": "https://open.spotify.com/playlist/5ACAHVlMPRrgnnZ8temmIh?si=LlvE8RZfS92RY-fUbqAX_g&pi=_Z5pjFSERLalA",
    "sad": "https://open.spotify.com/playlist/0RkK2ZAXWD5HEmCJZ00i1G?si=Aa_r-9rwSZuxa1ji7z51Jw",
    "angry": "https://open.spotify.com/playlist/5cwtgqs4L1fX8IKoQebfjJ?si=E4KoOSw1T3ShHyIhV4CabA",
    "surprise": "https://open.spotify.com/playlist/7vatYrf39uVaZ8G2cVtEik?si=mxgDHP14RSCGMsNSLbwTNA",
    "fear": "https://open.spotify.com/playlist/37i9dQZF1DXdpQPPZq3F7n?si=UJ5VJM6QbaWiDP9nExwnA",
    "neutral": "https://open.spotify.com/playlist/4nqbYFYZOCospBb4miwHWy?si=2S0YqR26RJSRrmwKcvsjlQ"
}

# Set Chrome as the default browser manually
chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe"
def open_in_chrome(url):
    subprocess.run([chrome_path, url])

# 🎭 Facial Emotion Detection with Live Camera Feed
def detect_face_emotion():
    st.write("📸 Opening camera for face detection...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Unable to access the webcam.")
        return "neutral"

    frame_placeholder = st.empty()
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            return emotion
        except Exception as e:
            st.error(f"⚠️ Face Emotion Detection Error: {e}")
            return "neutral"
    return "neutral"

# 🎧 Voice-Based Mood Detection
def detect_voice_mood(duration=3, samplerate=22050):
    st.write("🎤 Recording your voice... Please speak!")
    try:
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()

        audio_data = audio_data.flatten()
        mfccs = librosa.feature.mfcc(y=audio_data, sr=samplerate, n_mfcc=13)
        mean_mfccs = np.mean(mfccs, axis=1)

        if mean_mfccs[0] > 50:
            return "happy"
        elif mean_mfccs[0] < -20:
            return "sad"
        else:
            return "neutral"
    except Exception as e:
        st.error(f"⚠️ Voice Mood Detection Error: {e}")
        return "neutral"

# 🎵 Streamlit UI
st.title("🎶 Aura Tunes: AI Mood-Based Music Player")
st.write("Detect your mood using AI and get personalized music!")

if st.button("Start Detection"):
    face_mood = detect_face_emotion()
    voice_mood = detect_voice_mood()
    
    # Prioritize face mood if it's valid
    final_mood = face_mood if face_mood in mood_to_playlist else voice_mood

    if final_mood:
        st.success(f"✅ Your detected mood: **{final_mood.capitalize()}**")
        playlist_link = mood_to_playlist.get(final_mood, mood_to_playlist["happy"])

        # Automatically redirect to Spotify in Chrome
        st.write("🎧 Redirecting to your personalized playlist...")
        open_in_chrome(playlist_link)
    else:
        st.error("❌ Mood detection failed. Try again.")
