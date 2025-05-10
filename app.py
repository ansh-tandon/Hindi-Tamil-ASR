import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import tempfile
from pydub import AudioSegment
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🎵 Audio Feature Visualizer for Hindi-Tamil Mixed Audio")

# Upload mixed Hindi-Tamil audio
uploaded_file = st.file_uploader("Upload a mixed Hindi + Tamil audio file (.mp3 or .wav)", type=["mp3", "wav"])

if uploaded_file:
    # Save to temp file
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    if uploaded_file.name.endswith(".mp3"):
        audio = AudioSegment.from_file(uploaded_file, format="mp3")
        audio.export(temp_audio.name, format="wav")
    else:
        temp_audio.write(uploaded_file.read())
        temp_audio.flush()

    # Load audio with librosa
    y, sr = librosa.load(temp_audio.name, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Read the audio file as bytes for Streamlit playback
    audio_bytes = uploaded_file.read()

    # Audio Playback
    st.subheader("🎧 Play the Audio")
    st.audio(audio_bytes, format="audio/wav", start_time=0)

    ### Waveform Visualization
    st.subheader("🎛 Waveform of the Audio")
    fig, ax = plt.subplots(figsize=(15, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Audio Waveform")
    st.pyplot(fig)

    ### Spectrogram Visualization
    st.subheader("🌈 Spectrogram")
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(15, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    ax.set_title("Mel Spectrogram (dB)")
    st.pyplot(fig)

    ### MFCC Visualization
    st.subheader("📊 MFCC (Mel-Frequency Cepstral Coefficients)")
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    fig, ax = plt.subplots(figsize=(15, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax)
    ax.set_title("MFCC")
    st.pyplot(fig)

    ### Additional Features: Chroma, Spectral Contrast, Zero-Crossing Rate
    st.subheader("🔍 Additional Audio Features")

    # Chroma Feature
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    fig, ax = plt.subplots(figsize=(15, 4))
    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', ax=ax)
    ax.set_title("Chroma Feature")
    st.pyplot(fig)

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    fig, ax = plt.subplots(figsize=(15, 4))
    librosa.display.specshow(spectral_contrast, x_axis='time', ax=ax)
    ax.set_title("Spectral Contrast")
    st.pyplot(fig)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(zcr.T)
    ax.set_title("Zero Crossing Rate")
    st.pyplot(fig)

    ### Audio Duration and Info
    st.write(f"Duration of the audio: {duration:.2f} seconds")
    st.write(f"Sampling Rate: {sr} Hz")
    