import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import streamlit as st
import time
from gtts import gTTS
import tempfile
import google.generativeai as genai
from streamlit_js_eval import streamlit_js_eval
import speech_recognition as sr

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    import av
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.5-flash"


@st.cache_resource
def load_model():
    model_alphabet = joblib.load("gesture_model_alphabet.pkl")
    le_alphabet = joblib.load("label_encoder_alphabet.pkl")
    return model_alphabet, le_alphabet


model, label_encoder = load_model()
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def predict_hand(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8) as hands:
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = results.multi_hand_landmarks[0].landmark
            features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().reshape(1, -1)
            pred = model.predict(features)
            pred_label = label_encoder.inverse_transform(pred)[0]
            return pred_label, img
        else:
            return None, img


if WEBRTC_AVAILABLE:
    class SignLanguageProcessor(VideoProcessorBase):
        def __init__(self):
            self.note = ""
            self.last_prediction = ""
            self.last_pred_time = 0
            self.cooldown = 2.0
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
            )

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pred_label, conf = None, 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    features = []
                    for lm in hand_landmarks.landmark:
                        features.extend([lm.x, lm.y, lm.z])
                    features_np = np.array(features).reshape(1, -1)

                    try:
                        probs = model.predict_proba(features_np)
                        conf = float(np.max(probs))
                        pred = model.predict(features_np)
                        pred_label = label_encoder.inverse_transform(pred)[0]
                    except:
                        pred_label = None
                        conf = 0.0

            if pred_label and conf > 0.9:
                now = time.time()
                if pred_label != self.last_prediction or now - self.last_pred_time > self.cooldown:
                    self.note += pred_label
                    self.last_prediction = pred_label
                    self.last_pred_time = now
                cv2.putText(img, f"{pred_label} ({conf:.2f})", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(img, format="bgr24")


def cleanup_with_gemini(raw_text: str) -> str:
    try:
        prompt = f"""
        Susun ulang teks berikut hasil deteksi huruf menjadi kalimat yang lebih rapih:
        "{raw_text}"
        Jika tidak bisa disusun, kembalikan apa adanya.
        Hanya balas dengan kalimat yang sudah dirapihkan, tanpa tambahan apapun.
        """
        response = genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"[Gemini ERROR] {e}")
        return raw_text


st.set_page_config(page_title="Sancara - Translator", layout="wide")
st.title("🤟🗣️ Sancara - Sign Language & Speech Translator")

mode = st.radio("Pilih Mode Translator  :", ["Bahasa Isyarat", "Speech to Text"])

if mode == "Bahasa Isyarat":
    if WEBRTC_AVAILABLE and os.getenv("STREAMLIT_RUNTIME") is None:
        st.info("📷 Mode Kamera aktif (jalankan di lokal)")
        col1, col2 = st.columns([2, 1])

        with col1:
            ctx = webrtc_streamer(
                key="sancara",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                video_processor_factory=SignLanguageProcessor,
            )

        with col2:
            if ctx and ctx.video_processor:
                st.text_area("Hasil Deteksi", ctx.video_processor.note, height=300)
    else:
        st.info("📸 Mode Upload Gambar aktif (karena tidak ada akses kamera di cloud).")
        uploaded = st.file_uploader("Unggah gambar tangan", type=["jpg", "jpeg", "png"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            pred, result_img = predict_hand(img)
            if pred:
                st.success(f"Huruf terdeteksi: **{pred}**")
                tts = gTTS(text=f"Huruf {pred}", lang="id")
                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(tmpfile.name)
                st.audio(tmpfile.name)
            else:
                st.error("Tidak terdeteksi tangan.")
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)

elif mode == "Speech to Text":
    st.subheader("🎙️ Ucapkan sesuatu dalam Bahasa Indonesia")

    if st.button("Rekam Suara"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Silakan bicara SEKARANG... (rekam 5 detik)")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = recognizer.listen(source, timeout=8, phrase_time_limit=7)
            except sr.WaitTimeoutError:
                st.error("Timeout: Tidak ada suara terdeteksi.")
                st.stop()

        with st.spinner("Mengkonversi suara ke teks..."):
            try:
                text = recognizer.recognize_google(audio, language="id-ID")
                if text:
                    st.success(f"Teks terdeteksi: **{text}**")
                    kalimat_rapih = cleanup_with_gemini(text)
                    st.info(f"Kalimat rapih: **{kalimat_rapih}**")

                    tts = gTTS(text=kalimat_rapih, lang="id")
                    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(tmpfile.name)
                    st.audio(tmpfile.name)
                else:
                    st.error("Tidak ada teks terdeteksi.")
            except sr.UnknownValueError:
                st.error("Tidak bisa memahami audio")
            except sr.RequestError as e:
                st.error(f"Error service: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
