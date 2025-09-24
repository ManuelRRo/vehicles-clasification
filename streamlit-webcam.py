import cv2
import streamlit as st

# Abrir la cámara o un video
cap = cv2.VideoCapture(0)  # usa 0 para webcam, o "video.mp4" para archivo

st.title("Streaming de video con OpenCV y Streamlit")

# Contenedor para la imagen
frame_container = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir de BGR (OpenCV) a RGB (Streamlit usa RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mostrar frame en Streamlit
    frame_container.image(frame, channels="RGB")

cap.release()
