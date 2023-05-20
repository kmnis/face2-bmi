import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Webcam Live Feed")

capture = st.button('Capture')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)
captured_frame = None

while True:
    cap, frame = camera.read()
    
    if cap:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        if capture:
            captured_frame = np.copy(frame)
            capture = False

    if captured_frame is not None and not capture:
        st.image(captured_frame)
        break

camera.release()

