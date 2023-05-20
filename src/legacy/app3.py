import cv2
import streamlit as st
import numpy as np

st.title("Webcam Live Feed")

run = True

def change_run():
    global run
    run = False

# st.write(run)

capture = st.button('Capture', on_click=change_run)

# def show_run():
#     st.write(run)

# capture2 = st.button('Show Run', on_click=show_run)



FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
captured_frame = None

while run:
    cap, frame = camera.read()
    if cap:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        captured_frame = np.copy(frame)

else:
    if captured_frame is not None:
        st.image(captured_frame)
    st.write('Stopped')
