import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16
from PIL import Image

import warnings

sns.set_style("whitegrid")
warnings.filterwarnings("ignore")


data = pd.read_csv("../data/data.csv")
del data["Unnamed: 0"]

img_paths, y, train_test = [], [], []
for i in tqdm(range(len(data))):
    p = f"../data/Images/{data.name[i]}"
    if os.path.exists(p):
        img_paths.append(p)
        y.append(data.bmi[i])
        train_test.append(data.is_training[i])


y = np.array(y)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


from facenet_pytorch import MTCNN, InceptionResnetV1

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define the transformation to preprocess the images
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def extract_features(img):
    img = img.convert('RGB')
    face = mtcnn(img)
    if face is None:
        face = preprocess(img)

    img = torch.stack([face]).to(device)

    with torch.no_grad():
        features = resnet(img)

    return features[0].cpu().numpy()


import cv2
import streamlit as st
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image


def capture_face():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow("Webcam", rgb_frame)

        # Display the webcam feed
        # st.image(rgb_frame, channels="RGB")

        if st.button("Capture", key=f"{np.random.randint(0, 100000)}"):
            # Detect and capture the face
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_img = rgb_frame[y:y+h, x:x+w]

                # Preprocess the captured face
                pil_img = Image.fromarray(face_img)
                img = pil_img.resize((160, 160))
                img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

                # Generate embeddings using the InceptionResnetV1 model
                with torch.no_grad():
                    embeddings = resnet(img)

                return embeddings
            else:
                st.warning("No face detected. Please try again.")
                continue
        if st.button("Stop", key=f"{np.random.randint(0, 100000)}"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    st.title("Face Recognition App")

    embeddings = capture_face()

    if embeddings is not None:

        st.subheader("Prediction:")
        st.write(embeddings.shape)

if __name__ == '__main__':
    main()
