import streamlit as st

import os
import numpy as np
import pandas as pd
from glob import glob
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision.transforms as transforms
from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1

import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with st.spinner('Loading the models...'):
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(
        image_size=160, margin=40, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    mtcnn2 = MTCNN(
        image_size=160, margin=40, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
        device=device
    )

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

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


with open("/app/models/svm_original_casia-webface.p", "rb") as f:
    lr = pickle.load(f)

st.markdown("<center><h1>Know Your BMI</h1></center>", unsafe_allow_html=True)
st.caption("<center>Click a photo and the underlying Machine Learning model will predict your BMI</center>", unsafe_allow_html=True)

img_file_buffer = st.camera_input("Click a photo and the underlying Machine Learning model will predict your BMI", label_visibility="hidden")

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)
    
    detected_face = mtcnn2(img)
    if detected_face is None:
        st.write("No Face Detected")
    else:
        detected_face = Image.fromarray(detected_face.numpy().transpose(1, 2, 0).astype(np.uint8))
        st.image(detected_face, caption="Detected Face")
        embeddings = extract_features(img)
        bmi = round(lr.predict([embeddings])[0], 2) - 4
        st.write(f"Your BMI is {bmi}")
