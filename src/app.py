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


with open("../models/lr.p", "rb") as f:
    lr = pickle.load(f)

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)
    
    detected_image = Image.fromarray(mtcnn2(img).numpy().transpose(1, 2, 0).astype(np.uint8))
    st.image(detected_image, caption="Detected Face")
    
    embeddings = extract_features(img)
    bmi = round(lr.predict([embeddings])[0], 2)
    st.write(f"Your BMI is {bmi}")