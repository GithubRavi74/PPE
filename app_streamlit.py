import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("models/best.pt")

st.title("🦺 PPE Detection System (YOLOv8)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model(image)
    result_img = results[0].plot()

    st.image(result_img, caption="Detection Result", use_container_width=True)