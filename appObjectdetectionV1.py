
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")

# Streamlit app
st.title("Object Detection")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display original and resulting images if image uploaded
if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    # Display original image
    st.subheader("Original Image")
    st.image(image, channels="BGR", use_column_width=True)

    # Perform object detection
    results = model.predict(image)

    # Display resulting images
    for result in results:
        st.subheader("Result")
        st.image(result.plot(), channels="BGR", use_column_width=True)
