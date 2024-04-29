
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Set page title and favicon
st.set_page_config(
    page_title="OBJECT DETECTION",
    page_icon=":eyeglasses:"
)

# Set app title and description
st.title("ONE-DAY: OBJECT DETECTION")
st.write("Upload an image and choose a task and model version.")

# Sidebar
st.sidebar.header("Model Configuration")

# Choose task
task = st.sidebar.radio("Task", ["Detection", "Segmentation", "Classification"])

# Choose model version based on task
if task == "Detection":
    versions = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
elif task == "Segmentation":
    versions = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt"]
elif task == "Classification":
    versions = ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt"]

# Select model version
version = st.sidebar.radio("Model Version", versions)

# Upload image
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Default image
default_image_path = "image_default.jpg"

# Display original and resulting images if image uploaded
if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
else:
    # Load default image
    image = cv2.imread(default_image_path)


# Load YOLO model based on task and version selected
model = YOLO(version)

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

else:
    # Load default image
    image = cv2.imread(default_image_path)


    # Display original image
    st.subheader("Original Image")
    st.image(image, channels="BGR", use_column_width=True)

    # Perform object detection, instance segmentation, or classification
    results = model.predict(image)

    # Display resulting images
    for result in results:
        st.subheader("Result")
        st.image(result.plot(), channels="BGR", use_column_width=True)
