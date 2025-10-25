import streamlit as st
import cv2
import numpy as np
import PIL
from PIL import Image
import requests
import io

# Function to convert OpenCV image to PIL image
def cv2_to_pil(image):
    if image is None or image.size == 0:
        raise ValueError("Empty image passed to cv2_to_pil")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Function to detect faces
def detect_faces(frame, scale_factor, min_neighbors, rect_color_bgr):
# Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Convert the image to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color_bgr, 2)
    return frame


st.title("Face Detection App 😍😍")
st.write("Start the camera to detect faces using Viola-Jones algorithm.")

# Sidebar for settings
st.sidebar.title("Settings ⚙️")
rect_color = st.sidebar.color_picker("Pick Rectangle Color", "#FF0000")
bgr_color = tuple(int(rect_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))[::-1]
min_neighbors = st.sidebar.slider("Adjust minNeighbors", 1, 10, 5)
scale_factor = st.sidebar.slider("Adjust scaleFactor", 1.01, 1.5, 1.1)

# About section
st.sidebar.title("About 📖")
st.sidebar.write("This app uses OpenCV and Streamlit to detect faces in real-time from your webcam feed.")

# Contact info section
st.sidebar.title("Contact 📞")
st.sidebar.write("For any inquiries, please contact us at: support@example.com")

# Help section
st.sidebar.title("Help ❓")
st.sidebar.write("If you need help, please refer to the [documentation](https://example.com/docs) or contact support.")

# Necessary links section
st.sidebar.title("Links 🔗")
st.sidebar.write("[GitHub Repository](https://github.com/1Chizey/FaceDetectionApp)")
st.sidebar.write("[Official Website](https://example.com)")

# Complaint section
st.sidebar.title("For Complaints 📝")
complaint = st.sidebar.text_area("kindly type in your complaint here:")
if st.sidebar.button("Submit Complaint"):
    st.sidebar.write("Complaint submitted successfully!")
    # Here you can add code to save the complaint to a database or file

st.sidebar.button("Refresh Complaints")

# Add instructions
st.write("**Instructions:**")
st.write("- Use the sliders to adjust the detection parameters.")
st.write("- Choose the color of the rectangles around detected faces.")
st.write("- Click on 'Detect Faces' to run the face detection algorithm.")
st.write("- Click on 'Save Image' to save the image with detected faces.")

# Start the camera
cam = st.camera_input("Capture Image")
if cam is not None:
    bytes_data = cam.getvalue()
    frame = np.array(Image.open(io.BytesIO(bytes_data)))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = detect_faces(frame, scale_factor, min_neighbors, bgr_color)
    st.image(cv2_to_pil(frame), caption="Image with Detected Faces", use_column_width=True)

    # Add feature to save the image
    result_img = cv2_to_pil(frame)
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    if st.download_button("Save Image", data=byte_im, file_name="detected_faces.jpg", mime="image/jpeg"):
        st.write("Image saved successfully!")
