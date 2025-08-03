import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import math

# ------------------- Page Setup -------------------
st.set_page_config(page_title="Body Measurement System", layout="centered")
st.title("📏 Body Measurement System")
st.markdown("Upload your image or use webcam to get body measurements in pixels and centimeters.")

# ------------------- MediaPipe Setup -------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# ------------------- Helper Function -------------------
def calculate_distance(p1, p2, width, height):
    """Calculate Euclidean distance between two pose points in pixels"""
    x1, y1 = int(p1.x * width), int(p1.y * height)
    x2, y2 = int(p2.x * width), int(p2.y * height)
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return int(distance), (x1, y1), (x2, y2)

# ------------------- User Input (Height) -------------------
st.sidebar.header("📌 User Info")
user_height_cm = st.sidebar.number_input("Enter your real height (cm):", min_value=100, max_value=250, value=170)

# ------------------- Image Input Options -------------------
st.subheader("📸 Select Input Method")
input_method = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

image_data = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_data = Image.open(uploaded_file)
        st.image(image_data, caption="Uploaded Image", use_column_width=True)

elif input_method == "Use Webcam":
    captured_image = st.camera_input("Take a photo")
    if captured_image:
        image_data = Image.open(captured_image)
        st.image(image_data, caption="Captured Image", use_column_width=True)

# ------------------- Pose Detection + Measurement -------------------
if image_data:
    # Convert image to numpy array and RGB → BGR
    img_np = np.array(image_data)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    result = pose.process(img_rgb)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        img_height, img_width, _ = img_np.shape
        annotated_img = img_np.copy()

        # Shoulder width
        shoulder_px, pt1_s, pt2_s = calculate_distance(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            img_width, img_height
        )

        # Waist width
        waist_px, pt1_w, pt2_w = calculate_distance(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
            img_width, img_height
        )

        # Estimate body height in pixels (mid-shoulder to mid-ankle)
        shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
        ankle_y = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y +
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y) / 2
        body_height_px = abs((ankle_y - shoulder_y) * img_height)

        # Real-world conversion
        if body_height_px > 0:
            scale = user_height_cm / body_height_px
            shoulder_cm = round(shoulder_px * scale, 2)
            waist_cm = round(waist_px * scale, 2)
        else:
            scale = 0
            shoulder_cm = waist_cm = 0.0

        # Draw lines and labels
        mp_drawing.draw_landmarks(annotated_img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.line(annotated_img, pt1_s, pt2_s, (0, 255, 0), 2)
        cv2.putText(annotated_img, f'{shoulder_px}px', (pt1_s[0], pt1_s[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.line(annotated_img, pt1_w, pt2_w, (255, 0, 0), 2)
        cv2.putText(annotated_img, f'{waist_px}px', (pt1_w[0], pt1_w[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show final image and results
        st.image(annotated_img, caption="Pose Landmarks & Measurements", use_column_width=True)

        st.subheader("📐 Measurement Results")
        st.write(f"Scale Factor: {round(scale, 4)} cm/pixel")
        st.table({
            "Measurement": ["Shoulder Width", "Waist Width"],
            "Pixels": [shoulder_px, waist_px],
            "Centimeters": [shoulder_cm, waist_cm]
        })

    else:
        st.warning("⚠️ No pose detected. Please upload a clear front-facing full-body image.")
import os
import datetime

# Save image after annotations
# Convert RGB → BGR for correct OpenCV saving
save_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

# Make sure 'output' folder exists
os.makedirs("output", exist_ok=True)

# Unique file name using time
filename = f"output/result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

# Save image
cv2.imwrite(filename, save_img)
st.success(f"🖼️ Annotated image saved as: {filename}")

    