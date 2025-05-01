
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

st.markdown(
    """
    <style>
    .main {
        background-color: #f4fff5;
    }
    h1 {
        color: #2E8B57;
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>🌿 GreenBuddy</h1>", unsafe_allow_html=True)
st.write("Upload a clear photo of a leaf, and this app will detect the plant disease using a deep learning model trained on plant pathology data. 🍃")

MODEL_URL = 'https://huggingface.co/arpitarout01/Green_Buddy/resolve/main/plant_disease_detection_model.h5'  # Your Google Drive file ID
MODEL_PATH = 'plant_disease_detection_model.h5'

if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading model... (only once)'):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
# --- Load the Keras model ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- Class names (edit as per your model) ---
class_names=['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
# --- Preprocess uploaded image ---
def preprocess_image(image):
    image = image.resize((128, 128))  # Change if needed
    img_array = np.array(image) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha channel
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Streamlit UI ---
st.title("🌿  GreenBuddy ")
st.write("Upload a leaf image and I'll predict the disease!")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner("🔍 Analyzing leaf image..."):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

    st.success(f"Prediction: **{predicted_class}** 🍃")
    st.info(f"🧠 Confidence Score: {confidence:.2%}")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with 💚 by <a href='https://github.com/arpitarout01' target='_blank'>Arpita Rout</a></p>", unsafe_allow_html=True)
