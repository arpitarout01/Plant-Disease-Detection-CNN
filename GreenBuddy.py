
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
        if r.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
        else:
            st.error("❌ Failed to download the model file.")
            st.stop()
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()
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

disease_info = {
    'Apple___Apple_scab': {
        'description': 'A fungal disease causing olive-green to black spots on leaves.',
        'treatment': 'Use fungicides and prune infected leaves.'
    },
    'Potato___Late_blight': {
        'description': 'A serious disease causing dark lesions on leaves and tubers.',
        'treatment': 'Remove infected plants and use disease-resistant varieties.'
    },
    'Tomato___Early_blight': {
        'description': 'Causes dark spots and concentric rings on older leaves.',
        'treatment': 'Apply copper-based fungicides and rotate crops.'
    },
    'Tomato___healthy': {
        'description': 'No disease detected in the leaf.',
        'treatment': 'Continue with good farming practices!'
    },
    'Apple___Black_rot':{
        'description': 'Fungal disease causing reddish-brown leaf spots and black fruit rot.',
        'treatment': 'Manage with sanitation and fungicides.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Fungal disease needing cedar hosts, causing yellow-orange leaf spots.',
        'treatment': 'Manage by removing cedars and using fungicides.'
    },
    'Apple___healthy': {
        'description': 'No disease detected.',
        'treatment': 'No treatment needed, continue with good practices.'
    },
    'Blueberry___healthy': {
        'description': 'No disease detected.',
        'treatment': 'No treatment needed, continue with good practices.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'description': 'White powdery growth on leaves and shoots.',
        'treatment': 'Manage with good air circulation and fungicides.'
    },
    'Cherry_(including_sour)___healthy': {
        'description': 'No disease detected.',
        'treatment': 'No treatment needed, continue with good practices.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': 'Elongated tan to gray leaf lesions.',
        'treatment': 'Manage with resistant varieties, rotation, and fungicides.'
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Reddish-brown pustules on leaves.',
        'treatment': 'Manage with resistant varieties and fungicides.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Long, elliptical grayish-tan leaf lesions.',
        'treatment': 'Manage with resistant varieties, rotation, and fungicides.'
    },
    'Corn_(maize)___healthy': {
        'description': 'No disease detected.',
        'treatment': 'No treatment needed, continue with good practices.'
    },
    'Grape___Black_rot': {
        'description': 'Reddish-brown leaf spots and hard, black fruit rot.',
        'treatment': 'Manage with sanitation and fungicides.'
    },
    'Grape___Esca_(Black_Measles)': {
        'description': 'Complex fungal disease causing decline with leaf discoloration and berry shrivel.',
        'treatment': 'Focus on prevention and vine health.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': 'Reddish-brown leaf spots with gray centers.',
        'treatment': 'Manage with sanitation and fungicides.'
    },
    'Grape___healthy': {
        'description': 'No disease detected.',
        'treatment': 'No treatment needed, continue with good practices.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
            'description': 'Bacterial disease causing yellowing leaves and green, bitter fruit.',
            'treatment': 'Prevent by controlling psyllids and removing infected trees.'
    },
    'Peach___Bacterial_spot': {
            'description': 'Dark leaf spots and pitted fruit.',
            'treatment': 'Manage with resistant varieties and copper-based sprays.'
    },
    'Peach___healthy': {
        'description': 'No disease detected.',
        'treatment': 'No treatment needed, continue with good practices.'
    },
    'Pepper,_bell___Bacterial_spot': {
            'description': 'Water-soaked leaf spots and raised fruit lesions.',
            'treatment': 'Manage with disease-free seed, rotation, and copper sprays.'
    },
    'Pepper,_bell___healthy': {
        'description': 'No disease detected.',
        'treatment': 'No treatment needed, continue with good practices.'
    },
    'Potato___Early_blight': {
        'description': 'Dark leaf spots with concentric rings.',
        'treatment': 'Manage with rotation, sanitation, and fungicides.'
    },
    'Potato___healthy': {
        'description': 'No disease detected.',
        'treatment': 'No treatment needed, continue with good practices.'
    },
    'Raspberry___healthy': {
        'description': 'No disease detected.',
        'treatment': 'No treatment needed, continue with good practices.'
    },
    'Soybean___healthy': {
        'description': 'No disease detected.',
        'treatment': 'No treatment needed, continue with good practices.'
    },
    'Squash___Powdery_mildew': {
        'description': 'White powdery growth on leaves.',
        'treatment': 'Manage with resistant varieties, good air circulation, and fungicides.'
    },
    'Strawberry___Leaf_scorch': {
        'description': 'Small, dark purple to reddish-brown leaf spots.',
        'treatment': 'Manage with resistant varieties, sanitation, and fungicides.'
    },
    'Strawberry___healthy': {
        'description': 'No disease detected.',
        'treatment': 'No treatment needed, continue with good practices.'
    },
    'Tomato___Bacterial_spot': {
        'description': 'Water-soaked leaf spots and raised fruit lesions.',
        'treatment': 'Manage with disease-free seed, rotation, and copper sprays.'
    },
    'Tomato___Late_blight': {
        'description': 'Water-soaked leaf spots and fruit rot.',
        'treatment': 'Manage with disease-free seed, sanitation, and fungicides.'
    },
    'Tomato___Leaf_Mold': {
        'description': 'Pale leaf spots with grayish-purple mold underneath.',
        'treatment': 'Manage with resistant varieties, air circulation, and fungicides.'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Small leaf spots with gray centers and dark borders.',
        'treatment': 'Manage with sanitation, rotation, and fungicides.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Tiny pests causing yellow stippling and webbing.',
        'treatment': 'Treat with horticultural oil or insecticidal soap.'
    },
    'Tomato___Target_Spot': {
        'description': 'Leaf spots with concentric rings.',
        'treatment': 'Manage with rotation, sanitation, and fungicides.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Viral disease causing curled, yellow leaves and stunted growth.',
        'treatment': 'Prevent by controlling whiteflies and using virus-free transplants.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Viral disease causing mottled leaves and stunted growth.',
        'treatment': 'Prevent with sanitation and aphid control.'
    }
}

# --- Preprocess uploaded image ---
def preprocess_image(image):
    image = image.resize((128, 128))  # Change if needed
    img_array = np.array(image) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha channel
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Streamlit UI ---

uploaded_files = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files :
    for uploaded_file in uploaded_files:
        st.markdown("----")
        image = Image.open(uploaded_file)
        st.image(image, caption='🖼 Uploaded Image', use_column_width=True)
        
        with st.spinner("🔍 Analyzing leaf image..."):
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions)
            st.success(f"Prediction: **{predicted_class}** 🍃")
            st.info(f"🧠 Confidence Score: {confidence:.2%}")

            info = disease_info.get(predicted_class, {
            'description': 'No specific description available.',
            'treatment': 'Please consult an expert or local agri extension officer.'
            })

            st.markdown(f"**📝 Description:** {info['description']}")
            st.markdown(f"**💊 Treatment:** {info['treatment']}")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with 💚 by <a href='https://github.com/arpitarout01' target='_blank'>Arpita Rout</a></p>", unsafe_allow_html=True)
