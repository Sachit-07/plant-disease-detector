import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9aa0a6;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #1e2530;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌿 Plant Disease Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a leaf image to detect the disease and get AI-powered care advice.</p>', unsafe_allow_html=True)

class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Cleaner display names (underscores -> spaces, double underscores handled)
def clean_label(label):
    return label.replace("___", " - ").replace("__", " ").replace("_", " ")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/plant_disease_model.keras")

model = load_model()

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.3])

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded leaf", use_container_width=True)
        predict_clicked = st.button("🔍 Predict", use_container_width=True)

    with col2:
        if predict_clicked:
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)[0]
            predicted_index = np.argmax(predictions)
            predicted_class = class_names[predicted_index]
            confidence = predictions[predicted_index] * 100

            if confidence < 60:
                st.warning(f"⚠️ Low confidence ({confidence:.2f}%). This may not be a clear leaf image from a supported plant (Tomato, Potato, or Pepper). Try a clearer, closer photo of a single leaf.")
            else:
                is_healthy = "healthy" in predicted_class.lower()
                icon = "✅" if is_healthy else "🦠"

                st.markdown(f"""
                    <div class="result-card">
                        <h3>{icon} {clean_label(predicted_class)}</h3>
                    </div>
                """, unsafe_allow_html=True)

                st.progress(int(confidence))
                st.caption(f"Confidence: {confidence:.1f}%")

                with st.spinner("Getting AI explanation..."):
                    gemini_model = genai.GenerativeModel("gemini-3.6-flash")
                    prompt = f"""The plant disease detection model predicted: {predicted_class}.
                    Explain in simple, beginner-friendly terms:
                    1. What this disease/condition is
                    2. Common symptoms
                    3. General prevention tips
                    4. General care recommendations
                    Keep the response concise, under 150 words."""
                    response = gemini_model.generate_content(prompt)

                st.markdown("#### 🤖 AI Explanation")
                with st.container(border=True):
                    st.markdown(response.text)