import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image to detect the disease and get AI-powered care advice.")

# Class names must be in the SAME order as during training
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/plant_disease_model.keras")

model = load_model()

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    if st.button("Predict"):
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[predicted_index] * 100

        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")