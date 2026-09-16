# 🌿 Plant Disease Detector with AI Explanation

A Streamlit web app that identifies plant leaf diseases from an uploaded image using a MobileNetV2-based CNN, then explains the result using Google's Gemini API — covering what the disease is, symptoms, prevention, and care tips.

## How it works
1. Upload a leaf image (Tomato, Potato, or Pepper)
2. A trained CNN predicts the disease with a confidence score
3. Gemini explains the disease in plain language

## Dataset
[PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) (via Kaggle) — 15 classes, ~20,600 images across Tomato, Potato, and Pepper plants.

## Model
- MobileNetV2 (transfer learning, ImageNet weights, frozen base)
- Trained with class weighting to handle class imbalance
- Final validation accuracy: ~89.85%
- Full training, evaluation, and confusion matrix in `notebooks/plant_disease_detector.ipynb`

## Setup

### 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/plant-disease-detector.git
cd plant-disease-detector


### 2. Python version
**Requires Python 3.9–3.11.** TensorFlow does not yet support Python 3.12+ (including 3.14), so if your default Python is newer, install 3.11 separately and use it specifically for this project's virtual environment:

python3.11 -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt


### 3. Kaggle API (only needed if re-running the training notebook)
The dataset is downloaded via the Kaggle API inside the Colab notebook, not needed to just run the app.
1. Create a free account at [kaggle.com](https://kaggle.com)
2. Go to Account → API → **Create New Token**
3. In Google Colab, click the key icon (🔑) in the sidebar → Add new secret → name it `KAGGLE_TOKEN` → paste your token → enable notebook access
4. Run the notebook cells in order

### 4. Gemini API (needed to run the app)
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a file at `app/.env` containing:
    GEMINI_API_KEY=your_key_here


### 5. Run the app

cd app
streamlit run app.py

Opens at `localhost:8501`.


## Evaluation
- Precision, recall, F1-score per class, and confusion matrix in the notebook
- Notable confusion: Target Spot vs. Spider Mite damage and Early Blight, due to visually similar leaf spotting

## Limitations
- Only recognizes the 15 trained classes (Tomato, Potato, Pepper)
- No dedicated out-of-distribution detection — a confidence threshold (<60%) shows a warning instead of a prediction for unclear/unsupported images, but this is a lightweight safeguard, not a guarantee