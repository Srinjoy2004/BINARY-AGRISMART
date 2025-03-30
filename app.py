from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Load environment variables
load_dotenv()

app = Flask(__name__)

from flask_cors import CORS
CORS(app)


# Setup logging
logging.basicConfig(level=logging.INFO)

# Load models and data
try:
    model = joblib.load("crop_prob_model.pkl")
    label_mapping = joblib.load("crop_label_mapping.pkl")
    fertilizer_df = pd.read_csv("fertilizer.csv").set_index("Crop")
except Exception as e:
    logging.error(f"Error loading model or data: {e}")
    raise

# Configure Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyDD8QW1BggDVVMLteDygHCHrD6Ff9Dy0e8 "
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def gemini_fertilizer_advice(crop, n, p, k, ideal_n, ideal_p, ideal_k):
    def compare(val, ideal, name):
        if val < ideal:
            return f"{name} is low ({val} < {ideal})"
        elif val > ideal:
            return f"{name} is high ({val} > {ideal})"
        else:
            return f"{name} is optimal ({val} = {ideal})"

    n_status = compare(n, ideal_n, "Nitrogen")
    p_status = compare(p, ideal_p, "Phosphorus")
    k_status = compare(k, ideal_k, "Potassium")
    """
    Uses Google's Gemini AI to generate fertilizer recommendations.
    """
    try:
        prompt = f"""
    A farmer wants to grow {crop}.
    Ideal NPK values: Nitrogen: {ideal_n}, Phosphorus: {ideal_p}, Potassium: {ideal_k}.
    Current soil values: Nitrogen: {n} → {n_status}, Phosphorus: {p} → {p_status}, Potassium: {k} → {k_status}.
    Based on this, suggest what nutrients are deficient or excessive and recommend suitable fertilizers.
    Keep it concise within 50 words. Use Indian agricultural context. No headings or markdown.
    """
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "Could not generate fertilizer advice. Try again later."
    except Exception as e:
        logging.error(f"Gemini API Error: {e}")
        return "Fertilizer recommendation is unavailable at the moment."
        return {e}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        required_keys = {"N", "P", "K", "temperature", "humidity", "pH", "rainfall"}

        # Validate input data
        missing_keys = required_keys - data.keys()
        if missing_keys:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_keys)}"}), 400

        # Extract input features
        features = np.array([[data["N"], data["P"], data["K"], data["temperature"], 
                              data["humidity"], data["pH"], data["rainfall"]]])
        
        # Predict probabilities
        probs = model.predict_proba(features)[0]
        
        # Get Top-5 recommended crops
        top5_indices = np.argsort(probs)[::-1][:5]
        top5 = [{"crop": label_mapping[i], "probability": round(probs[i] * 100, 2)} for i in top5_indices]

        return jsonify({
            "recommendations": [crop["crop"] for crop in top5]
        })
    
    except Exception as e:
        logging.error(f"Recommendation Error: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

@app.route('/fertilizer', methods=['POST'])
def fertilizer():
    """
    Receives the selected crop from the frontend and returns fertilizer advice.
    """
    try:
        # Check if request contains JSON
        if not request.is_json:
            logging.error("Invalid request format: Expected JSON")
            return jsonify({"error": "Invalid request format, expected JSON"}), 400

        data = request.json
        selected_crop = data.get("crop")
        n, p, k = data.get("N"), data.get("P"), data.get("K")

        # Validate input fields
        if not selected_crop:
            logging.error("Missing crop selection")
            return jsonify({"error": "Crop selection is required"}), 400
        if None in [n, p, k]:
            logging.error("Missing NPK values")
            return jsonify({"error": "Soil NPK values (N, P, K) are required"}), 400

        # Ensure fertilizer_df is available
        if "fertilizer_df" not in globals():
            logging.error("Fertilizer data (fertilizer_df) is not loaded")
            return jsonify({"error": "Fertilizer data is not loaded"}), 500

        # Convert crop names to lowercase to avoid case sensitivity issues
        selected_crop = selected_crop.strip().lower()
        fertilizer_df.index = fertilizer_df.index.str.strip().str.lower()

        # Check if crop exists in DataFrame
        if selected_crop not in fertilizer_df.index:
            logging.error(f"No fertilizer data found for crop: {selected_crop}")
            return jsonify({"error": f"No fertilizer data available for {selected_crop}"}), 404

        # Fetch ideal NPK values for the selected crop
        ideal_n, ideal_p, ideal_k = fertilizer_df.loc[selected_crop, ["N", "P", "K"]]

        # Ensure the function exists
        if "gemini_fertilizer_advice" not in globals():
            logging.error("Missing function: gemini_fertilizer_advice")
            return jsonify({"error": "Fertilizer advice function is missing"}), 500

        # Get fertilizer recommendation
        fertilizer_advice = gemini_fertilizer_advice(selected_crop, n, p, k, ideal_n, ideal_p, ideal_k)

        logging.info(f"Fertilizer advice for {selected_crop}: {fertilizer_advice}")
        return jsonify({"fertilizer_advice": fertilizer_advice})

    except Exception as e:
        logging.error(f"Unexpected error in /fertilizer route: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your request."}), 500
    
# app.config['UPLOAD_FOLDER'] = 'uploads/'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dictionary mapping crops to their respective model paths
crop_models = {
    "potato": "models/potato_disease_model.h5",
    "apple": "models/apple_disease_model.h5",
    "corn": "models/corn_disease_model.h5",
    "grape": "models/grape_disease_model.h5",
    "cherry": "models/cherry_disease_model.h5",
}

# Define class labels for each crop
class_labels = {
    "potato": {0: "Potato Early Blight", 1: "Potato Late Blight", 2: "Potato Healthy"},
    "apple": {0: "Apple Scab", 1: "Apple Black Rot", 2: "Apple Cedar Rust", 3: "Apple Healthy"},
    "corn": {0: "Corn Cercospora", 1: "Corn Common Rust", 2: "Corn Leaf Blight", 3: "Corn Healthy"},
    "grape": {0: "Grape Black Rot", 1: "Grape Esca", 2: "Grape Leaf Blight", 3: "Grape Healthy"},
    "cherry": {0: "Cherry Powdery Mildew", 1: "Cherry Healthy"},
}

# Cache to store loaded models
loaded_models = {}

def load_model_for_crop(crop_name):
    """Loads the model for a specific crop."""
    if crop_name not in loaded_models:
        model_path = crop_models.get(crop_name)
        if model_path:
            print(f"Loading model for {crop_name} from {model_path}...")
            loaded_models[crop_name] = tf.keras.models.load_model(model_path)
        else:
            raise ValueError(f"No model found for crop: {crop_name}")
    return loaded_models[crop_name]

def preprocess_image(image_path):
    """Prepares the image for model prediction."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    image = cv2.resize(image, (224, 224))  
    image = np.expand_dims(image, axis=0)  
    return image

def predict_image(image_path, crop_name):
    """Predicts disease class for the given image."""
    image = preprocess_image(image_path)
    model = load_model_for_crop(crop_name)
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    predicted_class = class_labels[crop_name][class_index]
    return predicted_class, confidence

@app.route('/predictdisease', methods=['POST'])
def predict():
    """Handles image upload and disease prediction."""
    if 'image' not in request.files or 'crop' not in request.form:
        return jsonify({'error': 'Image and crop type required'}), 400

    file = request.files['image']
    crop_name = request.form['crop'].lower()
    
    if crop_name not in crop_models:
        return jsonify({'error': 'Invalid crop type'}), 400
    
    # Save the uploaded file temporarily
    temp_path = "temp_upload.jpg"
    file.save(temp_path)
    
    try:
        predicted_class, confidence = predict_image(temp_path, crop_name)
        response = {
            'crop': crop_name,
            'disease': predicted_class,
            'confidence': f"{confidence:.2%}"
        }
    except Exception as e:
        response = {'error': str(e)}
    
    # Clean up temporary file
    os.remove(temp_path)
    
    return jsonify(response)   

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# Load trained model and column names
model = pickle.load(open("prediction_model.pkl", "rb"))
training_columns = list(pd.read_csv("encoded_feature_columns.csv").columns)

# Validate if the loaded model is valid
if not isinstance(model, (RandomForestRegressor, DecisionTreeRegressor)):
    raise TypeError("Loaded model is not a valid RandomForestRegressor or DecisionTreeRegressor. Please check the model file.")

# Load dataset and extract unique values for dropdowns
apy_df = pd.read_csv("APY.csv", encoding="utf-8")
apy_df.rename(columns=lambda x: x.strip(), inplace=True)
states = sorted(apy_df['State'].dropna().unique().tolist())
districts = sorted(apy_df['District'].dropna().unique().tolist())
crops = sorted(apy_df['Crop'].dropna().unique().tolist())
seasons = sorted(apy_df['Season'].dropna().unique().tolist())

# Prediction function
def predict_production(state, district, crop, season, area, year):
    if area <= 0:
        return "Error: Area must be greater than zero."
    if year < 2000 or year > 2050:
        return "Error: Year should be within a reasonable range."

    row = pd.DataFrame({
        'State': [state],
        'District': [district],
        'Crop_Year': [year],
        'Season': [season],
        'Crop': [crop],
        'Area': [area]
    })
    
    # Encode categorical variables
    row_encoded = pd.get_dummies(row)
    
    # Ensure all required columns exist
    for col in training_columns:
        if col not in row_encoded.columns:
            row_encoded[col] = 0
    
    # Reorder columns to match training data
    row_encoded = row_encoded[training_columns]

    # Ensure model is valid before prediction
    if not isinstance(model, (RandomForestRegressor, DecisionTreeRegressor)):
        return "Error: Model is not properly loaded. Please check the model file."
    
    # Make prediction
    pred = model.predict(row_encoded)[0]
    return f"Predicted Production: {pred:.2f} tonnes"



@app.route('/predict-price', methods=['POST'])
def predict_price():
    data = request.json
    state = data['state']
    district = data['district']
    crop = data['crop']
    season = data['season']
    area = float(data['area'])
    year = int(data['year'])

    prediction = predict_production(state, district, crop, season, area, year)
    return jsonify({"prediction": prediction})


if __name__ == '__main__':
    app.run(debug=True)