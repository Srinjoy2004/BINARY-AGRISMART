# 🌾 SmartCrop AI - Your Intelligent Crop Advisor

## 📌 Overview
SmartCrop AI is an intelligent agricultural assistant designed to help farmers and agricultural enthusiasts make informed decisions regarding crop selection, fertilizer recommendations, and disease detection. With AI-powered analytics and an interactive chatbot, this tool enhances agricultural productivity by leveraging real-time data and machine learning models.

## ✨ Features
### 🌱 Crop Recommendation System
- Takes in essential nutrients such as **Phosphorus (P), Potassium (K), Nitrogen (N), pH levels**, and environmental factors like **temperature, moisture, and rainfall**.
- Predicts the **top 5 best crops** suitable for the given soil conditions.
- Provides **fertilizer recommendations** based on the selected crop.

### 🍃 Crop Disease Detection
- Accepts **leaf images** of crops as input.
- Detects if the crop is **affected by any disease** using a trained machine learning model.
- Provides insights into the possible disease and preventive measures.

### 🤖 AI Chatbot Assistant
- A conversational chatbot designed to assist users with:
  - **Crop selection & fertilizer recommendations**
  - **Disease identification & treatment suggestions**
  - **General agricultural guidance**

### 📊 Interactive Dashboard
- **Real-time Soil Nutrition Chart** *(Future update after hardware integration)*
- **Region-wise Crop Recommendation System**
- **General Crop Nutrition Recommendation System**

## 🚀 Tech Stack
- **Frontend:** React.js, Tailwind CSS
- **Backend:** Flask (Python), FastAPI
- **Machine Learning Models:** TensorFlow, Scikit-learn
- **Database:** Firebase, PostgreSQL
- **Chatbot:** OpenAI GPT

## 🛠 Installation & Setup
### 🔧 Prerequisites
Ensure you have the following installed:
- Python 3.x
- Node.js & npm
- Virtual environment setup (recommended)

### ⚙️ Backend Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/SmartCropAI.git
cd SmartCropAI/backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the backend
python app.py
```

### 🌍 Frontend Setup
```bash
cd ../frontend
npm install
npm start
```

## 📸 Usage
- **Step 1:** Enter soil nutrient values and environmental conditions.
- **Step 2:** Get recommended crops and fertilizers.
- **Step 3:** Upload crop leaf images to check for diseases.
- **Step 4:** Interact with the AI chatbot for expert guidance.
- **Step 5:** View insights and reports on the dashboard.

## 🤝 Contribution
We welcome contributions to improve SmartCrop AI! Follow these steps:
1. Fork the repository.
2. Create a new branch (`feature/your-feature`).
3. Commit your changes and push them.
4. Submit a pull request.

## 📄 License
This project is licensed under the **MIT License**.

## 🌟 Acknowledgments
Thanks to the contributors and the open-source community for their support in developing this project!

🚀 **Let's revolutionize agriculture with AI!** 🚀
