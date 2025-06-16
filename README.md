🍇 Wine Quality Classification
A supervised machine learning project to classify red wine quality as Good or Bad based on physicochemical features like acidity, pH, sulphates, and alcohol content.

🎊 Dataset
📍 Source: UCI ML Repository – Wine Quality (Red)
✅ Features: 11 numerical columns (e.g., fixed acidity, pH, alcohol)
🎯 Target: quality (converted to binary: Good / Bad)
🧠 ML Models Used
🌳 Decision Tree Classifier
🌲 Random Forest Classifier
Both trained using scikit-learn with evaluation on test data.

🧪 Evaluation Metrics
✔️ Accuracy
✔️ Precision / Recall / F1-score
✔️ Confusion Matrix
✔️ Feature Importance Plot (from Random Forest)
💾 Model Saving
Trained models are serialized using joblib:

wine_quality_rf_model.pkl
wine_quality_dt_model.pkl
▶️ How to Run This Project
1. Clone this repository
git clone https://github.com/MrCoss/wine_quality_classification.git
cd wine_quality_classification
2. Install dependencies
pip install -r requirements.txt
3. Run the Python script
python wine_quality_classifier.py
🔍 Prediction Example
At the end of the script, a sample from the test dataset is used to predict wine quality using both trained models.

📦 Requirements
Python 3.7+
pandas
scikit-learn
matplotlib
seaborn
joblib
📄 License
This project is open-source and free to use.

Made with ❤️ by MrCoss
