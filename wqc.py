# wine_quality_classifier.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load Data
print("Loading data...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Step 2: Preprocess
df['quality_label'] = df['quality'].apply(lambda x: 'Good' if x >= 7 else 'Bad')
df.drop('quality', axis=1, inplace=True)

# Step 3: Feature/Target Split
X = df.drop('quality_label', axis=1)
y = df['quality_label']

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Decision Tree
print("Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# Step 6: Train Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Step 7: Evaluation
print("\n Decision Tree Report:\n", classification_report(y_test, dt_preds))
print("Random Forest Report:\n", classification_report(y_test, rf_preds))

print("\n Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_preds))

print("\n Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# Step 8: Feature Importance (Random Forest)
print("\n Plotting Feature Importances...")
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Step 9: Save Models
print("\nSaving models...")
joblib.dump(rf_model, "wine_quality_rf_model.pkl")
joblib.dump(dt_model, "wine_quality_dt_model.pkl")
print("Models saved as 'wine_quality_rf_model.pkl' and 'wine_quality_dt_model.pkl'")

# Step 10: Load Models & Predict
print("\nLoading models for prediction...")
rf_loaded = joblib.load("wine_quality_rf_model.pkl")
dt_loaded = joblib.load("wine_quality_dt_model.pkl")

# Predict on a sample wine record (first record from test set)
sample = X_test.iloc[0]
print("\nSample wine features:")
print(sample)

rf_prediction = rf_loaded.predict([sample])[0]
dt_prediction = dt_loaded.predict([sample])[0]

print("\nRandom Forest Prediction:", rf_prediction)
print("Decision Tree Prediction:", dt_prediction)
print("Actual label:", y_test.iloc[0])
