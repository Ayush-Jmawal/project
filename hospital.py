# 🏥 Hospital Case Simulation: Disease Detection using ML (Diabetes Prediction)

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the dataset from local system
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv("diabetes.csv", header=None, names=columns)

# Step 3: Explore the data
print("🩺 First 5 rows of data:")
print(df.head())
print("\n📊 Class distribution:")
print(df['Outcome'].value_counts())

# Step 4: Visualize data
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 5: Preprocess data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate model
y_pred = model.predict(X_test)
print("📈 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Simulate hospital input
# Example patient: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
sample_patient = np.array([[2, 140, 75, 30, 0, 35.0, 0.6, 34]])
sample_scaled = scaler.transform(sample_patient)
prediction = model.predict(sample_scaled)

print("\n🔍 Prediction for simulated patient:")
print("🟥 DIABETIC" if prediction[0] == 1 else "🟩 NOT DIABETIC")
