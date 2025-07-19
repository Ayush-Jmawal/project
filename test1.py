# 🏥 Hospital Case Simulation: Disease Detection using ML (Diabetes Prediction)

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hospital Diabetes Detection", layout="centered")

# Title
st.title("🏥 Hospital Disease Detection (ML-powered Diabetes Prediction)")

# Step 1: Load Dataset
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv("diabetes.csv", header=None, names=columns)

# Show raw data
if st.checkbox("🔍 Show Raw Dataset"):
    st.dataframe(df.head())

# Step 2: Visualizations
if st.checkbox("📊 Show Heatmap of Feature Correlations"):
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Step 3: Preprocess and Train Model
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Show model performance
y_pred = model.predict(X_test)
st.write("📈 **Model Accuracy:**", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# Optional: Show classification report
if st.checkbox("🧾 Show Classification Report"):
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# Step 5: Patient Input Form
st.header("🧪 Enter Patient Data for Prediction")

preg = st.number_input("Pregnancies", 0, 20, 2)
gluc = st.number_input("Glucose", 0, 300, 140)
bp = st.number_input("Blood Pressure", 0, 200, 75)
skin = st.number_input("Skin Thickness", 0, 100, 30)
insulin = st.number_input("Insulin", 0, 1000, 0)
bmi = st.number_input("BMI", 0.0, 70.0, 35.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.6)
age = st.number_input("Age", 1, 100, 34)

if st.button("🔍 Predict"):
    sample_patient = np.array([[preg, gluc, bp, skin, insulin, bmi, dpf, age]])
    sample_scaled = scaler.transform(sample_patient)
    prediction = model.predict(sample_scaled)[0]

    if prediction == 1:
        st.error("🟥 The patient is likely DIABETIC.")
    else:
        st.success("🟩 The patient is NOT diabetic.")
