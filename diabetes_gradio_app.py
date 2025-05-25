import gradio as gr
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load model
model = load_model("diabetes_model.h5")

# Create and fit scaler (based on sample of Pima Indians dataset)
data = pd.DataFrame([
    [6,148,72,35,0,33.6,0.627,50,1],
    [1,85,66,29,0,26.6,0.351,31,0],
    [8,183,64,0,0,23.3,0.672,32,1],
    [1,89,66,23,94,28.1,0.167,21,0],
    [0,137,40,35,168,43.1,2.288,33,1],
    [5,116,74,0,0,25.6,0.201,30,0],
    [3,78,50,32,88,31.0,0.248,26,1],
    [10,115,0,0,0,35.3,0.134,29,0],
    [2,197,70,45,543,30.5,0.158,53,1],
    [8,125,96,0,0,0.0,0.232,54,1],
], columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
            "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"])
scaler = StandardScaler()
scaler.fit(data.drop("Outcome", axis=1))

# Prediction function
def predict(pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    result = "🩸 Diabetic" if prediction > 0.5 else "✅ Not Diabetic"
    return f"{result} (Probability: {prediction:.2f})"

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Pregnancies", value=1),
        gr.Number(label="Glucose", value=100),
        gr.Number(label="Blood Pressure", value=70),
        gr.Number(label="Skin Thickness", value=20),
        gr.Number(label="Insulin", value=80),
        gr.Number(label="BMI", value=25.0),
        gr.Number(label="Diabetes Pedigree Function", value=0.5),
        gr.Number(label="Age", value=30),
    ],
    outputs="text",
    title="🩺 Diabetes Prediction App",
    description="Enter patient health data to predict diabetes risk using a trained neural network model."
)

interface.launch()
