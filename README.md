# 🩺 Diabetes Prediction App (FYP)

A machine learning web application that predicts diabetes risk using a trained neural network on the Pima Indians Diabetes dataset.

## Features
- 8 health parameter inputs (Pregnancies, Glucose, BP, Skin Thickness, Insulin, BMI, DPF, Age)
- Real-time prediction with probability score
- Gradio web interface

## Tech Stack
- Python 3
- TensorFlow / Keras (model)
- Scikit-learn (preprocessing)
- Gradio (UI)

## Model
- Neural network trained on Pima Indians Diabetes dataset (768 samples, 8 features)
- Input features standardized with StandardScaler
- Binary classification (Diabetic / Not Diabetic)

## How to Run
```bash
pip install -r requirements.txt
python diabetes_gradio_app.py
