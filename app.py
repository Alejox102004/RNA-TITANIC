from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model #type: ignore
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

app = Flask(__name__)

# Cargar el modelo y el scaler
model = load_model('RNA_model.h5')

scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibir parámetros desde el formulario
        Pclass = int(request.form['Pclass'])
        Sex = int(request.form['Sex'])
        Age = float(request.form['Age'])
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = float(request.form['Fare'])
        Embarked = int(request.form['Embarked'])

        # Crear el DataFrame con los datos recibidos
        data = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]],
                            columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

        # Normalizar los datos usando el scaler entrenado
        data_scaled = scaler.transform(data)

        # Realizar la predicción
        prediction = model.predict(data_scaled)
        prediction = (prediction > 0.5).astype(int)  # Convertir en 0 o 1

        # Determinar el mensaje de acuerdo con la predicción
        if prediction == 1:
            result_message = "¡Sobrevivió!"
        else:
            result_message = "No sobrevivió."

        return render_template('index.html', prediction=result_message)

    except KeyError as e:
        return jsonify({'error': f"Faltó el campo {str(e)} en el formulario."}), 400

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Usa el puerto especificado por Railway
    app.run(host='0.0.0.0', port=port, debug=True)
    app.run(debug=True)