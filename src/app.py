from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Carregar o modelo treinado
model = joblib.load('C:\\Users\\inesm\\OneDrive\\Documentos\\GitHub\\ML_Group36\\src\\xgboost_model.pkl')

# Inicializar o Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Receber dados do request em formato JSON
    data = request.get_json()

    # Converter os dados em um DataFrame para o modelo
    input_data = pd.DataFrame([data])

    # Realizar previsão
    prediction = model.predict(input_data)

    # Retornar a previsão como JSON
    response = {'prediction': int(prediction[0])}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)