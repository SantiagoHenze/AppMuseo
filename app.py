from flask import Flask, request, jsonify
from flask_cors import CORS  # Importar CORS
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Configurar CORS
CORS(app)  # Habilitar CORS para todas las rutas

# Cargar el modelo ONNX
onnx_model = onnx.load("app.onnx")
ort_session = ort.InferenceSession("app.onnx")

# Transformaciones necesarias (ajusta según tu modelo)
def preprocess_image(image):
    # Redimensionar y convertir la imagen a tensor
    image = image.resize((224, 224))  # Cambia el tamaño si es necesario
    image = np.array(image).astype(np.float32) / 255.0  # Normalizar
    image = np.transpose(image, (2, 0, 1))  # Cambiar de (H, W, C) a (C, H, W)
    image = np.expand_dims(image, axis=0)  # Añadir dimensión de batch
    return image

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']  # Obtener la imagen del formulario
    image = Image.open(io.BytesIO(file.read()))  # Leer y abrir la imagen
    image = preprocess_image(image)  # Preprocesar la imagen

    # Realizar la inferencia
    outputs = ort_session.run(None, {onnx_model.graph.input[0].name: image})
    
    # Procesar los resultados (ajusta según tu modelo)
    predictions = outputs[0].argmax(axis=1).tolist()  # Obtener la clase más probable
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
