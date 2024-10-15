from flask import Flask, request, jsonify
from flask_cors import CORS
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Cargar el modelo ONNX
onnx_model = onnx.load("app.onnx")
ort_session = ort.InferenceSession("app.onnx")

def preprocess_image(image):
    image = image.resize((224, 224))  # Cambia el tamaño si es necesario
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess_image(image)

        outputs = ort_session.run(None, {onnx_model.graph.input[0].name: image})
        predictions = outputs[0].argmax(axis=1).tolist()
        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

