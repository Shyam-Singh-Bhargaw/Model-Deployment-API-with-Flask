from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Path to the saved model
model_path = 'C:\\Users\\thesh\\OneDrive\\Desktop\\jupyter\\pneumonia_detection_model.keras'
model = tf.keras.models.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if not file or not file.filename.endswith(('jpg', 'jpeg', 'png')):
        return jsonify({'error': 'Invalid file format'}), 400

    img = image.load_img(file, target_size=(128, 128), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = 'Pneumonia' if predictions[0][0] > 0.5 else 'Normal'

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
