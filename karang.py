from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Aktifkan CORS setelah app dibuat


# Load model dengan format .keras
model = tf.keras.models.load_model('https://github.com/anggitaac/terumbukarang_api/releases/download/karang/karang.keras')

# Fungsi untuk memproses gambar
def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).resize((150, 150))  # Resize gambar ke ukuran yang sesuai dengan model
    image = np.array(image) / 255.0  # Normalisasi gambar ke rentang [0, 1]
    return np.expand_dims(image, axis=0)  # Menambah dimensi untuk batch size

# Fungsi untuk mendecode hasil prediksi
def decode_prediction(prediction):
    labels = ["Bleaching", "Dead", "Healthy"]  # Label yang sesuai dengan kelas model Anda
    predicted_index = np.argmax(prediction)  # Dapatkan indeks dengan probabilitas tertinggi
    return labels[predicted_index]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil file gambar dari request
        file = request.files['image']
        
        # Memproses gambar dan prediksi
        image_data = file.read()  # Membaca data gambar
        image = preprocess_image(image_data)  # Memproses gambar
        prediction = model.predict(image)  # Membuat prediksi dengan model
        
        # Decode prediksi menjadi label yang lebih mudah dipahami
        label = decode_prediction(prediction)
        
        return jsonify({'prediction': label})  # Mengembalikan label hasil prediksi
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)


