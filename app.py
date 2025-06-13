import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import logging
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Configurar credenciales desde variable de entorno
credentials_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_CONTENT')
if not credentials_json:
    raise ValueError("La variable de entorno GOOGLE_APPLICATION_CREDENTIALS_CONTENT no está configurada")
with open('brainmriapp-credentials.json', 'w') as f:
    f.write(credentials_json)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'brainmriapp-credentials.json'

# Autenticar con Google Drive
try:
    credentials = service_account.Credentials.from_service_account_file('brainmriapp-credentials.json')
    drive_service = build('drive', 'v3', credentials=credentials)
except Exception as e:
    logging.error(f"Error al autenticar con Google Drive: {e}")
    raise

# Función para descargar archivos desde Drive solo si no existen
def download_file_if_not_exists(file_id, destination):
    if not os.path.exists(destination):
        try:
            request = drive_service.files().get_media(fileId=file_id)
            with open(destination, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    logging.info(f"Descargando {destination}: {int(status.progress() * 100)}%")
        except Exception as e:
            logging.error(f"Error al descargar archivo {file_id}: {e}")
            raise
    else:
        logging.info(f"El archivo {destination} ya existe, no se descargará nuevamente.")

# Descargar modelos al iniciar la aplicación solo si no existen
MODEL_JSON_ID = '1eRuupBoFuEB-VfhewOiS_SVEaTA2AgV8'  # ID del archivo JSON
WEIGHTS_ID = '1fv7XRHe9WsbZEEunX7V_WOpa4WgP6Wjt'  # ID del archivo de pesos
download_file_if_not_exists(MODEL_JSON_ID, 'resnet-50-MRI.json')
download_file_if_not_exists(WEIGHTS_ID, 'weights.hdf5')

# Cargar el modelo
try:
    with open('resnet-50-MRI.json', 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights('weights.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    logging.info("Modelo cargado exitosamente")
except Exception as e:
    logging.error(f"Error al cargar el modelo: {e}")
    raise

# Ruta principal para la interfaz
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logging.error("No se proporcionó archivo en request.files")
        return jsonify({'error': 'No se proporcionó archivo'}), 400
    file = request.files['file']
    try:
        # Leer la imagen en color (RGB)
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logging.error("No se pudo decodificar la imagen")
            return jsonify({'error': 'Archivo de imagen inválido'}), 400
        # Redimensionar a 256x256
        img_resized = cv2.resize(img, (256, 256))
        # Normalizar para la predicción
        img_input = img_resized / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        logging.info(f"Forma de la imagen procesada: {img_input.shape}")
        
        # Hacer la predicción con el modelo
        prediction = model.predict(img_input)
        logging.info(f"Predicción cruda: {prediction}")
        # Suponiendo que 0 = no tumor, 1 = tumor
        has_tumor = np.argmax(prediction[0])
        confidence = float(prediction[0][has_tumor]) * 100
        result = 'Tumor detectado' if has_tumor else 'No se detectó tumor'
        logging.info(f"Resultado: {result}, Confianza: {confidence}, has_tumor: {has_tumor}")

        # Convertir la imagen original a base64
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_original_base64 = base64.b64encode(buffered.getvalue()).decode()
        img_original_data = f"data:image/png;base64,{img_original_base64}"
        logging.info("Imagen original generada correctamente")

        # Si hay tumor, generar una versión con superposición roja
        img_tumor_data = ""
        if has_tumor:
            try:
                # Crear una copia de la imagen y aplicar superposición roja
                img_tumor = img_rgb.copy()
                # Crear una capa roja translúcida
                overlay = Image.new('RGBA', img_pil.size, (255, 0, 0, 128))  # Alfa 128 para visibilidad
                img_tumor_pil = Image.blend(img_pil.convert('RGBA'), overlay, 0.3)
                # Convertir a base64
                buffered = BytesIO()
                img_tumor_pil.save(buffered, format="PNG")
                img_tumor_base64 = base64.b64encode(buffered.getvalue()).decode()
                img_tumor_data = f"data:image/png;base64,{img_tumor_base64}"
                logging.info("Imagen de tumor generada correctamente")
            except Exception as e:
                logging.error(f"Error generando img_tumor: {e}")
                img_tumor_data = ""

        # Preparar y loguear la respuesta
        response = {
            'result': result,
            'confidence': confidence,
            'img_original': img_original_data,
            'img_tumor': img_tumor_data
        }
        logging.info(f"Respuesta enviada: {response}")
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error en la predicción: {e}")
        return jsonify({'error': 'Error al procesar la imagen'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
