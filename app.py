from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_images'

model = load_model('transfer_resnet.keras')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((320, 320))  # Match your model's input size
    image = np.array(image) / 255.0  # Normalize if required
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        image = preprocess_image(file_path)
        prediction = model.predict(image)
        result = "Cancerous" if prediction[0][0] > 0.5 else "Non-Cancerous"  # Adjust based on model output
        return render_template('result.html', result=result, image_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
