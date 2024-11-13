from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
model = load_model('transfer_resnet.keras')
path = '/home/aman/Codes/PROJECT/Major_Project/archive/test/1/729_748167281_png.rf.496a8581ecdbc0dfd0ffbb1affc2a1d1.jpg'
image = Image.open(path)

image = image.resize((320, 320))  # Match your model's input size
image = np.array(image) / 255.0  # Normalize if required
image = np.expand_dims(image, axis=0)

prediction = model.predict(image)
print(prediction)
