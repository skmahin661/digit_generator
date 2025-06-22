# app.py

from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO

app = Flask(__name__)
model = load_model("generator_model.h5")
latent_dim = 100

def generate_images(digit, num=5):
    noise = np.random.normal(0, 1, (num, latent_dim))
    labels = np.full((num, 1), digit)
    gen_imgs = model.predict([noise, labels])
    gen_imgs = 0.5 * gen_imgs + 0.5

    images = []
    for img in gen_imgs:
        fig = plt.figure(figsize=(1, 1))
        plt.axis('off')
        plt.imshow(img.squeeze(), cmap='gray')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)
        images.append(encoded)
    return images

@app.route('/', methods=['GET', 'POST'])
def index():
    images = []
    digit = None
    if request.method == 'POST':
        digit = int(request.form['digit'])
        images = generate_images(digit)
    return render_template('index.html', images=images, digit=digit)

if __name__ == '__main__':
    app.run(debug=True)
