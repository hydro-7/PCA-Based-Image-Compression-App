import os
from flask import Flask, render_template, request, redirect, url_for, send_file, session
import numpy as np
import cv2
from sklearn.decomposition import PCA

app = Flask(__name__)
app.secret_key = 'secret_key'  # Required for session

UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def compress_image_pca(image, num_components=50):
    r, g, b = cv2.split(image)
    r, g, b = r / 255, g / 255, b / 255

    pca_r, pca_g, pca_b = PCA(n_components=num_components), PCA(n_components=num_components), PCA(n_components=num_components)
    nr, ng, nb = pca_r.fit_transform(r), pca_g.fit_transform(g), pca_b.fit_transform(b)

    r, g, b = pca_r.inverse_transform(nr), pca_g.inverse_transform(ng), pca_b.inverse_transform(nb)
    final_image = (cv2.merge((r, g, b)))
    final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)

    return final_image

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    return render_template('upload.html')

@app.route('/loading', methods=['POST'])
def loading():
    file = request.files['image']
    compression_level = int(request.form['compression_level'])

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.png')
    file.save(image_path)

    session['compression_level'] = compression_level
    session['image_path'] = image_path

    return render_template('loading.html')

@app.route('/download_file')
def download_file():
    image_path = session.get('image_path')
    compression_level = session.get('compression_level')

    if not image_path or not compression_level:
        return redirect(url_for('upload_image'))

    image = cv2.imread(image_path)
    compressed_image = compress_image_pca(image, num_components=compression_level)

    compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed.png')
    cv2.imwrite(compressed_path, compressed_image)

    return render_template('download.html')

@app.route('/serve_file')
def serve_file():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed.png')
    return send_file(file_path, as_attachment=True, download_name='compressed.png')

if __name__ == '__main__':
    app.run(debug=True)
