import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import numpy as np
import cv2
from sklearn.decomposition import PCA

app = Flask(__name__)
os.makedirs('static', exist_ok=True)  
app.config['UPLOAD_FOLDER'] = 'static'

def compress_image_pca(image, num_components = 50):
    r, g, b = cv2.split(image)
    r, g, b = r/255, g/255, b/255
    pca_r, pca_g, pca_b = PCA(n_components=num_components), PCA(n_components=num_components), PCA(n_components=num_components)
    nr, ng, nb = pca_r.fit_transform(r), pca_g.fit_transform(g), pca_b.fit_transform(b)

    r, g, b = pca_r.inverse_transform(nr), pca_g.inverse_transform(ng), pca_b.inverse_transform(nb)
    final_image = (cv2.merge((r, g, b)))
    final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)

    return final_image


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.png')
            file.save(filepath)

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            compressed = compress_image_pca(image)

            compressed_bgr = cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR)
            compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed.png')
            cv2.imwrite(compressed_path, compressed_bgr)

            return redirect(url_for('download_image'))
    return render_template('upload.html')

@app.route('/download')
def download_image():
    return render_template('download.html')

@app.route('/download_file')
def download_file():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'compressed.png'), as_attachment=True, download_name='compressed.png')

if __name__ == '__main__':
    app.run(debug=True)