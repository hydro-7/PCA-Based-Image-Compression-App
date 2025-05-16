import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, session
from PIL import Image
from sklearn.decomposition import PCA

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def convert_to_jpg(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ['.jpg', '.jpeg']:
        im = Image.open(image_path).convert("RGB")
        new_path = os.path.splitext(image_path)[0] + ".jpg"
        im.save(new_path, "JPEG")
        os.remove(image_path)
        return new_path
    return image_path

# Fourier Transform
def compress_channel(channel_data, keep_ratio=0.05):
    f_transform = np.fft.fft2(channel_data)
    f_shifted = np.fft.fftshift(f_transform)

    magnitude = np.abs(f_shifted)
    threshold = np.percentile(magnitude, (1 - keep_ratio) * 100)
    mask = magnitude > threshold
    f_compressed = f_shifted * mask

    f_ishifted = np.fft.ifftshift(f_compressed)
    img_reconstructed = np.fft.ifft2(f_ishifted)
    return np.abs(img_reconstructed)


def compress_image_fourier(image):

    keep_ratio = 0.05  # Keep top 5% frequencies
    compressed_channels = []
    img_array = np.array(image)

    for i in range(3):  # R, G, B
        channel = img_array[:, :, i]
        compressed = compress_channel(channel, keep_ratio)
        compressed_channels.append(compressed)

    compressed_img = np.stack(compressed_channels, axis=2)
    image = np.clip(compressed_img, 0, 255).astype(np.uint8)

    return image


def compress_image_pca(image, num_components=50):

    r, g, b = cv2.split(image)
    r, g, b = r / 255, g / 255, b / 255

    pca_r, pca_g, pca_b = PCA(n_components=num_components), PCA(n_components=num_components), PCA(n_components=num_components)
    nr, ng, nb = pca_r.fit_transform(r), pca_g.fit_transform(g), pca_b.fit_transform(b)

    r, g, b = pca_r.inverse_transform(nr), pca_g.inverse_transform(ng), pca_b.inverse_transform(nb)
    final_image = (cv2.merge((r, g, b)))
    final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)

    quality = 10
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', final_image, encode_param)
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

    return decoded_img

def compress_image_svd(image, num_components=50):
    b, g, r = cv2.split(image)

    U_R, S_R, Vt_R = np.linalg.svd(r, full_matrices=False)
    U_G, S_G, Vt_G = np.linalg.svd(g, full_matrices=False)
    U_B, S_B, Vt_B = np.linalg.svd(b, full_matrices=False)

    n = num_components  # rank approximation parameter
    R_compressed = np.matrix(U_R[:, :n]) * np.diag(S_R[:n]) * np.matrix(Vt_R[:n, :])
    G_compressed = np.matrix(U_G[:, :n]) * np.diag(S_G[:n]) * np.matrix(Vt_G[:n, :])
    B_compressed = np.matrix(U_B[:, :n]) * np.diag(S_B[:n]) * np.matrix(Vt_B[:n, :])

    compressed_image = cv2.merge([np.clip(R_compressed, 1, 255), np.clip(G_compressed, 1, 255), np.clip(B_compressed, 1, 255)])
    compressed_image = compressed_image.astype(np.uint8)

    return compressed_image

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    return render_template('upload.html')

@app.route('/loading', methods=['POST'])
def loading():
    file = request.files['image']
    compression_level = int(request.form['compression_level'])

    filename = f"uploaded_{uuid.uuid4().hex}.png"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    # Convert to JPG if not already
    jpg_path = convert_to_jpg(image_path)

    session['compression_level'] = compression_level
    session['image_path'] = jpg_path

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
    return send_file(file_path, as_attachment=True, download_name='compressed.jpg')

if __name__ == '__main__':
    app.run(debug=True)
