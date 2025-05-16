# **Introduction**

**Principal Component Analysis (PCA)** is a statistical technique used for **dimensionality reduction**. It works by identifying the directions (principal components) in which the data varies the most, and projecting the data onto these directions.

In the context of image compression, an image is treated as numerical data, with each color channel (R, G, B) represented as a matrix. PCA transforms each of these matrices to reduce the number of features (pixels) while preserving the most important visual information. This process works because most of the image's data is stored within a few components. Hence, reducing the number of components doesnt change the quality of the image by a lot but does end up reducing the size of the image file. 


# **Flask App**

### Features

-  Upload an image (JPG/PNG)
-  Choose between 3 compression levels. (25, 50, 75 components)
-  Preview the compressed image
-  Download the result in one click
-  UI with Bootstrap
-  An option to toggle between light and dark mode


### Tech Stack Used

- **Backend**: Flask (Python)
- **Frontend**: HTML, Bootstrap, CSS, JS
- **Image Processing**: OpenCV, NumPy, Scikit-learn (PCA)


### Project Structure

```
pca-image-compressor/
│
├── app.py             
├── static/
│ ├── uploaded.png     # Temporary uploaded image
│ ├── style.css
│ ├── theme.js         
│ ├── wave-icon.jpg    # Favicon image
│ └── compressed.png   # Resulting compressed image
├── templates/
│ ├── upload.html      
│ └── download.html    
└── README.md
```

























# **PCA Steps**

1. The image is split into Red, Green, and Blue channels:

$$Image → R,G,B ∈ ℝ ^{H×W}$$

2. Normalize values to range [0, 1]

$$ [R_{norm} ,  G_{norm} , B_{norm}] = [\frac{R}{255}, \frac{G}{255}, \frac{B}{255}] $$

3. Fit PCA on each channel, i.e, for each channel matrix (X), do the following :
   - Centers the Data
   - Computes the Covarience Matrix
   - Performs Eigen Decomposition
  
$$X ∈ ℝ^{H x W}$$

4. Use the top $k$ eigen vectors to perform projections onto a lower dimension. This gives us a compressed representation as $Z ∈ ℝ ^{H x k}$

$$
Z = \bar{X} V_k 
$$
  

5. Reconstruct the channels using the Inverse Transform as :

$$ \bar{X} = ZV_k^{T} + \mu$$

6. Merge the cchannels and rescale accordingly
   - Combine the approximated channels into one image.
   - Multiply by 255 to scale back to [0, 255].
   - Clip and convert to uint8 for proper image format.


















# **Future Improvements**
- Make an option for svd based compression
- Make an option for grayscale images
- Drag Drop upload
- Compression metric
- Eventual online deployment
  
