from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import load_model
import os
from PIL import Image
import numpy as np

model = load_model("model_final.h5")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # img = request.form['file']

        # predict = model.predict(img_to_pred(img))

        if 'fname' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['fname']

        # If the user does not select a file, the browser may send an empty part without a filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        # Save the uploaded file to a specific directory
        upload_folder = os.getcwd() + "/static"
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Perform prediction on the uploaded image
        predict = model.predict(img_to_pred(Image.open(file_path)))

        classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']
        
        predicted_name = classes[np.argmax(predict)]

        return render_template('result.html', prediction=predicted_name, path_to_image=file.filename)
    return render_template('index.html')


def img_to_pred(image):
    new_size = (224, 224)
    image = image.resize(new_size)
    image = np.asarray(image)
    image = tf.expand_dims(image, 0)
    return image


if __name__ == "__main__":
    app.run(port=8000, debug=True)
