from flask import Flask, request, jsonify
from app.torch_utils import transform_image, get_prediction
from flask import render_template, flash, redirect, url_for
import os
import io
from PIL import Image

app = Flask(__name__)

image_folder = os.path.join('static', 'saved_images')
app.config['UPLOAD_FOLDER'] = image_folder

ALLOWED_EXTENSIONS = {'png','jpg', 'jpeg'}
cifar10_dict = {0:'plane', 1:'automob', 2:'bird', 3:'cat',4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        if file is None or file.filename == "":
            data = {'error':'no file'}
            return render_template("predict.html", predicted_dict=data)
        if not allowed_file(file.filename):
            data = {'error':' Uploaded image format not supported'}
            return render_template("predict.html", predicted_dict=data)

        #try:
        file_name = file.filename
        img_bytes = file.read()
        print('img_bytes:', type(img_bytes))
        tensor, pil_img = transform_image(img_bytes)
        print('tensor.size():', tensor.size())
        prediction = get_prediction(tensor)
        print('prediction:', prediction)
        class_idx = int((prediction[0][0]))
        data = {'prediction': class_idx, 'class_name': cifar10_dict[class_idx]}

        image_path = os.path.join('static', 'saved_images', file_name)
        print('image_path:', image_path)
        im = Image.open(io.BytesIO(img_bytes))
        im.save(image_path)

        disp_img_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        print('disp_image_path:', disp_image_path)
        return render_template("predict.html", predicted_dict=data, image_path=disp_img_path)
        #except:
        #    data = {'error':'error during prediction'}
        #    return render_template("predict.html", predicted_dict=data)
    else:
        return render_template("predict.html")


