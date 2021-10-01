from flask import Flask, request, jsonify
from app.torch_utils import transform_image, get_prediction
#from torch_utils import transform_image, get_prediction (This is for local)
from flask import render_template, flash, redirect, url_for
import io
import base64
from PIL import Image

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png','jpg', 'jpeg'}
cifar10_dict = {0:'plane', 1:'automob', 2:'bird', 3:'cat',4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        if file is None or file.filename == "":
            out_data = {'error':'no file'}
            return render_template("predict.html", predicted_dict=out_data)
        if not allowed_file(file.filename):
            out_data = {'error':' Uploaded image format not supported'}
            return render_template("predict.html", predicted_dict=out_data)

        try:
            img_bytes = file.read()
            tensor, pil_img = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            class_idx = int((prediction[0][0]))
            out_data = {'Predicted class': class_idx, 'Class Name': cifar10_dict[class_idx]}

            im = Image.open(io.BytesIO(img_bytes))
            data = io.BytesIO()
            im.save(data, 'JPEG')
            encoded_img_data = base64.b64encode(data.getvalue())

            return render_template("predict.html", predicted_dict=out_data, img_data=encoded_img_data.decode('utf-8'))
        except:
            out_data = {'error':'error during prediction'}
            return render_template("predict.html", predicted_dict=out_data)
    else:
        return render_template("predict.html")


