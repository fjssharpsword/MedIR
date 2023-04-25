import os
from flask import Flask, render_template, request, jsonify
from static.models import FR_UNet
import torch
from torch import multiprocessing as mp
import numpy as np

from torchvision.transforms import Grayscale, ToTensor
from io import BytesIO
import base64
from PIL import Image



app = Flask(__name__)

# load the pre-trained model
model = FR_UNet()
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load("./static/chase.pth")
model.load_state_dict(checkpoint['state_dict'])

def preprocess_image(image):
    image = Grayscale(1)(image)
    image = ToTensor()(image)
    _, h, w = image.shape  # attention：shape%16=
    if h > w:
        max_length = h
    else:
        max_length = w
    if max_length%16 != 0:
        a = max_length%16
        max_length = max_length - a
    pad = torch.nn.ConstantPad2d((0, max_length - w, 0, max_length - h), 0)
    image = pad(image)
    image = image.unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    return image

@app.route('/upload', methods=['POST'])
def predict():
    # Receive the uploaded image
    image_file = request.files['image']

    # Read the image into a PIL Image object
    image = Image.open(BytesIO(image_file.read()))

    # Save the uploaded image to the server
    num = len(os.listdir('./dataset/out')) + 1
    image.save(f"dataset/in/test_%05d.png" % num)

    # Preprocess the image
    image = preprocess_image(image)

    # Generate a new image using the model
    pre = model(image)
    pre = pre[0, 0, ...]
    predict = torch.sigmoid(pre).cpu().detach().numpy()
    predict_b = np.where(predict >= 0.5, 1, 0)
    generated_image = Image.fromarray(np.uint8(predict_b*255))
    path = f"dataset/out/predict_%05d.png" % num
    generated_image.save(path)

    # Convert the generated image to bytes
    buffered = BytesIO()
    generated_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Return the generated image to the user
    return jsonify({'image_str': img_str})

# Define the route for the HTML template
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    # app.run(host='0.0.0.0', port=6006, debug=False)
    # app.run(socket='/root/autodl-tmp/flaskProject-me/uwsgi.sock')
    app.run(host='0.0.0.0', port=6006, debug=False)
