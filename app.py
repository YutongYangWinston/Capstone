from flask import Flask, render_template
from flask import redirect
from flask import make_response, Response
from flask_cors import CORS
import pickle
import logging
import numpy as np
from flask import request
import os
from werkzeug.utils import secure_filename

import torch
from PIL import Image
from torchvision import transforms
from torchvision import datasets
import numpy as np
from net import Net
model= Net()
model = torch.load('model.pth')
transform = transforms.Compose([
    transforms.Scale([64, 64]),  # Scale 64×64
    transforms.ToTensor()
])
app = Flask(__name__)
CORS(app)
app.config["JSON_AS_ASCII"] = False

app.config['UPLOAD_FOLDER'] = 'upload/'



def get_predict(filename):
    img = Image.open(f"./upload/{filename}").convert('RGB')
    img = transform(img)
    img = img.view((-1, 3, 64, 64))
    predict = model(img)
    class_index = np.argmax(predict.detach().numpy())
    train_dataset = datasets.ImageFolder(root="./data/train", transform=transform)
    return train_dataset.classes[class_index]


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        prediction = get_predict(f.filename)
        return prediction

    else:
        return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)
