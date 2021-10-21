from flask import Flask, flash, request, redirect, url_for, render_template
import os
import io
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from werkzeug.utils import secure_filename

out_class = ['Negative', 'Positive']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.circuit_size = 2
        self.conv1 = nn.Conv2d(1, 6 , kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16,self.circuit_size**2)
        self.fc5 = nn.Linear(self.circuit_size**2, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3)
        #x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 3)
        x = F.relu(self.conv4(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = self.hybrid(x)
        x = self.fc5(x)
        x = torch.sigmoid(x)
        return x

def transform_image(img):
    transformations = transforms.Compose([transforms.Resize(200),
                                          transforms.Grayscale(),
                                          transforms.ToTensor()])
    image = Image.open(img)
    return transformations(image).unsqueeze(0)

def get_prediction(img):
    tensor = transform_image(img)
    output = model.forward(tensor)
    return out_class[int(output.item()>=0.5)]


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tech')
def tech():
    return render_template('tech.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        new_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(new_file)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        classname = get_prediction(new_file)
        return render_template('index.html', filename=filename,
        classname = classname)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    global model
    model= Net()
    checkpoint = torch.load('qcnn.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    app.run()
