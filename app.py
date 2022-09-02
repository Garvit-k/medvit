import io
import os
import urllib
import uuid
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class_index = {
    "0": "AbdomenCT",
    "1": "BreastMRI",
    "2": "CXR",
    "3": "ChestCT",
    "4": "Hand",
    "5": "HeadCT"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model(model_name='vit_base_patch8_224', pretrained=True, img_size=32, in_chans=1, num_classes=6,
                          drop_rate=0.1).to(device)

model.load_state_dict(torch.load("pretrainedbestmodel.pth"))

model.eval()


def transform_image(image_bytes):
    transforms = T.Compose([
        T.Resize(size=(32, 32)),
        T.Grayscale(),
        T.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return transforms(image).unsqueeze(0).to(device)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes).to(device)
    with torch.no_grad():
        outputs = model(tensor)
    print("model output")
    print(outputs)
    print("--------------")
    _, y_hat = outputs.max(1)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    return probabilities


ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'jfif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


classes = ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']


def predict(filename):
    img_bytes = open(filename, 'rb').read()
    result = get_prediction(img_bytes)

    dict_result = {}
    result = result.tolist()
    for i in range(6):
        dict_result[classes[i]] = result[i]

    prob = {k: v for k, v in sorted(dict_result.items(), key=lambda item: item[1], reverse=True)}
    resultprobs = {}
    counter = 0
    for key in prob.keys():
        counter += 1
        resultprobs["class" + str(counter)] = key
        resultprobs["prob" + str(counter)] = prob[key] * 100.0

    return resultprobs


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img: str = os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if request.form:
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()
                img = filename

                # class_result , prob_result = predict(img_path , model)

                predictions = predict(img_path)

            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accessible or inappropriate input'

            if len(error) == 0:
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)
            
        elif request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename

                # class_result , prob_result = predict(img_path , model)

                predictions = predict(img_path)

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if len(error) == 0:
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=False)
