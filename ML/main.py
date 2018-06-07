from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Activation,Input,Flatten,Dropout
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from keras.optimizers import SGD
from keras.preprocessing import image
import os
import base64
import sys
import io
import numpy as np
from PIL import Image
import flask
from flask import Flask, request, abort, make_response, current_app, jsonify
from flask_cors import CORS, cross_origin
import subprocess

app = flask.Flask(__name__)
CORS(app)
model = None

def get_model():
    model_path = "./model/vgg16.h5py"
    global model
    if not os.path.exists(model_path):
        # input_tensor = Input(shape=(224,224,3))
        # model = VGG16(weights='imagenet',include_top=True,input_tensor=input_tensor)
        code = subprocess.check_output('curl -sc /tmp/cookie \'https://drive.google.com/uc?export=download&id=11DRGCtOfm3CafFQnhk5tvHLV06NG0TaA\' > /dev/null && awk \'/_warning_/ {print $NF}\' /tmp/cookie', shell=True)
        code = code.decode().rstrip()
        subprocess.call('curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm={}&id=11DRGCtOfm3CafFQnhk5tvHLV06NG0TaA" -o {}'.format(code, model_path), shell=True)
        model = load_model(model_path)
        model.save(model_path)
    else:
        model = load_model(model_path)
    model._make_predict_function()
    # return model

def prepare_img(img, target_size=(224,224)):
    # img = image.load_img(img,target_size=(224,224))
    img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x

@app.route('/predict', methods=["POST"])
def predict():
    data = {"success": False}
    enc_data  = flask.request.form['img']
    dec_data = base64.b64decode( enc_data.split(',')[1] )

    # image = flask.request.files["image"].read()
    image = Image.open(io.BytesIO(dec_data)).convert("RGB")
    # import pdb; pdb.set_trace()
    image = prepare_img(image)

    preds = model.predict(image)
    results = decode_predictions(preds)
    data["predictions"] = []

    for (imagenetID, label, prob) in results[0]:
        r = {"label": label, "probability": float(prob)}
        data["predictions"].append(r)
        data["success"] = True
    print(data)
    return flask.jsonify(data)

# def create_last_conv2d_fine_tuning(classes):
#     # vgg16モデルを作る
#     vgg16_model = get_model()
#
#     input_tensor = Input(shape=(224,224,3))
#
#     x = vgg16_model.output
#     x = Flatten()(x)
#     x = Dense(2048, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(1024, activation='relu')(x)
#     predictions = Dense(classes, activation='softmax')(x)
#     model = Model(inputs=vgg16_model.input, outputs=predictions)
#     # 最後の畳み込み層より前の層の再学習を防止
#     for layer in model.layers[:15]:
#         layer.trainable = False
#
#     model.compile(loss='categorical_crossentropy',
#                  optimizer=SGD(lr=1e-4, momentum=0.9),
#                  metrics=['accuracy'])
#    return model

if __name__ == "__main__":
    print("Loading Keras model and Flask starting server...\n")
    get_model()
    app.run(host='0.0.0.0', port=80)
    # model = get_model()
    # model.summary()

    # filename = sys.argv[1]
    # img = image.load_img(filename,target_size=(224,224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # preds = model.predict(preprocess_input(x))
    # results = decode_predictions(preds, top=5)[0]
    # for result in results:
    #     print(result)
