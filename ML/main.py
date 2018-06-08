from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Activation,Input,Flatten,Dropout
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from keras.utils import np_utils
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
from sklearn.model_selection import train_test_split

app = flask.Flask(__name__)
CORS(app)
model = None

def get_model():
    model_path = "./model/vgg16.h5py"
    global model
    if not os.path.exists(model_path):
        input_tensor = Input(shape=(224,224,3))
        model = VGG16(weights='imagenet',include_top=False,input_tensor=input_tensor)
        # code = subprocess.check_output('curl -sc /tmp/cookie \'https://drive.google.com/uc?export=download&id=11DRGCtOfm3CafFQnhk5tvHLV06NG0TaA\' > /dev/null && awk \'/_warning_/ {print $NF}\' /tmp/cookie', shell=True)
        # code = code.decode().rstrip()
        # subprocess.call('curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm={}&id=11DRGCtOfm3CafFQnhk5tvHLV06NG0TaA" -o {}'.format(code, model_path), shell=True)
        # model = load_model(model_path)
        model.save(model_path)
    else:
        model = load_model(model_path)
    model._make_predict_function()

def prepare_img(img, target_size=(224,224)):
    # img = image.load_img(img,target_size=(224,224))
    img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x

def prepare_train():
    X = []
    Y = []

    # 対象Aの画像
    for picture in list_pictures('./ramen/'):
        img = img_to_array(load_img(picture, target_size=(224,224)))
        X.append(img)

        Y.append(0)


    # 対象Bの画像
    for picture in list_pictures('./ラーメン二郎/'):
        img = img_to_array(load_img(picture, target_size=(224,224)))
        X.append(img)

        Y.append(1)


    # arrayに変換
    X = np.asarray(X)
    Y = np.asarray(Y)
    # 画素値を0から1の範囲に変換
    X = X.astype('float32')
    X = X / 255.0

    # クラスの形式を変換
    Y = np_utils.to_categorical(Y, 2)

    # 学習用データとテストデータ
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=111)
    return X_train,X_test,y_train,y_test

@app.route('/predict', methods=["POST"])
def predict():
    data = {"success": False}
    enc_data  = flask.request.form['img']
    dec_data = base64.b64decode( enc_data.split(',')[1] )

    # image = flask.request.files["image"].read()
    image = Image.open(io.BytesIO(dec_data)).convert("RGB")
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

def create_last_conv2d_fine_tuning(classes):
    # vgg16モデルを作る
    get_model()
    global model
    vgg16_model = model

    x = vgg16_model.output
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=vgg16_model.input, outputs=predictions)
    # 最後の畳み込み層より前の層の再学習を防止
    for layer in model.layers[:15]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                 optimizer=SGD(lr=1e-4, momentum=0.9),
                 metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Loading Keras model and Flask starting server...\n")
    # get_model()
    # app.run(host='0.0.0.0', port=80)

    #-- TRAINING MODEL --#
    model = create_last_conv2d_fine_tuning(2)
    # 実行。出力はなしで設定(verbose=0)。
    X_train, X_test, y_train, y_test = prepare_train()
    history = model.fit(X_train, y_train, batch_size=5, epochs=100,
                    validation_data = (X_test, y_test), verbose = 1)



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
