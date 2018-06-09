from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Activation,Input,Flatten,Dropout
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
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
f_model = './model'

def get_model():
    model_path = "./model/vgg16.h5py"
    #global model
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
    #model._make_predict_function()
    return model

def prepare_img(img, target_size=(224,224)):
    # img = image.load_img(img,target_size=(224,224))
    img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    x = x.astype('float32')
    x = x / 255.0

    return x

def prepare_train():
    X = []
    Y = []

    # 対象Aの画像
    for picture in list_pictures('./ordinary/'):
        img = img_to_array(load_img(picture, target_size=(224,224)))
        X.append(img)

        Y.append(0)


    # 対象Bの画像
    for picture in list_pictures('./jiro/'):
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
    # global model
    ## API 部分
    from keras.models import model_from_json
    model = model_from_json(open("model/model.json").read())
    model.load_weights('model.ep10.h5py')
    print(type(model))
    ## Rails
    enc_data  = flask.request.form['img']
    dec_data = base64.b64decode( enc_data.split(',')[1] )
    # image = flask.request.files["image"].read()
    image = Image.open(io.BytesIO(dec_data)).convert("RGB")
    image = prepare_img(image)
    preds = model.predict(image)

    ## Local
    ## if flask.request.method == "POST":
    ##     if flask.request.files.get("image"):
    ##         image = flask.request.files["image"].read()
    ##         image = Image.open(io.BytesIO(image))
    ##         image = prepare_img(image)
    ##         import pdb;pdb.set_trace()

    ##         preds = model.predict(image)
    ##         # results = decode_predictions(preds)
    ##         data["predictions"] = []


    ##         for label,i in enumerate(preds[0]):
    ##            r = {"label": label, "probability": float(i)}
    ##            data["predictions"].append(r)
    ##            data["success"] = True

    # for (imagenetID, label, prob) in results[0]:
    #     r = {"label": label, "probability": float(prob)}
    #     data["predictions"].append(r)
    #     data["success"] = True
    data["predictions"] = []
    for label,i in enumerate(preds[0]):
        score = str(int(float(i) * 500))[:-1] + "0"
        r = {"label": label, "probability": score}
        data["predictions"].append(r)
        data["success"] = True
    print(data)
    return flask.jsonify(data)

def create_last_conv2d_fine_tuning(classes):
    # vgg16モデルを作る
    vgg16_model = get_model()
    # global model
    # vgg16_model = model
    input_tensor = Input(shape=(224,224,3))

    for layer in vgg16_model.layers:
        layer.trainable = False

    x = vgg16_model.output
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=vgg16_model.input, outputs=predictions)
    # 最後の畳み込み層より前の層の再学習を防止
    # for layer in model.layers[:15]:
    #     layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                 optimizer=SGD(lr=1e-4, momentum=0.9),
                 metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Loading Keras model and Flask starting server...\n")
    #get_model()
    app.run(host='0.0.0.0', port=80)
    # app.run()

#-- TRAINING --#
    # model = create_last_conv2d_fine_tuning(2)
    # ## # 実行。出力はなしで設定(verbose=0)。
    # X_train, X_test, y_train, y_test = prepare_train()
    # callbacks = []
    # callbacks.append(EarlyStopping("val_loss", patience=1))
    # callbacks.append(ModelCheckpoint(filepath="model.ep{epoch:02d}.h5py"))
    # history = model.fit(X_train, y_train, batch_size=32, epochs=10,\
    #                 validation_data=(X_test, y_test), verbose=1, callbacks=callbacks)
    # score = model.evaluate(X_test, y_test, verbose=0)
    # json_string = model.to_json()
    # open(os.path.join(f_model,'model.json'), 'w').write(json_string)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    # # Prediction
    # import numpy as np
    # from sklearn.metrics import confusion_matrix

    # predict_classes = model.predict_classes(X_test[1:10,], batch_size=32)
    # true_classes = np.argmax(y_test[1:10],1)
    # print(confusion_matrix(true_classes, predict_classes))



# Predict
    # model = get_model()
    # model.summary()

    ## model.load_weights('model.ep04.h5py')
    ## filename = sys.argv[1]
    ## img = image.load_img(filename,target_size=(224,224))
    ## x = image.img_to_array(img)
    ## x = np.expand_dims(x, axis=0)
    ## result = model.predict(x)
    ## print(result)
    # preds = model.predict(preprocess_input(x))
    ## results = decode_preditions(preds, top=5)[0]
    # for result in results:
    #     print(result)
    ## from keras.models import model_from_json
    ## model = model_from_json(open("model/model.json").read())
    ## model.load_weights('model.ep10.h5py')
    ## filename = sys.argv[1]
    ## img = image.load_img(filename,target_size=(224,224))
    ## x = image.img_to_array(img)
    ## x = np.expand_dims(x, axis=0)
    ## x = x.astype('float32')
    ## x = x / 255.0

    ## result = model.predict(x)
    ## print(result)
