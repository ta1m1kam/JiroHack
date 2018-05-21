from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Activation,Input,Flatten,Dropout
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from keras.optimizers import SGD
from keras.preprocessing import image
import os
import sys
import io
import numpy as np
from PIL import Image
import flask
import wget

app = flask.Flask(__name__)
model = None

def get_model():
    model_path = "./model/vgg16.h5py"
    global model
    if not os.path.exists(model_path):
        input_tensor = Input(shape=(224,224,3))
        model = VGG16(weights='imagenet',include_top=True,input_tensor=input_tensor)
        # wget.download('https://url', model_path)
        # model = load_model(model_path)
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

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            image = prepare_img(image)

            preds = model.predict(image)
            results = decode_predictions(preds)
            data["predictions"] = []

            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            data["success"] = True

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
    app.run()
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
