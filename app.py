import base64
import numpy as np
import io
from PIL import Image
from tensorflow import keras
# from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
import CV2
from flask_cors import CORS


app=Flask(__name__)
CORS(app)

def get_model_cxr():
    global vgg_model,resnet50_model,inceptionv3_model,xception_model
    vgg_model=load_model(r'E:\PROJECT\vgg16_cxr.h5')
    resnet50_model=load_model(r'E:\PROJECT\resnet_cxr.h5')
    inceptionv3_model=load_model(r'E:\PROJECT\InceptionV3_cxr.h5')
    xception_model=load_model(r'E:\PROJECT\Xception_cxr.h5')

def get_model_ct():
    global mobilenetv2_model
    mobilenetv2_model=load_model(r'E:\PROJECT\MobilenetV2_ct.h5')


def  preprocess_image(image,target_size):
    if image.mode!="RGB":
        image=image.convert("RGB")
    image=image.resize(target_size)
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)
    image=image/255.0
    return image


@app.route("/predict_cxr",methods=["POST"])
def predict_cxr():
    get_model_cxr()
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    image=Image.open(io.BytesIO(decoded))
    processed_image=preprocess_image(image,target_size=(224,224))

    prediction_vgg=vgg_model.predict(processed_image).tolist()
    prediction_resnet=resnet50_model.predict(processed_image).tolist()
    prediction_xception=xception_model.predict(processed_image).tolist()
    prediction_inception=inceptionv3_model.predict(processed_image).tolist()

    response={
        'prediction':{
            'vgg_normal':prediction_vgg[0][1],
            'vgg_covid':prediction_vgg[0][0],
            'resnet_normal':prediction_resnet[0][1],
            'resnet_covid':prediction_resnet[0][0],
            'xception_normal':prediction_xception[0][1],
            'xception_covid':prediction_xception[0][0],
            'inception_normal':prediction_inception[0][1],
            'inception_covid':prediction_inception[0][0]
        }
    }
    return jsonify(response)


@app.route("/predict_ct",methods=["POST"])
def predict_ct():
    get_model_ct()
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    image=Image.open(io.BytesIO(decoded))
    processed_image=preprocess_image(image,target_size=(224,224))

    prediction_mobilenet=mobilenetv2_model.predict(processed_image).tolist()

    response={
        'prediction':{
            'non_covid':prediction_mobilenet[0][1],
            'covid':prediction_mobilenet[0][0],
        }
    }
    return jsonify(response)

app.run(debug=True)
