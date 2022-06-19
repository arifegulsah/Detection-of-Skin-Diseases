import argparse
from email.mime import image

import numpy as np
import yaml
import cv2
from tensorflow.keras.applications.densenet import preprocess_input # densenet121 olması gerekiyor.
#from tensorflow.keras.applications import * 
from tensorflow.keras.models import load_model
import tensorflow as tf
import sklearn.preprocessing as preprocessing

from utils import download_model
from preprocessing import read_img_from_path, resize_img

"""

from tensorflow.keras.utils import download_model
from tensorflow.keras.utils import normalize
from fasttest.utils import download_model


from tensorflow import keras """

from tensorflow.keras.applications.imagenet_utils import preprocess_input



class ImagePredictor:
    def __init__(
        self, model_path, resize_size, targets , pre_processing_function=preprocess_input
    ):
        self.model_path = "../model.h5"
        self.pre_processing_function = pre_processing_function
        self.model = load_model(self.model_path)
        self.resize_size = resize_size
        self.targets = targets
        
    @classmethod
    def init_from_config_path(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        predictor = cls(
            model_path=config["model_path"],
            resize_size=config["resize_shape"],
            targets=config["targets"],
        )
        return predictor

    @classmethod
    def init_from_config_url(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
            
        return cls.init_from_config_path(config_path)

    def predict_from_file(self, img):

        image = cv2.resize(img, (256, 303)) #img.resize(200,200)

        image = np.reshape(image, [1,256,303,3])
        model = load_model('model5.h5')

        class_names = [] ## TEK TEK CLASS ISIMLERINI YAZ
        classes = np.argmax(model.predict(image), axis = -1) 
        names = [class_names[i] for i in classes]

        return names


