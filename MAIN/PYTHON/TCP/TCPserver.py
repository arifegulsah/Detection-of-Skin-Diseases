from email.mime import base
import os
from threading import Thread
from time import sleep
from typing import Any

from predictor import ImagePredictor

from PIL import Image

from cv2 import resize
import cv2
from preprocessing import resize_img
import socket
import sys
import base64
import numpy as np
import uuid
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageFile, Image
import re


ImageFile.LOAD_TRUNCATED_IMAGES = True

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

HOST_NAME = socket.gethostname() # It gives computer name like DESKTOP-9HL3
PORT = 11000
BUFFER_SIZE = 1024

server_address = (HOST_NAME, PORT)
print("Starting up on %s port %s" % server_address)
sock.bind(server_address)
sock.listen(1)

# While True give us always listener
while True:
    connection, client_address = sock.accept()
    data = connection.recv(BUFFER_SIZE)
    photo_path = data.decode("ascii")

    image = cv2.imread(photo_path)

    if(image is not None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor_config_path = "config.yaml"
    predictor = ImagePredictor.init_from_config_url(predictor_config_path)
    predictionResult = predictor.predict_from_file(image)

    if predictionResult:
        print("Response : " + str(predictionResult[0].encode('utf-8')))
        connection.sendall(predictionResult[0].encode('utf-8')) # .NET tarafına dönen class label.
    else:
        print("No more data from", client_address)

