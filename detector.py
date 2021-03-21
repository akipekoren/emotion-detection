import cv2
from tensorflow.keras.models import model_from_json


"""
Basic class for detector operator.
It uses model from ipynb file and predefined face classifier haarcascade_frontalface_default.
"""
class Detector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.model = model_from_json(open("fer.json", "r").read())
        self.model.load_weights("fer.h5")
