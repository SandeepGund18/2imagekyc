#  Final face recogniton

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
from numpy import asarray
from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
import cv2
import os
import pytesseract
import re

class face_rec():
    #def __init__(self):
     #   self.detector = dlib.get_frontal_face_detector()

    def face_detect(self,path, required_size=(224, 224)):
         # load image and detect faces
        image = plt.imread(path)
        detector = MTCNN()
        faces = detector.detect_faces(image)

        if len(faces) > 0:
            status = True

            # extract the first detected face
            face = faces[0]

            # extract the bounding box from the requested face
            x1, y1, width, height = face['box']
            x2, y2 = x1 + width, y1 + height

            # extract the face
            face_boundary = image[y1:y2, x1:x2]

            # resize pixels to the model size
            face_image = Image.fromarray(face_boundary)
            face_image = face_image.resize(required_size)
            face_array = asarray(face_image)

            return face_array, status
        else:
            status = False
            return image , status

    def id_verification(self,path):
        img =cv2.imread(path)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\WYH-IT-806-029\Tesseract-OCR\tesseract.exe'
        text =   pytesseract.image_to_string(img)
        id_proof=re.findall("\d{4}\s\d{4}\s\d{4}|[A-Z]{5}\d{4}[A-Z]",text)
        
        return id_proof
    
    def my_img(self):
        cam_port = 0
        cam = cv2.VideoCapture(cam_port)
        result, image = cam.read()
        if result:
            cv2.imwrite(r'static\images\my_pic.jpg',image)
            return r'static\images\my_pic.jpg'

    def my_img1(self):
        cam_port = 0
        cam = cv2.VideoCapture(cam_port)
        while True:
            result, image = cam.read()
            if result:
                cv2.imshow('Check Your Face',image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite(r'static\images\my_pic.jpg',image)
                break
        cam.release()
        cv2.destroyAllwindows()
        return r'static\images\my_pic.jpg'

    def get_model_scores(self,face1, face2):
        samples = asarray([face1, face2], 'float32')

        # prepare the data for the model
        samples = preprocess_input(samples, version=2)

        # create a vggface model object
        model = VGGFace(model='resnet50',
            include_top=False,
            input_shape=(224, 224, 3), 
            pooling='avg')

        # perform prediction
        return model.predict(samples)            


fr=face_rec()