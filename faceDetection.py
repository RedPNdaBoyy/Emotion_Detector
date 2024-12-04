
import cv2
import dlib
from fer import FER
import numpy as np
import time
from collections import defaultdict


sp = dlib.shape_predictor('URPATH/shape_predictor_68_face_landmarks.dat')  
face_encoder = dlib.face_recognition_model_v1('URPATH/dlib_face_recognition_resnet_model_v1.dat') 
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
emotion_detector = FER(mtcnn=True)






def get_emotion_from_img(img,model):
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(gray_img)
    face_descriptors=[]
    emotions=[]
    for face in faces:
        x1,y1,x2,y2=(face.left(),face.top(),face.right(),face.bottom())
        
        croped_face=img[max(0,y1):y2,max(0,x1):x2]
        
        if croped_face.size > 48:
            croped_face = cv2.resize(croped_face, (48, 48), interpolation=cv2.INTER_LINEAR)
            croped_face = np.expand_dims(croped_face, axis=0)
            score = model.predict(croped_face)
            emotion = np.argmax(score)
            score = np.max(score)
            croped_face=croped_face[0]
        else: 
            emotion=None
            score=None
            croped_face=img
        if emotion is None:
            emotion="UNKNOW"
        if score is None:
            score=0

        emotions.append(emotion)

        cv2.rectangle(gray_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

       
        shape = sp(gray_img, face)
        face_descriptor = face_encoder.compute_face_descriptor(img, shape)
        face_descriptors.append(face_descriptor)
    if len(emotions)>0:
        return emotions
    else: 
        return [7] 

        