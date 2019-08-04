import cv2
import os
import numpy as np
from PIL import Image
import pickle

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')

recognizer  = cv2.face.LBPHFaceRecognizer_create() #pip install opencv-contrib-python

base_directory  = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base_directory, "images")

x_train = []
y_label = []
cur_id = 0
label_lib = {}


for root,dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()
           # print(label, path)
            if not label in label_lib:
                label_lib[label] = cur_id
                cur_id += 1
            id__ = label_lib[label]
            #print(label_lib)

            pil_img = Image.open(path).convert("L")
            size = (600, 600)
            final_image = pil_img.resize(size, Image.ANTIALIAS)
            img_array = np.array(pil_img, "uint8")
            #print(img_array)
            faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5)


            for(x,y,w,h) in faces:
                roi = img_array[y: y+h, x:x+w]
                x_train.append(roi)
                y_label.append(id__)

with open("label.pickle", 'wb') as f:
    pickle.dump(label_lib, f)

#recognizer
recognizer.train(x_train, np.array(y_label))
recognizer.save("trainer.yml")











