import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

label = {"name": 1}
with open ("label.pickle", 'rb') as f:
        original_labels = pickle.load(f)
        labels = {v:k for k,v in original_labels.items()}

cap = cv2.VideoCapture(0)
while (True):
    #capture video frame by frame
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for(x,y,w,h) in faces:
        #regions of intrest
        r_gray = gray [y:y+h, x:x+w] #end of y-cord and  x-cord
        r_clr = frame[y:y+h, x:x+w]
        #recognition
        id_ , confidence = recognizer.predict(r_gray)
        if confidence>= 4 and confidence <= 85:
               font = cv2.FONT_HERSHEY_COMPLEX_SMALL
               name = labels[id_]
               color = (255, 0, 255)
               thick = 1
               cv2.putText(frame, name, (x,y),font, 1, color, thick, cv2.LINE_AA)

        img_face = "akshar.png"
        cv2.imwrite(img_face, r_clr)

        color = (0, 0, 255 ) # BGR format
        stroke = 2
        width = x + w
        height = y + w
        cv2.rectangle(frame, (x,y), (width, height), color, stroke)
        eyes = eye_cascade.detectMultiScale(r_gray)
       # for (l,m,n,o) in eyes:
           # cv2.rectangle(r_clr, (l,m), (l+n, m+o), (0, 255, 0), 2)


        #display result frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20)&0xFF == ord('q'):
        break
#release capture
cap.release()
cv2.destroyAllWindows()

