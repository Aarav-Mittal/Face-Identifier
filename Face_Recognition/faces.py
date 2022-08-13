from re import M
from turtle import width
import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier("cascades\data\haarcascade_eye.xml")
#smile_cascade = cv2.CascadeClassifier("cascades\data\haarcascade_smile.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

label = {"person_name": 1}
with open("labels.pickle", "rb") as f:  # f = files
    og_label = pickle.load(f)
    label = {v:k for k,v in og_label.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #roi = region of interest
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <=85:
            print(id_)
            print(label[id_])
            # puting label on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = label[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0) #BGR-- blue-green-red (0-255)  #colour of frame
        stroke = 2 #thickness of the frame
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
    '''        
        # detects eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        # detects smile
        smile = smile_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in smile:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    '''
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

