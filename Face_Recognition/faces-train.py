import os
import numpy as np
import cv2
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # we are getting the location of this file i.e faces-train.py
image_dir = os.path.join(BASE_DIR, "images") # in directory we finding "images"

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {} 
y_label = [] # contains values related to the labels
x_train = [] # contains pixel values

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file) # here we are getting paths of all files saved with extension .png or .jpg as specified above
            label = os.path.basename(root).replace(" ", "-").lower() # we get the label of the file whose path in found above
            #print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1          
            id_ = label_ids[label]
            #print(label_ids)
            #y_label.append(label) # some number
            #x_train.append(path) # verify this image, turn into a NUMBY array, GRAY
            pil_image = Image.open(path).convert("L") # turned the pixel value of image into grayscale
            #size = (600,600) # resizing the image
            #final_image = pil_image.resize(size, Image.Resampling.LANCZOS) # avoid destortion of image
            image_array = np.array(pil_image, "uint8") # turned the grayscale image into a numpy array i.e. into numbers 
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5,minNeighbors=5) # detects the face in the image

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+h]
                x_train.append(roi)
                y_label.append(id_)

# saving label ids
with open("labels.pickle", "wb") as f:  # wb = writing byte and f = files
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_label))
recognizer.save("trainner.yml")
