import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):
    
    # defining image path with the help of os for everything in the directory of os 
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    # creating loop for image path 
    for imagePath in imagePaths:

        # convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L') 
        # UNIT8 - takes element as unsigned integer and stores it using np array 
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        # multiscale method to convert images into grayscale images 
        faces = detector.detectMultiScale(img_numpy)

        # image size
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') 

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))
