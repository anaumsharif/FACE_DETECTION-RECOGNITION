import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3,640) # width
cam.set(4, 480) # set video height

# variable to store faces
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n Enter user id :')

#default messages 
print("\n [INFO] Initializing face capture....")
# Initialize individual sampling face count
count = 0

while(True):
    # to read the content in a varaible
    ret, img = cam.read()
    # converting color images to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # result image is stored in faces
    # standard default image size
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # images are either square or rectangle 
    # so 4 variables denote 4 corners of an  image 
    for (x,y,w,h) in faces:

        # default for every face detection algorithm 
        # 4 edges and incorporating them ---weights range from 0 to 255
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        # for every user a particular id is assigned 
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        # to show image  
        cv2.imshow('image', img)

    # how much time user needs to wait to capture image
    k = cv2.waitKey(100) & 0xff 
    # it waits for 27 secs 
    if k == 27:
        break
    elif count >= 30: 
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()