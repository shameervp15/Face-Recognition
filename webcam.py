import cv2 as cv
import os

cascPath = os.path.dirname(cv.__file__)+"\\data\\haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPath)

videocapture = cv.VideoCapture(0)

while True:
    ret, frames = videocapture.read()

    gray = cv.cvtColor(frames, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    #rectangular mark on faces
    for (x, y, w, h) in faces:
        cv.rectangle(frames, (x, y), (x+w, y+h), (0,255,0), 2)

    cv.imshow('Video', frames)
    key = cv.waitKey(1)

    if key == ord('q'):
        break

videocapture.release()
cv.destroyAllWindows()








