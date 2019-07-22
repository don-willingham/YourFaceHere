import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)
maker_man = cv2.imread('Tampa MMF Man.png')

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    # Find the largest face
    bigX = -1
    bigY = -1
    bigW = -1
    bigH = -1
    for (x, y, w, h) in faces:
        if bigX < 0:
            bigX = x
            bigY = y
            bigW = w
            bigH = h
        elif (w * h) > (bigW * bigH):
            bigX = x
            bigY = y
            bigW = w
            bigH = h

    if bigX > -1:
        cv2.rectangle(frame, (bigX, bigY), (bigX+bigW, bigY+bigH), (0, 255, 0), 2)
        #frame = cv2.addWeighted(frame,0.4,maker_man,0.1,0)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
