import cv2
import sys
import logging as log
import datetime as dt
import numpy as np
from time import sleep
from PIL import Image

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)
maker_man_raw = Image.open('Tampa MMF Man.png')
center_face_x = 416 / 739
center_face_y = 138 / 504
scale_factor = 0.5
if scale_factor > 1.01 or scale_factor < 0.99:
    maker_man_raw = maker_man_raw.resize((round(maker_man_raw.width*scale_factor), round(maker_man_raw.height*scale_factor)), Image.LANCZOS)
    center_face_x = int(round(center_face_x * scale_factor))
    center_face_y = int(round(center_face_y * scale_factor))

video_capture = cv2.VideoCapture(0)
anterior = 0
place_x = -1
place_y = -1

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert to gray scale
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
        place_x = bigX - int(round(bigW/2)) - center_face_x
        place_y = bigY - int(round(bigH/2)) - center_face_y

    # Convert ot PIL format
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 'RGB')
    # Add alpha channel
    frame_pil.putalpha(1)

    if place_x > -1 and place_y > -1:
        frame_pil.alpha_composite(maker_man_raw, (place_x, place_y), (0,0))

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', cv2.cvtColor(np.asarray(frame_pil), cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
