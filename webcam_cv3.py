import cv2
import sys
import datetime as dt
import numpy as np
from time import sleep
from PIL import Image

def get_largest_face(faces):
    # Find the largest face
    big_x = -1
    big_y = -1
    big_w = -1
    big_h = -1
    for (x, y, w, h) in faces:
        if big_x < 0:
            big_x = x
            big_y = y
            big_w = w
            big_h = h
        elif (w * h) > (big_w * big_h):
            big_x = x
            big_y = y
            big_w = w
            big_h = h
    return(big_x, big_y, big_w, big_h)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
maker_man_raw = Image.open('Tampa MMF Man.png')
center_face_x = 416 / 739
center_face_y = 138 / 504
scale_factor = 0.5
if scale_factor > 1.01 or scale_factor < 0.99:
    maker_man_raw = maker_man_raw.resize((round(maker_man_raw.width*scale_factor), round(maker_man_raw.height*scale_factor)), Image.LANCZOS)
    center_face_x = int(round(center_face_x * scale_factor))
    center_face_y = int(round(center_face_y * scale_factor))

video_capture = cv2.VideoCapture(0)
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

    # Find the largest face
    (big_x, big_y, big_w, big_h) = get_largest_face(faces)

    # Draw a rectangle around the largest face
    if big_x > -1:
        cv2.rectangle(frame, (big_x, big_y), (big_x+big_w, big_y+big_h), (0, 255, 0), 2)
        place_x = big_x - int(round(big_w/2)) - center_face_x
        place_y = big_y - int(round(big_h/2)) - center_face_y

    # Convert ot PIL format
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 'RGB')
    # Add alpha channel
    frame_pil.putalpha(1)

    if place_x > -1 and place_y > -1:
        frame_pil.alpha_composite(maker_man_raw, (place_x, place_y), (0,0))

    # Display the resulting frame
    cv2.imshow('Video', cv2.cvtColor(np.asarray(frame_pil), cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
