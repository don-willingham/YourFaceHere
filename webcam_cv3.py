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

# Referenced http://my.execpc.com/~steidl/robotics/first_order_lag_filter.html
def lag(new, k, old):
    return(k * new + ((1-k)*old))

def composite(frame_pil, maker, place_x, place_y):
    place_x = int(round(place_x))
    place_y = int(round(place_y))

    if (place_x < 0):
        maker = maker.crop((-1 * place_x, 0, maker.width - 1, maker.height - 1))
        place_x = 0
    if (place_y < 0):
        maker = maker.crop((0, -1 * place_y, maker.width - 1, maker.height - 1))
        place_y = 0

    frame_pil.alpha_composite(maker, (place_x, place_y), (0,0))

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
maker_man_raw = Image.open('Tampa MMF Man.png')
center_face_x_raw = 416
center_face_y_raw = 138
face_width_raw = 171
face_height_raw = 133
video_capture = cv2.VideoCapture(0)
place_x = -1
place_y = -1
scale_factor = 1.0
maker_man_scaled = maker_man_raw

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
        center_you_x = int(round(big_x + big_w/2))
        center_you_y = int(round(big_y + big_h/2))
        scale_factor = lag(((big_h / face_height_raw) + (big_w / face_width_raw)) / 2.0, 0.25, scale_factor)
        center_face_x = center_face_x_raw
        center_face_y = center_face_y_raw
        maker_man_scaled = maker_man_raw
        if scale_factor > 1.01 or scale_factor < 0.99:
            maker_man_scaled = maker_man_raw.resize((int(round(maker_man_raw.width*scale_factor)),
                                                     int(round(maker_man_raw.height*scale_factor))),
                                                    Image.LANCZOS)
            center_face_x = int(round(center_face_x * scale_factor))
            center_face_y = int(round(center_face_y * scale_factor))
        else:
            maker_man_scaled = maker_man_raw
        place_x = center_you_x - center_face_x
        place_y = center_you_y - center_face_y

    # Convert ot PIL format
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 'RGB')
    # Add alpha channel
    frame_pil.putalpha(1)
    # Overlay maker man
    composite(frame_pil, maker_man_scaled, place_x, place_y)

    # Display the resulting frame
    cv2.imshow('Video', cv2.cvtColor(np.asarray(frame_pil), cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
