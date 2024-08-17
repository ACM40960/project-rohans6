import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from playsound import playsound
from threading import Thread

def start_alarm(sound):
    """Play the alarm sound"""
    playsound(sound)

from transformers import TFViTModel

# Load the model with custom objects
custom_objects = {'TFViTMainLayer': TFViTModel}
model = tf.keras.models.load_model("savedmodels/drowsiness.h5", custom_objects=custom_objects)

# Define categories based on model output
classes = ['Closed', 'Open', 'no_yawn', 'yawn']

# Load Haar cascades
face_cascade = cv2.CascadeClassifier("Cascade/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("Cascade/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("Cascade/haarcascade_righteye_2splits.xml")
mouth_cascade = cv2.CascadeClassifier("Cascade/haarcascade_mcs_mouth.xml")  # Assuming you have a mouth cascade

cap = cv2.VideoCapture(0)
count = 0
alarm_on = False
alarm_sound = "Cascade/alarm.mp3"
status1 = ''
status2 = ''
mouth_status = ''

while True:
    _, frame = cap.read()
    height = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)  # Adjust the scale factor and minNeighbors if needed

        # Eye detection and classification
        for (x1, y1, w1, h1) in left_eye:
            cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
            eye1 = roi_color[y1:y1+h1, x1:x1+w1]
            eye1 = cv2.resize(eye1, (224, 224))  # Resize to match ViT input size
            eye1 = eye1.astype('float32') / 255.0
            eye1 = img_to_array(eye1)
            eye1 = np.expand_dims(eye1, axis=0)
            pred1 = model.predict(eye1)
            status1 = np.argmax(pred1)
            break

        for (x2, y2, w2, h2) in right_eye:
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
            eye2 = cv2.resize(eye2, (224, 224))  # Resize to match ViT input size
            eye2 = eye2.astype('float32') / 255.0
            eye2 = img_to_array(eye2)
            eye2 = np.expand_dims(eye2, axis=0)
            pred2 = model.predict(eye2)
            status2 = np.argmax(pred2)
            break

        # Mouth detection and classification
        for (mx, my, mw, mh) in mouth:
            y = int(y + h / 2)
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 255), 1)
            mouth_region = roi_color[my:my+mh, mx:mx+mw]
            mouth_region = cv2.resize(mouth_region, (224, 224))  # Resize to match ViT input size
            mouth_region = mouth_region.astype('float32') / 255.0
            mouth_region = img_to_array(mouth_region)
            mouth_region = np.expand_dims(mouth_region, axis=0)
            mouth_pred = model.predict(mouth_region)
            mouth_status = np.argmax(mouth_pred)
            break

        # Check if either eye is closed or mouth is yawning
        if status1 == 0 or status2 == 0 or mouth_status == 3:  # Assuming 'Closed' is class 0 and 'yawn' is class 3
            count += 1
            cv2.putText(frame, "Drowsiness Detected, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            # if drowsiness is detected for a certain number of consecutive frames, start the alarm
            if count >= 5:
                cv2.putText(frame, "Drowsiness Alert!!!", (100, height-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                if not alarm_on:
                    alarm_on = True
                    # play the alarm sound in a new thread
                    t = Thread(target=start_alarm, args=(alarm_sound,))
                    t.daemon = True
                    t.start()
        else:
            cv2.putText(frame, "Normal", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            count = 0
            alarm_on = False

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
