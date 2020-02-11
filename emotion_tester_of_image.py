# -*- coding: utf-8 -*-
import pickle
import cv2
import numpy as np
from keras.preprocessing import image
import face_recognition


with open('C:\\Users\\Hp\\PycharmProjects\\emotion_analysis\\emotion-analysis-model','rb') as f: # provide file name with path
  model = pickle.load(f)

image_path = "C:\\Users\\Hp\\PycharmProjects\\group1.jpg"  # image name with path
face_haar_cascade = cv2.CascadeClassifier('C:\\Users\\Hp\\haarcascade_frontalface_default.xml')  # Cascade file with path

test_img = face_recognition.load_image_file(image_path)

gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)


faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


for (x,y,w,h) in faces_detected:

    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
    roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
    roi_gray=cv2.resize(roi_gray,(48,48))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255

    predictions = model.predict(img_pixels)

    #find max indexed array
    max_index = np.argmax(predictions[0])

    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    predicted_emotion = emotions[max_index]

    cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    test_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

resized_img = cv2.resize(test_img, (600, 520))
cv2.imshow("Image Test", resized_img)

cv2.destroyAllWindows