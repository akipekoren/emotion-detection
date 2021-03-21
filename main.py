import pandas as pd 
import cv2
import numpy as np
from keras.preprocessing import image 
from detector import Detector
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

print("PROGRAM STARTS")

emotion_detector = Detector()
emotion_list = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')  
program_name = "Emotion Detection"

while True:  
    ret,test_img=emotion_detector.cap.read()    #capture image 
    gray_image= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  
    faces_detected = emotion_detector.face_cascade_classifier.detectMultiScale(gray_image, 1.32, 5)  
  
    for (x,y,w,h) in faces_detected:  
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
        roi_gray=gray_image[y:y+w,x:x+h] 
        roi_gray=cv2.resize(roi_gray,(48,48))  
        img_pixels = image.img_to_array(roi_gray)  
        img_pixels = np.expand_dims(img_pixels, axis = 0) 
        img_pixels /= 255  
        predictions = emotion_detector.model.predict(img_pixels)  
        chosen_emotion = np.argmax(predictions[0])                      #choose the max prob emotion
        predicted_emotion = emotion_list[chosen_emotion]  
  
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), emotion_detector.font, 1, (0,0,255), 2)  
  
    img_new_size = cv2.resize(test_img, (900, 600))  
    cv2.imshow(program_name,img_new_size)  
  
    if cv2.waitKey(10) == ord('x'):
        break  
  
emotion_detector.cap.release()  
cv2.destroyAllWindows 