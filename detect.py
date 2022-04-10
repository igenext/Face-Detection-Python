import numpy as np
import cv2 
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while(True):  
         
    ret, frame = cap.read()  
    
    image = frame
         
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
      
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    faces_rects = haar_cascade_face.detectMultiScale(image_gray, scaleFactor = 1.2, minNeighbors = 5)

    print('Faces found: ', len(faces_rects))
        # Display the resulting frame  
    
    for (x,y,w,h) in faces_rects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame',image)  
    
    if cv2.waitKey(1) & 0xFF == ord('c'):  
        break  
    
cap.release()  
cv2.destroyAllWindows()  
