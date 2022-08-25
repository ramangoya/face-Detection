import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib
import mtcnn
image =cv2.imread("face.jpg")
#1load the Image
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image=cv2.resize(image,(500,300))
#2.Convert Image  to gray scale
gray_scale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
rects=detector.detectMultiScale(gray_scale,scaleFactor=1.05)
for (x,y,w,h) in rects:
  cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
plt.imshow(image)
plt.show()
