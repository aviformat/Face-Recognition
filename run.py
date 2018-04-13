import cv2
import os
import numpy as np
import sys
from sklearn import svm
import collections
import pickle

params=[]
names=['abhishek','ankit','avi','shahsank','utsav']
hog = cv2.HOGDescriptor()
camera=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt.xml')

clf = pickle.load(open('finalized_model.sav', 'rb'))
while(True):
        read,img=camera.read()

        faces=face_cascade.detectMultiScale(img,1.3,5)
        for (x,y,w,h) in faces:
            img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

            # try1 = np.asarray(img)
            # cv2.imshow("blah", try1)
            # cv2.waitKey(0)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # roi=gray[x:x+w,y:y+h]
            roi=gray[y:y + h, x:x + w]

            roi=cv2.resize(roi,(64,128))
            # try1 = np.asarray(roi)
            # cv2.imshow("blah", try1)
            # cv2.waitKey(0)
            roi=hog.compute(roi)
            #print roi.shape

            # cv2.destroyAllWindows()
            roi = np.asarray(roi)
            #print roi.shape
            roi=roi.ravel()
            # im1 = cv2.GaussianBlur(roi, (5, 5), 0)
            # im2 = cv2.GaussianBlur(roi, (3, 3), 0)
            # Dog_img = im2 - im1
            # Dog_img=hog.compute(Dog_img)
            # roi = Dog_img.ravel()
            roi=roi.reshape(1,-1)
            #print len(roi[0])
            #roi = list(roi)
            #print roi

            params = clf.predict(roi)
            # class_probabilities = clf.predict_proba(roi)
            # print class_probabilities
            # print params
            # print "Label: %d"%(params[0])
            cv2.putText(img,str(names[params[0]]),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)

        cv2.imshow("camera",img)
        if cv2.waitKey(1000/12) & 0xff == ord("q"):
            break
cv2.destroyAllWindows()

