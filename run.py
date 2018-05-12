import cv2
import os
import numpy as np
import sys
from sklearn import svm
import collections
import pickle
from pymongo import MongoClient

params=[]
count=0
name1,outputs="",""

names=['abhishek','ankit','avi','ravi','shahsank','vinayak']
hog = cv2.HOGDescriptor()
camera=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt.xml')

clf = pickle.load(open('final.sav', 'rb'))
client = MongoClient("mongodb://iot:iot@ds119049.mlab.com:19049/iot")
db = client.iot
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
            roi = cv2.bilateralFilter(roi, 8, 50, 50)
            # try1 = np.asarray(roi)
            # cv2.imshow("blah", try1)
            # cv2.waitKey(0)
            roi=hog.compute(roi)
            #print roi.shape

            # cv2.destroyAllWindows()
            # roi = np.asarray(roi)
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
            #print params
            class_probabilities = clf.predict_proba(roi)
            #print class_probabilities
            # print params[0]
            # print class_probabilities[0][params[0]]

            # print params
            # print "Label: %d"%(params[0])
            if max(class_probabilities[0])>0.60:
                cv2.putText(img,str(names[params[0]]),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
                name1 = names[params[0]]
                outputs = "Authorized Person Recognized"
                count += 1



            else:
                cv2.putText(img,'Unknown',(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
                name1 = "NA"
                outputs =  "Unknown Person Detected"
                count += 1


        if count>50:
            break

        cv2.imshow("camera",img)
        if cv2.waitKey(1000/12) & 0xff == ord("q"):
            break
db.security.delete_many({})

db.security.insert_one(
    {
        "name": name1,
        "result": outputs
    }
)

cursor = db.security.find({})
for docs in cursor:
    print docs['name']
    print docs['result']
cv2.destroyAllWindows()

