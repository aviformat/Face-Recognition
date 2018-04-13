import cv2
import os
import numpy as np
import sys
from sklearn import svm
import collections
import pickle

X,y=[],[]
names=[]
hog = cv2.HOGDescriptor()
c=0
for dirname,dirnames,filenames in os.walk("data/"):
    dirnames.sort()
    for subdir in dirnames:
        a=map(str,subdir.split('\n'))
        names.append(a)
        subject_path = os.path.join(dirname, subdir)
        for filename in os.listdir(subject_path):

            im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, (64, 128))
            # try1 = np.asarray(im)
            # cv2.imshow("blah", try1)
            # cv2.waitKey(0)
            # im1 = cv2.GaussianBlur(img,(5,5),0)
            # im2 = cv2.GaussianBlur(img, (3, 3), 0)
            # im=im2-im1
            im=hog.compute(im)
            X.append(im)
            y.append(c)

        c+=1



print os.path.join(subject_path, filename), c
# cv2.imshow('image',im)
# cv2.waitKey(0)
X=np.asarray(X)
print X.shape
nsamples, nx, ny = X.shape
X = X.reshape((nsamples, nx * ny))

print X.shape
print collections.Counter(y)
print names

clf=svm.SVC(C=1,kernel='rbf',probability=True)
clf.fit(np.asarray(X),np.asarray(y))
# clf=cv2.face.LBPHFaceRecognizer_create()
# clf.train(np.asarray(X),np.asarray(y))
filename = 'rbfTrue.sav'
pickle.dump(clf, open(filename, 'wb'))


# params=[]
# camera=cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt.xml')
#
#
# while(True):
#         read,img=camera.read()
#
#         faces=face_cascade.detectMultiScale(img,1.3,5)
#         for (x,y,w,h) in faces:
#             img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
#
#             # try1 = np.asarray(img)
#             # cv2.imshow("blah", try1)
#             # cv2.waitKey(0)
#             gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#             # roi=gray[x:x+w,y:y+h]
#             roi=gray[y:y + h, x:x + w]
#
#             roi=cv2.resize(roi,(64,128))
#             # try1 = np.asarray(roi)
#             # cv2.imshow("blah", try1)
#             # cv2.waitKey(0)
#             roi=hog.compute(roi)
#             #print roi.shape
#
#             # cv2.destroyAllWindows()
#             roi = np.asarray(roi)
#             #print roi.shape
#             roi=roi.ravel()
#             # im1 = cv2.GaussianBlur(roi, (5, 5), 0)
#             # im2 = cv2.GaussianBlur(roi, (3, 3), 0)
#             # Dog_img = im2 - im1
#             # Dog_img=hog.compute(Dog_img)
#             # roi = Dog_img.ravel()
#             roi=roi.reshape(1,-1)
#             #print len(roi[0])
#             #roi = list(roi)
#             #print roi
#             params = clf.predict(roi)
#             # class_probabilities = clf.predict_proba(roi)
#             # print class_probabilities
#             # print params
#             # print "Label: %d"%(params[0])
#             cv2.putText(img,str(names[params[0]]),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
#
#         cv2.imshow("camera",img)
#         if cv2.waitKey(1000/12) & 0xff == ord("q"):
#             break
# cv2.destroyAllWindows()
