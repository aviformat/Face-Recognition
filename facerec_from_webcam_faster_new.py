# import os
import face_recognition
import pickle
import cv2
import dlib
from collections import Counter
# from imutils import face_utils
from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import numpy as np
from scipy.spatial import distance as dist

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def adjust_gamma(image, gamma):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

clf2 = SVC(kernel='linear', probability=True, tol=1e-3)
face_cascade_haar = cv2.CascadeClassifier('haarcascade_profileface.xml')
face_cascade_haar1 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
encodings = []
imgNameList = []




#opens the training model 
f=open("trainnew.dat","rb")
clf2 = pickle.load(f)




# Initialize some variables

face_locations = []

face_encodings = []
final_name=[]
neck_dir=[]
face_names = []
face_proba = []

process_this_frame = True

capture=cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
j=0
contraint=2
while True:

    # Grab a single frame of video

    ret, frame = capture.read()

    frame1 = frame.copy()
    frame1 = cv2.flip(frame1, 1)

    # print (frame.shape)
    small_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    small_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # Resize frame of video to 1/4 size for faster face recognition processing
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

    small_frame=adjust_gamma(small_frame,1.5)
    rgb_small_frame = frame[:, :, ::-1]

    small_frame1 = adjust_gamma(small_frame1, 1.5)
    rgb_small_frame1 = frame1[:, :, ::-1]


    faces = face_cascade_haar.detectMultiScale(small_frame,1.1,11,0,(20,20),(400,400))
    faces1 = face_cascade_haar.detectMultiScale(small_frame1,1.1,11,0,(20,20),(400,400))
    faces2 = face_cascade_haar1.detectMultiScale(small_frame,1.1,11,0,(20,20),(400,400))
    if len(faces2)>0:
        for x, y, w, h in faces2:
            # print ("frontal")
            loc = [y, x + w, y + h, x]
            # print (loc)
            # print(str(w) + '*' + str(h))
            # loca=list(tuple(loc))
            # print (loca)
            loca = list(zip(loc))
            l1, l2 = [], []
            # print (loca)
            # print("front")
            neck_dir.append("front")
            for i in range(3):
                loca[0] = (loca[0] + loca[i + 1])
            # print (loca[0])


            face_encodings = face_recognition.face_encodings(rgb_small_frame, [loca[0]])

            # for x,y,w,h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face_names = []
            name = "Unknown"
            if face_encodings:
                face_pred = clf2.predict(face_encodings)
                face_proba = clf2.predict_proba(face_encodings)

                name = face_pred
            # f.close()
            face_names.append(name)
            final_name.append(str(name))
            
            cv2.putText(frame, str(name), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            
    elif len(faces)>0:
        for x,y,w,h in faces:
            loc=[y,x+w,y+h,x]
            # print (loc)
            # print(str(w) + '*' + str(h))
            # loca=list(tuple(loc))
            # print (loca)
            loca = list(zip(loc))
            l1, l2 = [], []
            # print (loca)
            for i in range(3):
                loca[0] = (loca[0] + loca[i + 1])
            # print (loca[0])


            face_encodings = face_recognition.face_encodings(rgb_small_frame, [loca[0]])

            # for x,y,w,h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            neck_dir.append("right")
            
            face_names = []
            name = "Unknown"
            if face_encodings:
                face_pred = clf2.predict(face_encodings)
                face_proba = clf2.predict_proba(face_encodings)

                name = face_pred
            
            
            cv2.putText(frame, str(name), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255,
                        2)

    
    elif len(faces1)>0:
        for x, y, w, h in faces1:
            loc = [y, x + w, y + h, x]
            # print (loc)
            # print(str(w) + '*' + str(h))
            # loca=list(tuple(loc))
            # print (loca)
            loca = list(zip(loc))
            l1, l2 = [], []
            # print (loca)
            for i in range(3):
                loca[0] = (loca[0] + loca[i + 1])
            # print (loca[0])

            # print("left")
            neck_dir.append("left")
            face_encodings = face_recognition.face_encodings(rgb_small_frame1, [loca[0]])

            # for x,y,w,h in faces:
            cv2.rectangle(frame, (640 - (x + w), y), (640 - x, y + h), (255, 0, 0), 2)
            
            face_names = []
            name = "Unknown"
            if face_encodings:
                face_pred = clf2.predict(face_encodings)
                face_proba = clf2.predict_proba(face_encodings)

                name = face_pred
            f.close()
            
            cv2.putText(frame, str(name), (640 - (x + w), y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

  


    #
    j+=1
    if j>30:
        break
   
    # Display the resulting image

    cv2.imshow('Video', frame)



    # Hit 'q' on the keyboard to quit!

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break


counter=Counter(final_name)
counter=counter.most_common()

print (counter[0][0])

direc=Counter(neck_dir).values()
# print (direc)
count=0
for dir in direc:
    if int(dir)>=contraint:
        count+=1
if count==3:
    print ("original face")
else:
    print ("fake face")
capture.release()

cv2.destroyAllWindows()
