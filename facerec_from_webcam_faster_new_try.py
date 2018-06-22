# import os
# import face_recognition
# import pickle
import cv2
# from imutils import face_utils
# import imutils
import dlib
import numpy as np
from scipy.spatial import distance as dist
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def adjust_gamma(image, gamma):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def calc_ear(eye):
    # print (eye[0][1:])
    A=dist.euclidean(eye[1][1:],eye[5][1:])
    B=dist.euclidean(eye[2][1:],eye[4][1:])
    C=dist.euclidean(eye[0][1:],eye[3][1:])

    ear1=(A+B)/(2.0*C)
    A = dist.euclidean(eye[7][1:], eye[11][1:])
    B = dist.euclidean(eye[8][1:], eye[10][1:])
    C = dist.euclidean(eye[6][1:], eye[9][1:])

    ear2 = (A + B) / (2.0 * C)
    return (ear1+ear2)/2.0


def rect_to_bb(rect):
   
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):

    coords = np.zeros((12, 3), dtype=dtype)
    for i in range(0, 68):
        if i>=36 and i<=47:
            coords[i-36] = (i-36,shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("face_landmarks.dat")
face_cascade_haar = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

capture=cv2.VideoCapture(0)
X,Y=[],[]
count=0
blah=0
blink,width=[],[]
num_blink=0
j=0
min_dist=2
threshold=0.06
while True:
    ret,frame=capture.read()
    frame=cv2.resize(frame,(500,500))
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray=adjust_gamma(frame_gray,1.5)
    rects=detector(frame_gray,1)
    # print (rect)
    for (i, rect) in enumerate(rects):
        
        (x, y, w, h) = rect_to_bb(rect)
       
        shape = predictor(frame_gray, rect)

        # print (shape)
        shape = shape_to_np(shape)
        

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame,"No. of blinks: "+str(num_blink),(10,30),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,127,255),3)
        # show the face number
        if num_blink<3:
            cv2.putText(frame, "fake Face".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "real Face".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (n,x, y) in shape:
            if n<6:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        ear=calc_ear(shape)
        # den=w
        width.append(w)
        
        blink.append(ear)

        X.append(count)
        Y.append(ear)
        if (len(Y)>=min_dist*2+1):
            if (Y[count-(min_dist*2)]-Y[count-min_dist]>threshold or Y[count]-Y[count-min_dist]>threshold) and (abs(width[count-(min_dist*2)]-width[count])<10):
                num_blink+=1


        count += 1
        j+=1

    plt.plot(X,Y)
    plt.pause(0.042)
    #

    cv2.imshow("window",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
plt.show()
print (width)
print (blink)


