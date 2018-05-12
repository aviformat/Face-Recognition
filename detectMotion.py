import cv2
#inport cv
import numpy as np
from pymongo import MongoClient
import base64
#from pymongo import Binary

client=MongoClient("mongodb://iot:iot@ds119049.mlab.com:19049/iot")
db=client.iot
output=""
cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()
count=0
while (True):
#     print(1)
    ret, frame = cap.read()
#     print(1)
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    n_white_pix = np.sum(fgmask == 255)
    # print n_white_pix
    if n_white_pix>1000:
        # print count
        count+=1
    if count>100:
        cv2.imwrite('blah1.jpg',frame)
        output="yes"
        db.security1.delete_many({})

        db.security1.insert_one(
            {
                "result": output
            }
        )
    else:
        output="no"
        db.security1.delete_many({})

        db.security1.insert_one(
            {
                "result": output
            }
        )

        cursor = db.security1.find({})
        for docs in cursor:
            print docs['result']




    #cursor=db.security.find({})


    cv2.imshow('frame',fgmask)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()