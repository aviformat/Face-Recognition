import cv2

def detect():
    face_cascade=cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt2.xml')
    lefteye_cascade=cv2.CascadeClassifier('./cascades/haarcascade_lefteye_2splits.xml')
    righteye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_righteye_2splits.xml')
    camera=cv2.VideoCapture(0)
    while(True):
        ret,frame=camera.read()

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            roi_gray=gray[y:y+h,x:x+w]
            lefteye=lefteye_cascade.detectMultiScale(roi_gray,1.2,5,0,(50,50))
            righteye = righteye_cascade.detectMultiScale(roi_gray, 1.2, 5, 0, (50, 50))
            for (ex,ey,ew,eh) in lefteye:
                cv2.rectangle(img[y:y+h,x:x+w],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            for (ex,ey,ew,eh) in righteye:
                cv2.rectangle(img[y:y+h,x:x+w],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow("camera",frame)
        if cv2.waitKey(1000/12) & 0xff == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    detect()
