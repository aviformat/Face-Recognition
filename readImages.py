import cv2
import os
import numpy as np
import sys
from sklearn import svm

def read_images(path,sz=None):
    c=0
    X,y=[],[]

    for dirname,dirnames,filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path=os.path.join(dirname,subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if (filename == ".directory"):
                        continue
                    filepath=os.path.join(subject_path,filename)
                    im=cv2.imread(os.path.join(subject_path,filename),cv2.IMREAD_GRAYSCALE)
                    if (sz is not None):
                        im = cv2.resize(im,(200,200))

                    X.append(np.asarray(im,dtype=np.uint8))
                    y.append((c))
                except IOError,(errno,strerror):
                    print "I/O error({0}):{1}".format(errno,strerror)
                except:
                    print "Unexpected error:",sys.exc_info()[0]
                    raise
            c=c+1
    return [X,y]

if __name__=="__main__":

    out_dir=None
    names=['abhishek','avi']
    print sys.argv
    if len(sys.argv)<2:
        print "USAGE: facerec_demo.py </path/to/images> [</path/to/store/images/avi>]"
        sys.exit()


    [X,y]=read_images(sys.argv[1])
    #print X,y
    y=np.asarray(y,dtype=np.int32)
    print "hello"
    if len(sys.argv) == 3:
        out_dir=sys.argv[2]
    # model=cv2.face.LBPHFaceRecognizer_create()
    # model.train(np.asarray(X),np.asarray(y))
    X=np.asarray(X)
    y=np.asarray(y)
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples, nx * ny))

    clf = svm.SVC()
    clf.fit(np.asarray(X), np.asarray(y))
    #print X.shape,y
    params=[]
    camera=cv2.VideoCapture(0)
    face_cascade=cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    while(True):
        read,img=camera.read()
        faces=face_cascade.detectMultiScale(img,1.3,5)
        for (x,y,w,h) in faces:
            img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            roi=gray[x:x+w,y:y+h]
            try:
                #print "1"
                roi=cv2.resize(roi,(200,200),interpolation=cv2.INTER_LINEAR)
                #params=model.predict(roi)
                roi=roi.ravel()
                #print roi.shape
                #print "2"
                roi=roi.reshape(1,-1)
                #params=clf.predict(roi)
                #print clf.score(roi,sample_weight=None)
                params = clf.predict(roi)
                #a,res=clf.predict(roi,flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
                #print res
                print "hello"
                print params
                #print "Label: %s,Confidence : %.2f"%(params[0],params[1])
                #if params[1]>70:
                cv2.putText(img,names[params[0]],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
            except:
                #print "hi"
                continue

        cv2.imshow("camera",img)
        if cv2.waitKey(1000/12) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()

