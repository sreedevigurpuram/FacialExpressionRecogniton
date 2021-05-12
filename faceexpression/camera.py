import cv2
from model import FacialExpressionModel
import numpy as np
# video capture
rgb = cv2.VideoCapture(0)
print("capture")
# face detector
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def __get_data__():
    st,fr = rgb.read()
    #gray scale conversion
    print(fr)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    # face detection
    faces = facec.detectMultiScale(gray, 1.3, 5)
    
    return faces, fr, gray

def start_app(cnn):
    EMOTIONS_LIST = ["Angry", "Disgust","Fear", "Happy",
                     "Sad", "Surprise","Neutral"]
    lis=[]
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    # contineous frame reading
    while True:
        ix += 1
    
        faces, fr, gray_fr = __get_data__()
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            
            roi = cv2.resize(fc, (48, 48))
            # emotion prediction from face
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            # add text in the window
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            lis.append(pred)
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Out', fr)
    cv2.destroyAllWindows()
    dic={}
    for i in EMOTIONS_LIST:
        dic[i]=lis.count(i)
    print("dictionary")
    return dic

def main():
    # load trained models
    model = FacialExpressionModel("face_model.json", "face_model.h5")
    values=start_app(model)
    return values
##main()
