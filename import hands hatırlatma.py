import cv2
import mediapipe as mp
import time 


cap=cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

ctime=0
ptime=0

mpHand=mp.solutions.hands
hands=mpHand.Hands()
mpDraw=mp.solutions.drawing_utils

while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    results=hands.process(imgRGB)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHand.HAND_CONNECTIONS)

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv2.putText(img , "FPS " +str(int(fps)),(10,75),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)




    cv2.imshow("img",img)
    cv2.waitKey(1)
