import  cv2
import numpy as np
import time
import  os
from Hand import HandTrsckingModule as htm
folderPath = "header"
myList = os.listdir(folderPath)
overLayList = []
brushThich = 15
errorThich = 120
xp , yp = 0 , 0
imgCanvas = np.zeros((720,1280,3),np.uint8)
for imgPath in myList:
    img = cv2.imread(f'{folderPath}/{imgPath}')
    overLayList.append(img)
header = overLayList[0]
drawColor = (255,0,255)
cap = cv2.VideoCapture(0)
cap.set(3,1200)
cap.set(4,720)
detector = htm.handDetector()
while True :
    # 1. Import image
    success,img = cap.read()
    img = cv2.flip(img,1)
    # 2. find Hand Landmarks
    img = detector.findHands(img)
    lmlist = detector.findPosition(img,0,False)
    if len(lmlist) != 0:
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]
    # 3. check which fingers are up
    fingers = detector.fingersUp()
    if len(fingers)!=0:
        if fingers[1]and fingers[2]:
            xp, yp = 0, 0
            if y1< 125:
                if 500 < x1 <640:
                    header = overLayList[1]
                    drawColor = (255,0,255)
                if 680 < x1 < 800:
                    header = overLayList[2]
                    drawColor = (0,0,255)
                if 810 < x1 <940:
                    header = overLayList[3]
                    drawColor = (0,255,0)
                if 990 < x1 <1120:
                    header = overLayList[4]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        if fingers[1] and fingers[2]== False:
            cv2.circle(img, (x1, y1 ), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp,yp = x1 ,y1
            if drawColor == (0,0,0):
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor,errorThich)
                cv2.line(img, (xp, yp), (x1, y1), drawColor, errorThich)
            else:
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThich)
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThich)

            xp, yp = x1, y1
    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)
    img[0:125,0:1278] = header
    cv2.imshow("test",img)
    # cv2.imshow("Drawig",imgCanvas)
    cv2.waitKey(1)

