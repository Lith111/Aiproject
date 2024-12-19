import cv2
import time
from poseModule import poseDetector


cap = cv2.VideoCapture(0)
cTime = 0
pTime = 0
pose = poseDetector()
while True:
    success, img = cap.read()
    img = pose.findpose(img, draw=True)
    lmlist = pose.getpoition(img, draw=True)
    if  len(lmlist) != 0 :
        print(lmlist[1])
        cv2.circle(img, (lmlist[1][1], lmlist[1][2]), 10, (0, 0, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    cv2.imshow("Images", img)
    cv2.waitKey(1)
