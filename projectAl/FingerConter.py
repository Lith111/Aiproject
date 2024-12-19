import os
import time
import cv2
from Hand import HandTrsckingModule as htm

wcam, hcam = 648, 488
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
folderpath = "../fingers"
overlayList = []
myList = os.listdir(folderpath)

for imgpath in myList:
    image = cv2.imread(f"{folderpath}/{imgpath}")
    overlayList.append(image)
# print(myList)
print(len(overlayList))
ptime = 0
detctor = htm.handDetector()
tipIds = [4, 8, 12, 16, 20]
while True:
    sccuses, img = cap.read()
    img = detctor.findHands(img)
    lmList = detctor.findPosition(img, draw=False)
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totelFingers = fingers.count(1)
        print(totelFingers)
        h, w, c = overlayList[totelFingers].shape
        img[0:h, 0:w] = overlayList[totelFingers]
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    cv2.imshow("Fingers controers", img)
    cv2.waitKey(1)
