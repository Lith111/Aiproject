import numpy as np
import cv2
import time
from Hand import HandTrsckingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
cap = cv2.VideoCapture(0)
ptime = 0
hand = htm.handDetector()
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0, None)
minVol = volRange[0]
maxVol = volRange[1]

while True:
    success, img = cap.read()
    img = hand.findHands(img,draw=False)
    lmlist = hand.findPosition(img,draw=False)
    if len(lmlist) != 0:
        # print(lmlist[4] , lmlist[8])
        x1, y1 = lmlist[4][1] , lmlist[4][2]
        x2, y2 = lmlist[8][1],lmlist[8][2]
        cx, cy = (x1 + x2 )//2 , (y1 + y2 )//2
        cv2.circle(img,(x1,y1),15,(255,0,0),cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 0, 255), cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3 )
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        lenght = math.hypot(x2 -x1 ,y2 - y1)
        if lenght < 58 :
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        # print(lenght)
        vol = np.interp(lenght ,[50 , 217],[minVol,maxVol])
        print(lenght)
        volume.SetMasterVolumeLevel(vol, None)
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
               (255, 0, 255), 3)
    cv2.imshow("volume", img)
    cv2.waitKey(1)