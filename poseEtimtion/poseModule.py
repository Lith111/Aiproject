import math

import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self):
        self.results = None
        self.mpDraw = mp.solutions.drawing_utils
        self.mpose = mp.solutions.pose
        self.pose = self.mpose.Pose()

    def findpose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,
                                           self.results.pose_landmarks,
                                           self.mpose.POSE_CONNECTIONS
                                           )
        return img
    def getpoition (self , img , draw = True):
        self.lmlist= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return self.lmlist

    def findAngel(self,img,p1,p2,p3,draw=True):
        x1,y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]
        # calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2 ,x3 - x2)
                             - math.atan2(y1 - y2,x1 - x2))
        if angle <= 0:
            angle += 360

        if draw :
            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0) )
            cv2.circle(img, (x2, y2), 15, (255, 0, 0))
            cv2.circle(img, (x3, y3), 15, (255, 0, 0),)
            cv2.putText(img,str(int(angle)),(x2-50,y2+50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        return angle
def main():
    cap = cv2.VideoCapture("../videoes/3.mp4")
    cTime = 0
    pTime = 0
    pose = poseDetector()
    while True:
        success, img = cap.read()
        img = pose.findpose(img,draw=False)
        lmlist = pose.getpoition(img,draw=False)
        cv2.circle(img, (lmlist[1][1], lmlist[1][2]), 10, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Images", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
