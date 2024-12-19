import cv2
import mediapipe as mp
import time


class faceDetection():
    def __init__(self):
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faseDetection = self.mpFaceDetection.FaceDetection(0.75)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.faseDetection.process(imgRGB)
        bbboxs = []
        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                bboxc = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), \
                    int(bboxc.width * iw), int(bboxc.height * ih)
                bbboxs.append([id, bbox, detection.score])
                if draw:
                    self.facyDraw(img,bbox)
                    cv2.putText(img,
                                f"{int(detection.score[0] * 100)}",
                                (bbox[0], bbox[1] - 30),
                                cv2.FONT_HERSHEY_PLAIN,
                                2,
                                (255, 0, 255),
                                2)
        return img, bbboxs
    def facyDraw(self,img,bbox,l=30,t=7,rt=1):
        x,y,w,h = bbox
        x1,y1 = x+w,y+h
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        cv2.line(img,(x,y),(x+l,y),(255,0,255),t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return  img

def main():
    cap = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
    detec = faceDetection()
    while True:
        success, img = cap.read()
        img,bboxs = detec.findFaces(img)
        print(bboxs)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Images", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
