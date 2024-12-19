import time
import mediapipe as mp
import cv2


class faceMeah():
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMash = mp.solutions.face_mesh
        self.faceMeh = self.mpFaceMash.FaceMesh()
        self.drawSpac = self.mpDraw.DrawingSpec(color=(255, 0, 55),
                                                thickness=1,
                                                circle_radius=1)

    def findMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.faceMeh.process(self.imgRGB)
        faces = []
        if self.result.multi_face_landmarks:
            for faceLms in self.result.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMash.FACEMESH_CONTOURS,
                                               self.drawSpac, self.drawSpac)
                face = []
                for id, lms in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lms.x * iw), int(lms.y * ih)
                    cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.3,
                                (255  , 0, 0), 1)
                    face.append([x, y])
            faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
    fm = faceMeah()
    while True:
        success, img = cap.read()
        img, faces = fm.findMesh(img, draw=False)
        if len(faces) != 0:
            print(faces[0][12])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Images", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
