import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
mpFaceMash = mp.solutions.face_mesh
faceMeh = mpFaceMash.FaceMesh(max_num_faces=10)
drawSpac = mpDraw.DrawingSpec(color=(255,0,55),thickness=1,circle_radius=1)
ptime =0
while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = faceMeh.process(imgRGB)
    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms,mpFaceMash.FACEMESH_CONTOURS,drawSpac,drawSpac)
            for id, lms in enumerate (faceLms.landmark):
                ih,iw,ic = img.shape
                x , y = int(lms.x * iw) , int(lms.y * ih)
                print(id,x , y)

    ctime =time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img,f"{int(fps)}",(28,78),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv2.imshow("image",img)
    cv2.waitKey(1)