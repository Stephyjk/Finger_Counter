import cv2
import numpy as np
import time
import math
import HandTrackingModule as htm
import os


class VideoCapture(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        folderPath = 'FingerImages'
        myList = os.listdir(folderPath)
        overlayList = []
        pTime = 0

        for imPath in myList:
            image = cv2.imread(f'{folderPath}/{imPath}')
            overlayList.append(image)

        detector = htm.handDetector(detectionCon=0.75)

        tipIds = [4, 8, 12, 16, 20]
        ret, img = self.video.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            fingers = []

            # left thumb
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # fingers minus thumb
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)
            h, w, c = overlayList[totalFingers-1].shape
            img[0:h, 0:w] = overlayList[totalFingers-1]

            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (45, 375),
                        cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 20)

        # cTime = time.time()
        # fps = 1/(cTime - pTime)
        # pTime = cTime

        # cv2.putText(img, f'FPS: {int(fps)}', (400, 70),
        #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
