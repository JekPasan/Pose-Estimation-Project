# --------------------
# ---- the module ----
# --------------------

import cv2
import mediapipe as mp
import time

class PoseEstimator():
    def __init__(self, static_mode=False, upper_body_only=False, smoothness=True, detection_confidence=0.5, tracking_confidence=0.5):
        self.static_mode = static_mode
        self.upper_body_only = upper_body_only
        self.smoothness = smoothness
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.cap = cv2.VideoCapture(0)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.static_mode, self.upper_body_only, self.smoothness, self.detection_confidence, self.tracking_confidence)

        self.mp_draw = mp.solutions.drawing_utils
    
    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img
    
    def find_position(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cX, cY = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cX, cY])
                if draw and (id == 0 or id == 1 or id == 4 or id == 13 or id == 14):
                    cv2.circle(img, (cX,cY), 6, (132, 44, 76), cv2.FILLED)
        return lm_list
        

def main():
    cap = cv2.VideoCapture(0)

    current_time = 0
    previous_time = 0
    frames = 0

    detector = PoseEstimator()

    while True:
        succes, img = cap.read()
        img = detector.find_pose(img)
        lm_list = detector.find_position(img)
        while current_time == previous_time:
            current_time = time.time()
        fps = 1/(current_time-previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (5, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

        cv2.imshow("Video", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()