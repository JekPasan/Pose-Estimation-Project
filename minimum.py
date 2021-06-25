# -----------------------------------------
# ---- the bare minimum necessary code ----
# -----------------------------------------

import cv2
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils

# video = input("what video do you want to proccess? ")
cap = cv2.VideoCapture(0)

current_time = 0
previous_time = 0
frames = 0

while True:
    succes, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    while current_time == previous_time:
        current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (5, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

    cv2.imshow("Video", img)
    cv2.waitKey(1)