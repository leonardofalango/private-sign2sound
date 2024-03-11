import cv2
import time
import logging
import pandas as pd
import mediapipe as mp

set_label = "A"
path = "var/ASL/data.csv"

mp_hands = mp.solutions.hands.Hands(
    max_num_hands=2,
)
mp_drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands

lms = [[] for x in range(21)]
label = []

logging.basicConfig(
    filename='var/logs/hand_detector.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

try:
    cap = cv2.VideoCapture(0)
except Exception as e:
    logging.error(e)

logging.info("Application is running")
start = time.time()
while True:
    current = time.time()
    ret, frame = cap.read()
    if ret == False:
        break

    # Find the hand and its landmarks
    img = mp_hands.process(frame)
    if img.multi_hand_landmarks:
        logging.debug(img.multi_hand_landmarks)
        for hand_landmarks in img.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, hands.HAND_CONNECTIONS)
            

    
    if current - start >= 60:
        logging.debug("FPS: %s", str(60/(current - start)))
        start = time.time()

    cv2. imshow("Image", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('m'):
        if img.multi_hand_landmarks:
            for hand_landmarks in img.multi_hand_landmarks:
                for lm_index in range(21):
                    lms[lm_index].append((hand_landmarks.landmark[lm_index].x, hand_landmarks.landmark[lm_index].y, hand_landmarks.landmark[lm_index].z))
        
                label.append(set_label)
        else:
            logging.error("No hand detected")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

data = {}
for i in range(21):
    data['label'] = set_label
    for j in range(len(lms)):
        data[f'landmark_{i}'] = lms[i][j]
                  
df = pd.DataFrame(data)

df_old = pd.read_csv(path)
df = pd.concat([df, df_old], ignore_index=True)
df.to_csv(path, index=False)

cap.release()
cv2.destroyAllWindows()