import cv2
import time
import pickle
import logging
import numpy as np
import pandas as pd
import mediapipe as mp

set_label = "A"
path = "var/ASL/data.csv"
running = True
fps = 0

mp_hands = mp.solutions.hands.Hands(
    max_num_hands=2,
)
mp_drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands

font = cv2.FONT_HERSHEY_SIMPLEX

aux = [0 for init in range(21 * 3)]
x = [0 for init in range(21 * 3)]
data_list = []


logging.basicConfig(
    filename="var/logs/hand_detector.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

try:
    cap = cv2.VideoCapture(0)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

except Exception as e:
    logging.error(e)

logging.info("Application is running")

start = time.time()
while running:
    current = time.time()
    ret, frame = cap.read()
    if ret == False:
        break

    img = mp_hands.process(frame)
    if img.multi_hand_landmarks:
        logging.debug(img.multi_hand_landmarks)
        for hand_landmarks in img.multi_hand_landmarks:
            for lm_index in range(21):
                x[lm_index] = hand_landmarks.landmark[lm_index].x
                x[lm_index + 21] = hand_landmarks.landmark[lm_index].y
                x[lm_index + 21 * 2] = hand_landmarks.landmark[lm_index].z

            mp_drawing.draw_landmarks(frame, hand_landmarks, hands.HAND_CONNECTIONS)

            df = pd.DataFrame(x).T
            pred = model.predict(df)

            cv2.putText(
                frame,
                str(pred),
                (10, 80),
                font,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            logging.debug("Prediction %s", pred)

    if current - start > 1:
        fps = 60 / (current - start)
        start = time.time()
        logging.debug("FPS: %s", fps)

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        font,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord("m"):
        if img.multi_hand_landmarks:
            for hand_landmarks in img.multi_hand_landmarks:
                for lm_index in range(21):
                    aux[lm_index] = hand_landmarks.landmark[lm_index].x
                    aux[lm_index + 21] = hand_landmarks.landmark[lm_index].y
                    aux[lm_index + 21 * 2] = hand_landmarks.landmark[lm_index].z
                data_list.append(aux)
        else:
            logging.error("No hand detected")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False

labels = [set_label for i in range(len(data_list))]
df = pd.DataFrame(data_list)
df.T
df["label"] = labels

logging.critical(df)

df.to_csv(path, index=False)

cap.release()
cv2.destroyAllWindows()
