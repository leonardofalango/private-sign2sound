import os
import cv2
import time
import logging
import pandas as pd
import mediapipe as mp

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
)
mp_drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands

lms = [[] for x in range(21)]
label = []

path = './ASL_Dataset/Train'
logging.basicConfig(
    filename='var/logs/static_image_process.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Application started")
try:    
    for folder in os.listdir(path):
        logging.info("Processing folder: %s/%s", path, folder)
        
        error_counter = 0
        counter = 0
        for file in os.listdir(f"{path}/{folder}"):
            
            logging.debug("Processing file: %s/%s/%s", path, folder, file)
            
            img = cv2.imread(f"{path}/{folder}/{file}")
            img = mp_hands.process(img)
            
            if img.multi_hand_landmarks:
                logging.debug(f"Hand detected in {file}")
                for hand_landmarks in img.multi_hand_landmarks:
                    for lm_index in range(21):
                        lms[lm_index].append((hand_landmarks.landmark[lm_index].x, hand_landmarks.landmark[lm_index].y, hand_landmarks.landmark[lm_index].z))
                    label.append(folder)
                    counter += 1
                
                logging.debug("Hand detected in %s/%s/%s", path, folder, file)
            else:
                logging.debug("No hand detected in %s/%s/%s", path, folder, file)
                error_counter += 1
                
        logging.info("Total images for letter %s: %s", folder, counter)
        logging.info("Total errors for letter %s: %s", folder, error_counter)
        logging.info("Difference for letter %s: %s", folder, counter - error_counter)

except Exception as e:
    
    logging.error(e)
    logging.error("Application is closing")

finally:
    
    df = pd.DataFrame(lms)
    df = df.T
    df['label'] = label
    
    df.to_csv("var/ASL/dataset_data.csv", index=False)
    
    
    logging.info("Data saved to var/ASL/dataset_data.csv")
    logging.info("Application is closing")
