
import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

extract_dir = "Alphabet"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7

data = []

with mp_hands.Hands(static_image_mode=True,
                    max_num_hands=MAX_HANDS,
                    min_detection_confidence=MIN_DETECTION_CONFIDENCE) as hands:

    for label_folder in os.listdir(extract_dir):
        folder_path = os.path.join(extract_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue

        for img_file in tqdm(os.listdir(folder_path), desc=f"Processing {label_folder}"):
            img_path = os.path.join(folder_path, img_file)

            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                results = hands.process(image_rgb)
            except Exception:
                continue

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    features = []
                    for lm in hand_landmarks.landmark:
                        features.extend([lm.x, lm.y, lm.z])

                    handedness = None
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        try:
                            handedness = results.multi_handedness[idx].classification[0].label
                        except Exception:
                            handedness = None

                    features.append(handedness if handedness is not None else "")
                    features.append(img_file)
                    features.append(label_folder)
                    data.append(features)

coords = [f"{axis}{i}" for i in range(21) for axis in ['x', 'y', 'z']]
columns = coords + ['handedness', 'filename', 'label']

df = pd.DataFrame(data, columns=columns)

out_csv = "features_alphabet.csv"
df.to_csv(out_csv, index=False)
print(f"done — saved {out_csv}")
