
import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

extract_dir = "Dataset_FINAL2"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parameters - adjust as needed
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7

data = []

# Use context manager so resources are cleaned up
with mp_hands.Hands(static_image_mode=True,
                    max_num_hands=MAX_HANDS,
                    min_detection_confidence=MIN_DETECTION_CONFIDENCE) as hands:

    for label_folder in os.listdir(extract_dir):
        folder_path = os.path.join(extract_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue

        for img_file in tqdm(os.listdir(folder_path), desc=f"Processing {label_folder}"):
            img_path = os.path.join(folder_path, img_file)

            # Read image safely
            image = cv2.imread(img_path)
            if image is None:
                # skip non-image files
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                results = hands.process(image_rgb)
            except Exception:
                # if MediaPipe fails on some images, skip them
                continue

            if results.multi_hand_landmarks:
                # results.multi_handedness gives handedness info parallel to landmarks
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    features = []
                    for lm in hand_landmarks.landmark:
                        # normalized coordinates (x,y) in [0,1] relative to image
                        features.extend([lm.x, lm.y, lm.z])

                    # get handedness if available ("Left" / "Right")
                    handedness = None
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        try:
                            handedness = results.multi_handedness[idx].classification[0].label
                        except Exception:
                            handedness = None

                    # Append metadata: handedness, filename, label
                    features.append(handedness if handedness is not None else "")
                    features.append(img_file)
                    features.append(label_folder)
                    data.append(features)

# Build columns: x0,y0,z0,...,x20,y20,z20,handedness,filename,label
coords = [f"{axis}{i}" for i in range(21) for axis in ['x', 'y', 'z']]
columns = coords + ['handedness', 'filename', 'label']

df = pd.DataFrame(data, columns=columns)

# Save CSV
out_csv = "features_alphabet.csv"
df.to_csv(out_csv, index=False)
print(f"done — saved {out_csv}")