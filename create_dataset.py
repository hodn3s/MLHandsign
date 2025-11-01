import os
import pickle
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

# Path to dataset
DATA_DIR = './data'

data = []
labels = []

# Iterate through all subfolders in the dataset directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Skip anything that isn’t a directory (e.g., .gitignore)
    if not os.path.isdir(dir_path):
        print(f"⚠️ Skipping non-directory: {dir_path}")
        continue

    print(f"📁 Processing folder: {dir_}")

    # Process each image in the subfolder
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)

        # Skip non-image files
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"⏭️ Skipping non-image file: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Could not read image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Save processed data
output_file = 'data.pickle'
with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n✅ Dataset successfully saved to '{output_file}'")
print(f"📊 Total samples: {len(data)}")
