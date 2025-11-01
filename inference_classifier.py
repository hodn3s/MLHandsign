import pickle
import cv2
import mediapipe as mp
import numpy as np

# --- Load trained model ---
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# --- Camera setup (auto-detect working index) ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ Camera index 0 not found. Trying index 1...")
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ No available camera detected. Exiting...")
    exit()

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
    13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}


print("✅ Inference started. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Failed to grab frame.")
        continue

    H, W, _ = frame.shape
    yellow_overlay = np.full(frame.shape, (0, 255, 255), dtype=np.uint8)  # BGR for yellow
    frame = cv2.addWeighted(frame, 0.5, yellow_overlay, 0.5, 0)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux, x_, y_ = [], [], []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        # Bounding box coordinates
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # --- Predict gesture ---
        data_np = np.asarray(data_aux).reshape(1, -1)
        prediction = model.predict(data_np)
        prediction_proba = model.predict_proba(data_np)

        predicted_class = int(prediction[0])
        predicted_label = labels_dict[predicted_class]
        confidence = np.max(prediction_proba) * 100

        # --- Display on screen ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.putText(
            frame,
            f'{predicted_label} ({confidence:.1f}%)',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )

    cv2.imshow('Sign Language Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting inference loop.")
        break

cap.release()
cv2.destroyAllWindows()
