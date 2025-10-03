import os
import argparse
import cv2
import mediapipe as mp
import joblib
import numpy as np


def main(model_path: str, le_path: str, camera_index: int, max_hands: int = 2):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"Label encoder file not found: {le_path}")

    # Load pipeline (should include scaler inside) and label encoder
    model = joblib.load(model_path, mmap_mode=None)
    label_encoder = joblib.load(le_path, mmap_mode=None)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # More robust MediaPipe Hands initialization
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=max_hands,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                per_hand_predictions = []  # store tuples (label_idx_or_pred, confidence_or_None)

                # First pass: compute predictions/probabilities for every detected hand, still draw landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    features = []
                    for lm in hand_landmarks.landmark:
                        features.extend([lm.x, lm.y, lm.z])

                    features_np = np.array(features).reshape(1, -1)

                    # Ensure runtime feature vector matches the model's expected number of features.
                    expected_n = None
                    try:
                        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
                            expected_n = getattr(model.named_steps['scaler'], 'n_features_in_', None)
                        # fallback to classifier inside pipeline
                        if expected_n is None and hasattr(model, 'named_steps') and 'clf' in model.named_steps:
                            expected_n = getattr(model.named_steps['clf'], 'n_features_in_', None)
                        # final fallback to model itself (if not pipeline)
                        if expected_n is None:
                            expected_n = getattr(model, 'n_features_in_', None)
                    except Exception:
                        expected_n = None

                    if expected_n is not None:
                        cur_n = features_np.shape[1]
                        if cur_n < expected_n:
                            # pad with zeros (assume missing numeric features are zero/neutral)
                            pad_width = expected_n - cur_n
                            features_np = np.hstack([features_np, np.zeros((1, pad_width))])
                            print(f"[deploy] Warning: padded features from {cur_n} to {expected_n} (added {pad_width} zeros)")
                        elif cur_n > expected_n:
                            # truncate extra features
                            features_np = features_np[:, :expected_n]
                            print(f"[deploy] Warning: truncated features from {cur_n} to {expected_n}")

                    probs = None
                    prediction = None

                    # Try predict; handle pipeline or standalone classifier
                    try:
                        prediction = model.predict(features_np)
                    except Exception:
                        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
                            features_scaled = model.named_steps['scaler'].transform(features_np)
                            prediction = model.named_steps['clf'].predict(features_scaled)
                            if hasattr(model.named_steps['clf'], 'predict_proba'):
                                try:
                                    probs = model.named_steps['clf'].predict_proba(features_scaled)
                                except Exception:
                                    probs = None
                        else:
                            raise

                    # Try to obtain probabilities if possible
                    if probs is None:
                        try:
                            if hasattr(model, 'predict_proba'):
                                probs = model.predict_proba(features_np)
                        except Exception:
                            try:
                                if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
                                    if 'scaler' in model.named_steps:
                                        scaled = model.named_steps['scaler'].transform(features_np)
                                        probs = model.named_steps['clf'].predict_proba(scaled)
                                    else:
                                        probs = model.named_steps['clf'].predict_proba(features_np)
                            except Exception:
                                probs = None

                    # Record prediction index and confidence (if available)
                    if prediction is not None:
                        try:
                            pred_idx = int(np.array(prediction).ravel()[0])
                        except Exception:
                            # if prediction is non-numeric label, try to map via label_encoder
                            try:
                                pred_idx = label_encoder.transform(np.array(prediction).ravel())[0]
                            except Exception:
                                pred_idx = None
                    else:
                        pred_idx = None

                    conf = None
                    if probs is not None:
                        try:
                            conf = float(np.max(probs))
                            # if pred_idx is None, derive from probs argmax
                            if pred_idx is None:
                                pred_idx = int(np.argmax(probs, axis=1)[0])
                        except Exception:
                            conf = None

                    per_hand_predictions.append((pred_idx, conf))

                # Choose one hand to display prediction for: prefer highest confidence; otherwise first
                chosen_label = None
                chosen_conf = None
                if len(per_hand_predictions) > 0:
                    # filter those with non-None pred_idx
                    valid = [(i, p, c) for i, (p, c) in enumerate(per_hand_predictions) if p is not None]
                    if any(c is not None for (_, p, c) in valid):
                        # choose by highest confidence
                        best = max(valid, key=lambda t: (t[2] if t[2] is not None else -1))
                        chosen_pred_idx, chosen_conf = best[1], best[2]
                    elif len(valid) > 0:
                        # no confidences, pick first valid
                        chosen_pred_idx, chosen_conf = valid[0][1], valid[0][2]
                    else:
                        # nothing numeric; skip displaying
                        chosen_pred_idx = None

                    if chosen_pred_idx is not None:
                        try:
                            chosen_label = label_encoder.inverse_transform([int(chosen_pred_idx)])[0]
                        except Exception:
                            # fallback: show raw index
                            chosen_label = str(chosen_pred_idx)

                # Display one label at top-center if we have one
                if chosen_label is not None:
                    h, w, _ = image.shape
                    display_text = f"Prediksi: {chosen_label}"
                    if chosen_conf is not None:
                        display_text += f" ({chosen_conf:.2f})"
                    # draw at top-center
                    text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    text_x = max(10, (w - text_size[0]) // 2)
                    text_y = 30
                    cv2.putText(image, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Gesture Alphabet Recognition", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy gesture recognition with saved pipeline and label encoder')
    # parser.add_argument('--model', type=str, default='modelAlphabet/new/gesture_model_alphabet.pkl')
    # parser.add_argument('--le', type=str, default='modelAlphabet/new/label_encoder_alphabet.pkl')
    parser.add_argument('--model', type=str, default='modelNumerik/new/gesture_model_numerik.pkl')
    parser.add_argument('--le', type=str, default='modelNumerik/new/label_encoder_numerik.pkl')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    args = parser.parse_args()

    main(args.model, args.le, args.camera)