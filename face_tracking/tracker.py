import cv2
import mediapipe as mp
import numpy as np
import time

import urllib.request
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe provides a high-level solution called "FaceLandmarker"
# We use the Tasks API here because the older solutions API is not
# fully supported on Python 3.12+ on Windows.

def build_face_tracker():
    """
    Returns a configured MediaPipe FaceLandmarker object.
    Automatically downloads the model file if it is not present.
    """
    model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    if not os.path.exists(model_path):
        print("Downloading Face Landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector


def extract_landmarks(detector, frame_bgr):
    """
    Takes a single BGR frame (from cv2) and returns:
        - landmarks_norm: list of (x, y, z) in normalized [0,1] space
        - landmarks_px: list of (x, y) in pixel coordinates
        - head_pose: dict with estimated rotation angles

    Returns (None, None, None) if no face is detected.
    """
    h, w = frame_bgr.shape[:2]
    
    # MediaPipe Tasks API expects an mp.Image
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    results = detector.detect(mp_image)

    if not results.face_landmarks:
        return None, None, None

    # Take the first detected face
    face_landmarks = results.face_landmarks[0]

    # Extract normalized (x, y, z)
    landmarks_norm = [
        (lm.x, lm.y, lm.z) for lm in face_landmarks
    ]

    # Convert normalized to pixel coordinates for visualization
    landmarks_px = [
        (int(lm.x * w), int(lm.y * h)) for lm in face_landmarks
    ]

    # Estimate head pose using key landmarks
    head_pose = estimate_head_pose(landmarks_norm)

    return landmarks_norm, landmarks_px, head_pose


def estimate_head_pose(landmarks_norm):
    """
    Estimates yaw (left-right), pitch (up-down), and roll (tilt)
    using 6 key landmark indices known in MediaPipe Face Mesh.

    Index reference (MediaPipe 468-point model):
    1   = nose tip
    33  = left eye outer corner
    263 = right eye outer corner
    61  = left mouth corner
    291 = right mouth corner
    199 = chin

    We compare left vs right eye positions to estimate yaw,
    and nose vs chin positions to estimate pitch.
    This is a geometry approximation, not a full PnP solve.
    It is accurate enough for communication scoring purposes.
    """
    if len(landmarks_norm) < 468:
        return {"yaw": 0, "pitch": 0, "roll": 0}

    nose = landmarks_norm[1]
    left_eye = landmarks_norm[33]
    right_eye = landmarks_norm[263]
    chin = landmarks_norm[199]

    # Yaw: if right eye x is much greater than left eye x → facing forward
    # If difference shrinks → head is turning
    eye_dx = right_eye[0] - left_eye[0]
    yaw = (0.5 - nose[0]) * 100  # positive = turned right

    # Pitch: nose tip y relative to eye midpoint y
    eye_mid_y = (left_eye[1] + right_eye[1]) / 2
    pitch = (nose[1] - eye_mid_y) * 100  # positive = looking down

    # Roll: angle of the eye line relative to horizontal
    roll = np.degrees(np.arctan2(
        right_eye[1] - left_eye[1],
        right_eye[0] - left_eye[0]
    ))

    return {"yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2)}


def run_tracker_demo():
    """
    Opens webcam and shows live landmark overlay.
    Press 'q' to quit.

    This is purely for verifying that tracking works.
    It does NOT do emotion recognition yet.
    """
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    face_mesh = build_face_tracker()

    print("Face tracker running. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks_norm, landmarks_px, head_pose = extract_landmarks(face_mesh, frame)

        if landmarks_px:
            # Draw all 468 landmark dots on the frame
            for (x, y) in landmarks_px:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Display head pose estimates on screen
            cv2.putText(frame, f"Yaw: {head_pose['yaw']:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, f"Pitch: {head_pose['pitch']:.1f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, f"Roll: {head_pose['roll']:.1f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        cv2.imshow("Face Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_tracker_demo()