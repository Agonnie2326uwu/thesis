from ultralytics import YOLO
import cv2
from pathlib import Path

candidates = [
    Path(__file__).resolve().parent / "models" / "best.pt",
    Path(__file__).resolve().parent
    / "runs"
    / "detect"
    / "agonnie2326uwu"
    / "thesissticker"
    / "train"
    / "weights"
    / "best.pt",
]
weights = next((p for p in candidates if p.is_file()), candidates[0])
if not weights.is_file():
    raise FileNotFoundError(f"Trained weights not found: {weights}")

model = YOLO(str(weights))
cap = None
for camera_index in (0, 1):
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
        candidate = cv2.VideoCapture(camera_index, backend)
        if candidate.isOpened():
            cap = candidate
            print(f"Using camera {camera_index} with backend {backend}.")
            break
        candidate.release()
    if cap is not None:
        break

if cap is None:
    raise RuntimeError(
        "Could not open cameras 0 or 1 with Media Foundation or DirectShow. "
        "Check Windows camera permissions, confirm a camera is connected, and close "
        "other apps that may be using it."
    )

print("Webcam started.")
print("Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        results = model.predict(
            source=frame,
            conf=0.60,
            imgsz=640,
            verbose=False,
        )
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO26 Vehicle Sticker Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()