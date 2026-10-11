
import cv2
from ultralytics import YOLO
from pathlib import Path

# Newly trained model (tracked in git so classmates get it on clone)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"

CONFIDENCE = 0.15
CAMERA_INDEX = 0

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")

model = YOLO(str(MODEL_PATH))
print("Loaded model:", MODEL_PATH.resolve())
print("Classes:", model.names)

# Find class IDs by their actual names
privacy_ids = {
    class_id
    for class_id, name in model.names.items()
    if any(word in name.lower() for word in ("blurr", "face", "plate", "privacy"))
}

sticker_ids = {
    class_id
    for class_id, name in model.names.items()
    if "sticker" in name.lower()
}

print("Privacy class IDs:", privacy_ids)
print("Sticker class IDs:", sticker_ids)

if not privacy_ids:
    raise RuntimeError("No privacy class found in the model.")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check CAMERA_INDEX.")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read frame from webcam.")
            break

        
   
        results = model.predict(
            source=frame,
            imgsz=416,
            conf=CONFIDENCE,
            verbose=False
        )



        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                # Clamp coordinates to the image boundaries
                height, width = frame.shape[:2]
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                class_name = model.names[class_id]

                if class_id in privacy_ids:
                    # Blur the actual pixels inside the box
                    roi = frame[y1:y2, x1:x2]
                    if roi.size:
                        # Strong pixelation-style blur
                        small_w = max(1, (x2 - x1) // 12)
                        small_h = max(1, (y2 - y1) // 12)
                        small = cv2.resize(
                            roi, (small_w, small_h),
                            interpolation=cv2.INTER_LINEAR
                        )
                        blurred = cv2.resize(
                            small, (x2 - x1, y2 - y1),
                            interpolation=cv2.INTER_NEAREST
                        )
                        frame[y1:y2, x1:x2] = blurred

                    # Label the blurred region
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (255, 0, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"Blurr {confidence:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2
                    )

                elif class_id in sticker_ids:
                    # Leave sticker image clear and draw a green box
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"{class_name} {confidence:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2
                    )

        cv2.imshow("PSU Sticker Detection - Privacy Blur", frame)

        # Press Q or Esc to close
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
