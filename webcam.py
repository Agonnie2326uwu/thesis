from ultralytics import YOLO
import cv2
from pathlib import Path

# Portable weights path so classmates can clone and run.
# Newly trained weights live in models/best.pt (copied from the latest run).
weights = Path(__file__).resolve().parent / "models" / "best.pt"
if not weights.is_file():
    raise FileNotFoundError(f"Trained weights not found: {weights}")

# Load the NEW trained model
model = YOLO(str(weights))

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

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
            conf=0.40,
            imgsz=640,
            verbose=False
        )

        # Process detections
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_name = model.names[class_id]

            # Blur faces
            if class_name == "BlurrFace":
                face = frame[y1:y2, x1:x2]

                if face.size > 0:
                    blurred_face = cv2.GaussianBlur(
                        face,
                        (51, 51),
                        0
                    )

                    frame[y1:y2, x1:x2] = blurred_face

            # Draw vehicle sticker detection
            elif class_name == "PSU vehicle sticker":
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{class_name} {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.imshow(
            "YOLO26 Vehicle Sticker + Face Blur",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()