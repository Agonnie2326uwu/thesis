import cv2

for camera_index in range(5):
    print(f"Testing camera {camera_index}...")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if cap.isOpened():
        print(f"SUCCESS: Camera {camera_index} is available.")

        ret, frame = cap.read()

        if ret:
            print(f"SUCCESS: Camera {camera_index} returned a video frame.")
            cv2.imshow(f"Camera {camera_index}", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        else:
            print(f"Camera {camera_index} opened, but no frame was received.")

        cap.release()
    else:
        print(f"FAILED: Camera {camera_index} could not be opened.")