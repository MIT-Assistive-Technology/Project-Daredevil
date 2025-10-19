import cv2
from ultralytics import YOLO

# Load YOLO model (can use 'yolov8n.pt' for faster speed)
model = YOLO('yolov8n.pt')

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Frame size: width={frame_width}, height={frame_height}")
print(f"Min coordinates: (0, 0)")
print(f"Max coordinates: ({frame_width - 1}, {frame_height - 1})")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            # Print coordinates
            print(f"{label}: ({x1:.0f}, {y1:.0f}), ({x2:.0f}, {y2:.0f})  conf={conf:.2f}")

            # Draw on frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display video
    cv2.imshow("YOLO Live", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
