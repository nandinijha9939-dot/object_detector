import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO('yolov8n.pt')  # You can also use 'yolov8s.pt', 'yolov8m.pt', etc.

# Open webcam (0 for default camera)
cap = cv2.VideoCapture(0)

# Shrink factor (0.8 = 80% of original size)
shrink_factor = 0.7  # Adjust this value (0.5 to 0.9 works well)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run object detection
    results = model(frame, conf=0.6)  # confidence threshold 0.5
    
    # Draw results on frame
    annotated_frame = frame.copy()
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # 🔥 SHRINK THE BOX - Only for persons
                if class_name == 'person':
                    # Calculate center and size
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Shrink the box
                    new_width = int(width * shrink_factor)
                    new_height = int(height * shrink_factor)
                    
                    # Calculate new coordinates (centered)
                    x1 = center_x - new_width // 2
                    y1 = center_y - new_height // 2
                    x2 = center_x + new_width // 2
                    y2 = center_y + new_height // 2

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw label on top of the box
                label = f"{class_name}"
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw background for text
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_height -10),
                    (x1 + text_width , y1),
                    (0, 255, 0),
                    -1
                )
                
                # Put text on top of the box
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )
    
    # Display the frame
    cv2.imshow('Object Detection', annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()