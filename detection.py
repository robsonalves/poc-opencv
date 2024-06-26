import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.saved_model.load("ssd_mobilenet_v2_fpnlite_320x320/saved_model")

# Load the labels
labels = {}
with open('coco_labels.txt', 'r') as f:
    for line in f:
        index, label = line.strip().split(':')
        labels[int(index)] = label

# Function to run object detection
def detect_objects(frame):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    return detections

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run detection
    detections = detect_objects(rgb_frame)

    # Visualization of the results of a detection
    for i in range(int(detections['num_detections'])):
        box = detections['detection_boxes'][i].numpy()
        class_id = int(detections['detection_classes'][i])
        score = detections['detection_scores'][i].numpy()
        if score > 0.5:
            y1, x1, y2, x2 = box
            h, w, _ = frame.shape
            x1, x2, y1, y2 = int(x1 * w), int(x2 * w), int(y1 * h), int(y2 * h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{labels[class_id]}: {score:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
