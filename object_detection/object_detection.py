import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Read image
img = cv2.imread("image.jpg")
height, width, _ = img.shape

# Convert image to blob
blob = cv2.dnn.blobFromImage(
    img, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False
)
net.setInput(blob)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Forward pass
detections = net.forward(output_layers)

boxes = []
confidences = []
class_ids = []

# Process detections
for output in detections:
    for detect in output:
        scores = detect[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = int(detect[0] * width)
            center_y = int(detect[1] * height)
            w = int(detect[2] * width)
            h = int(detect[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Remove overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes
if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        if class_ids[i] < len(classes):
            label=classes[class_ids[i]]
        else:
            label="Unknown"
        label = classes[class_ids[i]]
        confidence = confidences[i]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{label} {confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# Show output
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()