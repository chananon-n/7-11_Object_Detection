import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")  # Update path if needed

# Load an image
image_path = "/Users/chananonkanunghet/Desktop/IMG_7007.jpeg"  # Replace with your image file
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Run object detection
results = model(image)

# Visualize the results
annotated_image = results[0].plot()

# Show the image
cv2.imshow("YOLOv8 Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
