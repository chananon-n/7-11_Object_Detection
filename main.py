from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolo11n.pt")

results = model.train(data="config.yaml", epochs=100,patience=20)

# Export the model to TFLite format
model.export(format="tflite")  # creates 'yolo11n_float32.tflite'