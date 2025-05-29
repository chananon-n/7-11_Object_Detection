from ultralytics import YOLO

model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(model_path)

print(f"Exporting {model_path} to TFLite (FP32)...")
# Try exporting to FP32 first to see if it's an FP16 issue
model.export(format='tflite', imgsz=640) # Removed half=True
print("Export complete! Check for the .tflite file in the 'best_saved_model' directory.")