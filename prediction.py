from ultralytics import YOLO

model = YOLO("yolov8n.pt")
#model.predict(source="vehicles/car_DOWN_0102.png", save=True, conf=0.5, save_txt=True)
model.export(format="onnx")