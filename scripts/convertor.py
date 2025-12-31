from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Export the model to ExecuTorch format
model.export(format="executorch")  # creates 'yolo11n_executorch_model' directory

executorch_model = YOLO("yolo11n_executorch_model")

results = executorch_model.predict("https://ultralytics.com/images/bus.jpg")