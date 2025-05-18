from config import config
from ultralytics import YOLO

# model = YOLO('models/best.pt')
model = YOLO(config['paths']['model_path'])

# results = model.predict('input_videos/08fd33_4.mp4', save=True)
results = model.predict(config['paths']['video_input'], save=True)
print(results[0])
print('==========================')
for box in results[0].boxes:
    print(box)
