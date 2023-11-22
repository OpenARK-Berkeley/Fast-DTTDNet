# YOLOv8-seg Model with TensorRT

### Deployment
If you've finetuned yolo for RGB-D masking, to deploy yolo segmentation model.
```
pip install onnx
pip install ultralytics
```
```
dpkg-query -W tensorrt # check your tensorrt version
pip install tensorrt --extra-index-url https://pypi.nvidia.com
``` 
```
bash export-seg.sh
bash build-seg.sh
```

### Finetune
If we construct the dataset in the format compatible with YOLOv8, it is easy to fine-tune the pre-trained YOLOv8 model.
```
from ultralytics import YOLO

yolo = YOLO('yolov8n.pt')
yolo.train(data='/content/data/data.yaml', epochs=5)
valid_results = yolo.val()
print(valid_results)
```

### Run
```
def run_yolo(yolo, image_url, conf=0.25, iou=0.7):
    results = yolo(image_url, conf=conf, iou=iou)
    res = results[0].plot()[:, :, [2,1,0]]
    return Image.fromarray(res)
    
yolo = YOLO('runs/detect/train/weights/best.pt')

image_url = 'test-01.jpg'
predict(image_url)  
```
