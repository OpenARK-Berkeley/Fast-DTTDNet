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

### Train