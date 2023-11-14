# YOLOv8-seg Model with TensorRT

### Deployment
If you've finetuned yolo for RGB-D masking, to deploy yolo segmentation model.
```
pip install onnx
pip install ultralytics
```
```
dpkg-query -W tensorrt # check your tensorrt version
python3 -m pip install --upgrade tensorrt
bash export-seg.sh
cd yolo-tensorrt
``` 

### Train