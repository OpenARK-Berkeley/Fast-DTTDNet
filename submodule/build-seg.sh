python3 yolo-tensorrt/build.py \
    --weights yolov8s-seg.onnx \
    --fp16  \
    --device cuda:0 \
    --seg