python3 yolo-tensorrt/export-seg.py \
    --weights yolov8s-seg.pt \
    --opset 11 \
    --sim \
    --input-shape 1 3 640 640 \
    --device cuda:0