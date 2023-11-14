python3 yolo-tensorrt/infer-seg.py \
    --engine yolov8s-seg.engine \
    --imgs yolo-tensorrt/data \
    --show \
    --out-dir outputs \
    --device cuda:0