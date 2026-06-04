from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"runs\detect\runs\train\kaganga_v1-4\weights\best.pt")
    metrics = model.val(data="datasets/data.yaml", split="val", imgsz=416, workers=0)
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")