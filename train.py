"""
train.py - Training YOLOv8 untuk Aksara Kaganga (Ulu Rejang)
============================================================
Struktur folder proyek yang diharapkan:
    project/
    ├── datasets/
    │   ├── train/images/ & labels/
    │   ├── valid/images/ & labels/
    │   ├── test/images/  & labels/
    │   └── data.yaml
    ├── train.py         <- file ini
    └── runs/            <- hasil training (dibuat otomatis)

Instalasi dependencies:
    pip install ultralytics
"""

from ultralytics import YOLO
import os


# -----------------------------------------
# KONFIGURASI
# GPU: NVIDIA GeForce RTX 3050 Laptop (4GB VRAM)
# -----------------------------------------
DATA_YAML   = "datasets/data.yaml"   # path ke data.yaml
MODEL       = "yolov8n.pt"           # yolov8n (nano) = paling ringan
EPOCHS      = 100                    # jumlah epoch training
IMG_SIZE    = 640                    # dikurangi dari 640 -> hemat RAM & VRAM
BATCH_SIZE  = 4                      # dikurangi -> hemat RAM sistem
WORKERS     = 0                      # 0 = matikan multiprocessing (aman di Windows)
DEVICE      = "cuda"                 # GPU NVIDIA RTX 3050 Laptop
PROJECT_DIR = "runs/train"           # folder output hasil training
RUN_NAME    = "kaganga_v1"           # nama eksperimen
# -----------------------------------------


def main():
    # Validasi data.yaml
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"File '{DATA_YAML}' tidak ditemukan.\n"
            "Pastikan data.yaml ada di folder datasets/ dan path sudah benar."
        )

    print("=" * 55)
    print("  Training YOLOv8 - Aksara Kaganga (Ulu Rejang)")
    print("=" * 55)
    print(f"  Model     : {MODEL}")
    print(f"  Data      : {DATA_YAML}")
    print(f"  Epochs    : {EPOCHS}")
    print(f"  Img size  : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch     : {BATCH_SIZE}")
    print(f"  Workers   : {WORKERS}")
    print(f"  Device    : {DEVICE}")
    print(f"  Output    : {PROJECT_DIR}/{RUN_NAME}")
    print("=" * 55)

    # Load model (download otomatis jika belum ada)
    model = YOLO(MODEL)

    # Mulai training
    results = model.train(
        data        = DATA_YAML,
        epochs      = EPOCHS,
        imgsz       = IMG_SIZE,
        batch       = BATCH_SIZE,
        workers     = WORKERS,
        device      = DEVICE,
        project     = PROJECT_DIR,
        name        = RUN_NAME,
        cache       = False,         # jangan cache gambar ke RAM

        # Augmentasi - penting karena dataset kecil (~5 gambar/kelas)
        augment     = True,
        hsv_h       = 0.015,         # variasi warna hue
        hsv_s       = 0.7,           # variasi saturasi
        hsv_v       = 0.4,           # variasi kecerahan
        fliplr      = 0.0,           # JANGAN flip horizontal (aksara bisa terbalik)
        flipud      = 0.0,           # JANGAN flip vertikal
        degrees     = 5.0,           # rotasi kecil (aksara miring sedikit)
        translate   = 0.1,           # pergeseran posisi
        scale       = 0.5,           # zoom in/out
        mosaic      = 1.0,           # gabung 4 gambar (efektif untuk dataset kecil)

        # Optimasi
        optimizer   = "AdamW",
        lr0         = 0.001,         # learning rate awal
        patience    = 20,            # early stopping jika tidak ada peningkatan

        # Lainnya
        save        = True,
        save_period = 10,            # simpan checkpoint setiap 10 epoch
        plots       = True,          # buat grafik loss & metrik
        verbose     = True,
    )

    # Ambil path output yang sebenarnya dari hasil training
    save_dir = results.save_dir
    best_model_path = os.path.join(save_dir, "weights", "best.pt")

    print("\nTraining selesai!")
    print(f"  Folder output : {save_dir}")
    print(f"  Model terbaik : {best_model_path}")

    # Evaluasi otomatis pada validation set
    print("\nEvaluasi pada Validation Set...")
    model_best = YOLO(best_model_path)
    metrics = model_best.val(
        data    = DATA_YAML,
        split   = "val",
        imgsz   = IMG_SIZE,
        workers = WORKERS,
    )

    print("\nHasil Evaluasi:")
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")

if __name__ == "__main__":
    main()