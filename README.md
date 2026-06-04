# Comvis_Kaganga_Kelompok_9

Model deteksi Aksara Kaganga / Ulu Rejang berbasis YOLOv8.

## Anggota Kelompok

1. Aditya Saputa (G1A023024)
2. Muhammad Yasser Ghifari Tegar Awally (G1A023030)
3. Migel Ray Sirait (G1A023088)

## Tentang Proyek

Proyek ini membangun pipeline computer vision untuk mendeteksi Aksara Kaganga pada gambar menggunakan YOLOv8. Hasil deteksi berupa bounding box, label kelas, dan confidence score untuk setiap aksara yang terdeteksi.

Repo ini berisi script untuk training, evaluasi, dan prediksi, beserta dataset yang sudah disiapkan dalam format YOLO.

## Teknologi

| Komponen | Keterangan |
| --- | --- |
| Bahasa | Python |
| Framework | Ultralytics YOLOv8 |
| Task | Object Detection |
| Format dataset | YOLO |
| Perangkat | GPU NVIDIA CUDA / CPU |

## Struktur Proyek

File dan folder utama yang benar-benar ada di proyek ini:

```text
project/
├── datasets/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── eval.py
├── predict.py
├── train.py
├── yolov8n.pt
├── yolov8s.pt
├── yolo26n.pt
└── runs/
    └── detect/
        ├── runs/
        │   ├── predict/
        │   │   └── hasil/
        │   └── train/
        │       └── kaganga_v1/
        └── val/
```

Catatan:

- `cek.py` tidak ada di repo ini.
- Fungsi untuk prediksi satu gambar sudah ada di `predict.py` lewat `predict_single()`.

## Dataset

Dataset disimpan di folder `datasets/` dan dikonfigurasi lewat `datasets/data.yaml`.

Ringkasan dataset lokal:

| Split | Jumlah gambar |
| --- | ---: |
| Train | 1403 |
| Validation | 166 |
| Test | 64 |

Dataset memiliki `253` kelas aksara/suku kata.

Contoh kelas: `a`, `ba`, `ka`, `nga`, `nyu`, `yang`, dan variasi suku kata lain yang didefinisikan di `datasets/data.yaml`.

Sumber dataset:

| Informasi | Nilai |
| --- | --- |
| Platform | Roboflow |
| Workspace | `novalrizkiansyah-ymail-com` |
| Project | `aksara-ulu-rejang` |
| Versi | 4 |
| Lisensi | CC BY 4.0 |
| URL | `https://universe.roboflow.com/novalrizkiansyah-ymail-com/aksara-ulu-rejang/dataset/4` |

Format dataset mengikuti struktur YOLO:

```text
datasets/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Label `.txt` menggunakan format berikut:

```text
class_id x_center y_center width height
```

## File Utama

| File | Fungsi |
| --- | --- |
| `train.py` | Melatih model YOLOv8 pada dataset Kaganga. |
| `eval.py` | Menjalankan evaluasi model pada validation set. |
| `predict.py` | Menjalankan prediksi pada gambar atau folder gambar. |
| `datasets/data.yaml` | Konfigurasi path dataset, jumlah kelas, dan nama kelas. |
| `yolov8n.pt` | Pretrained weights yang dipakai sebagai model awal training. |

## Konfigurasi Training

Konfigurasi utama training ada di `train.py`:

| Parameter | Nilai |
| --- | --- |
| `DATA_YAML` | `datasets/data.yaml` |
| `MODEL` | `yolov8n.pt` |
| `EPOCHS` | `100` |
| `IMG_SIZE` | `640` |
| `BATCH_SIZE` | `4` |
| `WORKERS` | `0` |
| `DEVICE` | `cuda` |
| `PROJECT_DIR` | `runs/train` |
| `RUN_NAME` | `kaganga_v1` |
| `optimizer` | `AdamW` |
| `save_period` | `10` |
| `patience` | `20` |

Augmentasi yang dipakai mencakup perubahan warna, rotasi kecil, translasi, scaling, dan mosaic. Flip horizontal dan vertical dimatikan agar bentuk aksara tidak berubah makna.

## Cara Menjalankan

### 1. Install dependency

```bash
pip install ultralytics
```

### 2. Training model

```bash
python train.py
```

Training akan menyimpan hasil ke:

```text
runs/detect/runs/train/kaganga_v1/
```

### 3. Evaluasi model

```bash
python eval.py
```

`eval.py` memakai model berikut:

```text
runs\detect\runs\train\kaganga_v1\weights\best.pt
```

### 4. Prediksi gambar

```bash
python predict.py
```

Konfigurasi utama prediksi di `predict.py`:

| Parameter | Nilai default |
| --- | --- |
| `MODEL_PATH` | `runs\detect\runs\train\kaganga_v1\weights\best.pt` |
| `SOURCE` | `datasets/test/images` |
| `CONF` | `0.25` |
| `IOU` | `0.45` |
| `IMG_SIZE` | `416` |
| `SAVE_DIR` | `runs/predict` |

## Output Prediksi

Hasil prediksi disimpan di:

```text
runs/predict/hasil/
```

Output yang dihasilkan meliputi:

- gambar dengan bounding box dan label
- file label `.txt`
- confidence score per deteksi
- ringkasan jumlah aksara terdeteksi per gambar
- urutan aksara dari kiri ke kanan

## Artefak Training

Folder training biasanya berisi artefak berikut:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `BoxPR_curve.png`
- `BoxF1_curve.png`
- `BoxP_curve.png`
- `BoxR_curve.png`
- `val_batch*_pred.jpg`
- `val_batch*_labels.jpg`

## Ringkasan Command

```bash
pip install ultralytics
python train.py
python eval.py
python predict.py
```

## Catatan

- README ini disesuaikan dengan file dan folder yang memang ada di repo.
- Jika kamu mengganti nama folder training atau model `best.pt`, sesuaikan juga path di `eval.py` dan `predict.py`.
