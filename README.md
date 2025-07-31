# YOLOv11 Next-Gen Object Detection using PyTorch

### _Achieve SOTA results with just 1 line of code!_
# 🚀 Demo

output.mp4


### ⚡ Installation (30 Seconds Setup)

```
conda create -n YOLO python=3.9
conda activate YOLO
pip install thop
pip install tqdm
pip install PyYAML
pip install opencv-python
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
```

### 🏋 Train

* Configure your dataset path in `main.py` for training
* Run `python main.py --train` for training

### 🧪 Test/Validate

* Configure your dataset path in `main.py` for testing
* Run `python main.py --Validate` for validation

### 🔍 Inference (Webcam or Video)

* Run `python main.py --inference` for inference

### 📊 Performance Metrics & Pretrained Checkpoints

| Model                                                                                | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
|--------------------------------------------------------------------------------------|----------------------|-------------------|--------------------|------------------------|
| [YOLOv11n](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_n.pt) | 39.5                 | 54.8              | **2.6**            | **6.5**                |
| [YOLOv11s](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_s.pt) | 47.0                 | 63.5              | 9.4                | 21.5                   |
| [YOLOv11m](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_m.pt) | 51.5                 | 68.1              | 20.1               | 68.0                   |
| [YOLOv11l](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_l.pt) | 53.4                 | 69.7              | 25.3               | 86.9                   | 50.7                 | 68.9              | 86.7               | 205.7                  |
| [YOLOv11x](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_x.pt) | 54.9                 | 71.3              | 56.9               | 194.9                  | 50.7                 | 68.9              | 86.7               | 205.7                  |

### 📈 Additional Metrics
### 📂 Dataset structure

    ├── COCO 
        ├── images
            ├── train2017
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
                ├── 2222.txt
            ├── val2017
                ├── 1111.txt
                ├── 2222.txt

⭐ Star the Repo!

If you find this project helpful, give us a star ⭐ 

#### 🔗 Reference

* https://github.com/ultralytics/ultralytics
* https://github.com/Shohruh72/YOLOv11
# YOLOv11
