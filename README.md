# YOLOv11 Next-Gen Object Detection using PyTorch

### _Achieve SOTA results with just 1 line of code!_
# 🚀 Demo

[demo video](output.mp4)


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

### 📋 Usage Examples

#### 1. Basic Training
```bash
# Train from scratch
python main.py --train --epochs 100 --batch-size 32

# Train with custom dataset path
python main.py --train --data-dir /path/to/your/dataset --epochs 50
```

#### 2. Resume Training
```bash
# Resume from last checkpoint
python main.py --train --resume weights/last.pt

# Resume from best checkpoint
python main.py --train --resume weights/best.pt
```

#### 3. Transfer Learning
```bash
# Initialize with Ultralytics pretrained weights
python main.py --train --weights yolov11n.pt --epochs 50
```

#### 4. Validation
```bash
# Validate model performance
python main.py --validate

# Validate with custom settings
python main.py --validate --batch-size 16 --inp-size 512
```

#### 5. Evaluation
```bash
# Full evaluation (accuracy and speed)
python main.py --evaluate

# Accuracy-only evaluation
python main.py --evaluate --eval-mode accuracy

# Speed-only evaluation
python main.py --evaluate --eval-mode speed

# Quick evaluation for development
python main.py --evaluate --eval-mode quick
```

#### 6. Inference
```bash
# Run inference on input.mp4
python main.py --inference

# Run inference with custom input size
python main.py --inference --inp-size 512
```

### 🏋 Train

* Configure your dataset path in `main.py` for training
* Run `python main.py --train` for training
* Run `python main.py --train --resume weights/last.pt` to resume training from a checkpoint
* Run `python main.py --train --weights path/to/pretrained.pt` to initialize with pretrained weights

#### Training Command Line Arguments

* `--epochs N` - Number of training epochs (default: 2)
* `--batch-size N` - Batch size for training (default: 32)
* `--inp-size N` - Input image size (default: 640)
* `--num-cls N` - Number of object classes (default: 80)
* `--data-dir PATH` - Path to dataset directory (default: 'COCO')
* `--weights PATH` - Path to pretrained weights file
* `--resume PATH` - Path to checkpoint file for resuming training

### 🧪 Test/Validate

* Configure your dataset path in `main.py` for testing
* Run `python main.py --validate` for validation

#### Validation Command Line Arguments

* `--batch-size N` - Batch size for validation (default: 32)
* `--inp-size N` - Input image size (default: 640)
* `--num-cls N` - Number of object classes (default: 80)
* `--data-dir PATH` - Path to dataset directory (default: 'COCO')

### 📊 Evaluation (Accuracy and Speed)

* Run `python main.py --evaluate` for full evaluation (accuracy and speed)
* Run `python main.py --evaluate --eval-mode accuracy` for accuracy-only evaluation
* Run `python main.py --evaluate --eval-mode speed` for speed-only evaluation
* Run `python main.py --evaluate --eval-mode quick` for quick evaluation (reduced dataset)
* Run `python eval.py` directly for evaluation with more options

#### Evaluation Command Line Arguments

* `--eval-mode MODE` - Evaluation mode: full, accuracy, speed, or quick (default: full)
* `--batch-size N` - Batch size for evaluation (default: 32)
* `--inp-size N` - Input image size (default: 640)
* `--num-cls N` - Number of object classes (default: 80)
* `--data-dir PATH` - Path to dataset directory (default: 'COCO')

#### Evaluation Metrics
* **mAP@0.5:0.95** - Mean Average Precision across IoU thresholds from 0.5 to 0.95
* **mAP@0.5** - Mean Average Precision at IoU threshold 0.5
* **Precision** - Model's ability to identify only relevant objects
* **Recall** - Model's ability to find all relevant objects
* **FPS** - Frames Per Second for inference speed

### 🚀 Key Features

* **Breakpoint Training (断点续训)** - Resume training from checkpoints to prevent losing progress
* **Error Tolerance (错误容忍)** - Continue training even when encountering problematic batches
* **Comprehensive Evaluation (评估)** - Detailed accuracy and speed metrics with visualization
* **Ultralytics Weight Support** - Load pretrained weights from Ultralytics format
* **Advanced Data Augmentation** - Mosaic, MixUp, HSV, and geometric transformations
* **Model Exponential Moving Average (EMA)** - Improved model stability and performance
* **Distributed Training Support** - Train on multiple GPUs for faster convergence
* **Gradient Clipping** - Prevent gradient explosion during training
* **Mixed Precision Training** - Reduce memory usage and accelerate training

### 🛠 Advanced Training Configuration

Training can be customized through `utils/args.yaml`:

* **Checkpoint Configuration**:
  * `checkpoint_freq` - Save checkpoint every N epochs
  * `save_best` - Save best model based on mAP
  * `save_last` - Save last model
* **Error Tolerance Configuration**:
  * `enable_error_tolerance` - Enable batch error tolerance mechanism
  * `max_consecutive_errors` - Maximum consecutive errors before stopping
  * `error_log_file` - File to log error batches

### 📊 Performance Metrics & Pretrained Checkpoints

| Model                                                                                | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
|--------------------------------------------------------------------------------------|----------------------|-------------------|--------------------|------------------------|
| [YOLOv11n](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_n.pt) | 39.5                 | 54.8              | **2.6**            | **6.5**                |
| [YOLOv11s](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_s.pt) | 47.0                 | 63.5              | 9.4                | 21.5                   |
| [YOLOv11m](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_m.pt) | 51.5                 | 68.1              | 20.1               | 68.0                   |
| [YOLOv11l](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_l.pt) | 53.4                 | 69.7              | 25.3               | 86.9                   |
| [YOLOv11x](https://github.com/Shohruh72/YOLOv11/releases/download/v1.0.0/v11_x.pt) | 54.9                 | 71.3              | 56.9               | 194.9                  |

### 🔍 Inference (Webcam or Video)

* Run `python main.py --inference` for inference
* Input video should be named `input.mp4`
* Output will be saved as `output.mp4` or `output.avi` if H264 codec fails
* Supports multiple video codecs for compatibility

#### Inference Command Line Arguments

* `--inp-size N` - Input image size (default: 640)
* `--num-cls N` - Number of object classes (default: 80)

### 📈 Additional Metrics

### 📂 Dataset structure

### 🚨 Troubleshooting & FAQ

#### Common Issues

**Q: Training stops due to batch errors**
A: The error tolerance mechanism is enabled by default. Check `weights/error_batches.log` for details on problematic batches. You can adjust `max_consecutive_errors` in `utils/args.yaml` to control when training should stop.

**Q: CUDA out of memory error**
A: Reduce batch size with `--batch-size` argument or enable mixed precision training (enabled by default on CUDA devices).

**Q: Video inference produces no output**
A: Ensure `input.mp4` exists in the project directory. Check that OpenCV is properly installed and supports the video codec.

**Q: Evaluation metrics are low**
A: Ensure your dataset is properly formatted and the model is trained for sufficient epochs. Check `utils/args.yaml` for data augmentation settings.

#### Performance Tips

* Use `--eval-mode quick` for faster evaluation during development
* Enable distributed training with multiple GPUs for faster training
* Adjust learning rate and batch size based on your GPU memory
* Use `--epochs` to control training duration
* Monitor `weights/step.csv` for training progress and metrics

#### Checkpoint Management

* `weights/last.pt` - Most recent model checkpoint
* `weights/best.pt` - Best performing model based on mAP
* `weights/epoch_N.pt` - Periodic checkpoints (if enabled)
* Check `weights/step.csv` for training metrics over time

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
