# Chromosome-Segmentation-PoolFormer

Welcome to the **Chromosome-Segmentation-PoolFormer** project! This repository contains a deep learning pipeline for automated chromosome analysis from microscopic images.

The pipeline is divided into three distinct stages to handle the complexity of chromosome segmentation, classification, and separation of overlapping or touching instances. 

## 🧬 Project Overview
The accurate segmentation and classification of chromosomes are critical for karyotyping and diagnosing genetic disorders. This codebase implements a state-of-the-art approach leveraging the feature extraction capabilities of **PoolFormer** (via OpenMMLab's `mmpretrain`) combined with robust segmentation and classification models.

### Three-Stage Pipeline
1. **Stage 1: Segmentation (`stage1_seg`)**
   - **Goal:** Segment chromosomes from the background.
   - **Architecture:** `PoolFormerUNet` – A custom U-Net-like architecture utilizing a PoolFormer `m36` backbone for the encoder and a custom decoder with Deep Supervision.
   - **Loss:** Combined Dice Loss and Focal Loss.
   
2. **Stage 2: Classification (`stage2_cls`)**
   - **Goal:** Classify chromosome patches into three categories: `single`, `overlapping`, and `touching`.
   - **Architecture:** Image Classifier built using `mmpretrain` with a PoolFormer `m36` backbone and a `LinearClsHead`.
   - **Features:** Supports advanced augmentations (Mixup, CutMix, Random Erasing) to handle class imbalance and dataset variations.
   
3. **Stage 3: Separation (`stage3_sep`)** (Work in Progress)
   - **Goal:** Separate chromosomes that are classified as overlapping or touching into individual instances.
   - **Architecture:** Generative Adversarial Networks (e.g., Pix2Pix or CycleGAN). *Currently pending full implementation.*

---

## 📂 Directory Structure

```text
.
├── configs/               # YAML configuration files for the 3 stages
│   ├── stage1_seg.yaml
│   ├── stage2_cls.yaml
│   └── stage3_sep.yaml
├── data/                  # Processed datasets for training
├── data_raw/              # Raw, unprocessed images and annotations
├── experiments/           # Auto-generated directory for weights, logs, and outputs
├── src/                   # Main source code directory
│   ├── stage1_seg/        # Scripts for Stage 1 (Segmentation)
│   ├── stage2_cls/        # Scripts for Stage 2 (Classification)
│   ├── stage3_sep/        # Scripts for Stage 3 (Separation)
│   └── utils.py           # Shared utilities (e.g., ExperimentLogger)
└── README.md              # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (Recommended for training)
- [PyTorch](https://pytorch.org/get-started/locally/) 

### Clone and Install
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kihurata/Chromosome-Segmentation-PoolFormer.git
   cd Chromosome-Segmentation-PoolFormer
   ```

2. **Install OpenMMLab dependencies:**
   The project heavily relies on OpenMMLab frameworks (`mmengine`, `mmpretrain`). Please install them along with standard machine learning libraries:
   ```bash
   pip install torch torchvision
   pip install mmengine
   pip install -U openmim
   mim install mmpretrain
   ```
   
3. **Install other dependencies:**
   ```bash
   pip install pyyaml pillow numpy scikit-learn
   ```

*(Note: If you encounter issues with `mmpretrain` installations, refer to the [official OpenMMLab documentation](https://mmpretrain.readthedocs.io/en/latest/get_started.html).)*

---

## 🚀 Usage

Training for each stage is managed by configuration files located in the `configs/` directory. Checkpoints, metrics, and logs are automatically saved under the `experiments/` directory.

### Stage 1: Train the Segmentation Model
You can start training the PoolFormerUNet for semantic segmentation:
```bash
python src/stage1_seg/train_seg.py --config configs/stage1_seg.yaml
```

### Stage 2: Train the Classification Model
To train the classifier to identify `single`, `overlapping`, and `touching` instances:
```bash
python src/stage2_cls/train_cluster.py --config configs/stage2_cls.yaml
```

### Stage 3: Train the Separation Model (WIP)
*The training script is currently a template. Once implemented, run:*
```bash
python src/stage3_sep/train_sep.py --config configs/stage3_sep.yaml
```

---

## 🔧 Configuration Details
Each stage uses a dedicated YAML file in `configs/` allowing you to easily adjust hyperparameters without altering the source code:
- **`data_root`**: Path to the respective dataset splits.
- **`batch_size`** & **`epochs`**: Standard training parameters.
- **`model`**: Architecture settings (e.g., `num_classes`, `pretrained` weights path, `input_size`).
- **`loss`** & **`augmentation`**: Controls functions like mixup, cutmix, rotatation angles, etc.

---

## 📊 Logging & Tracking
The training scripts use a custom `ExperimentLogger` (`src/utils.py`). For every run, an experiment folder is created in `experiments/` (e.g., `experiments/poolformer_m36_seg_<timestamp>/`). This folder contains:
- Copies of the training config and transform scripts for reproducibility.
- `metrics.csv`: Tabular logging of loss, IoU, F1 scores, and Learning Rate.
- Saved best and latest model checkpoints (`.pth`).

## ✨ Contributions
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
