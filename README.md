
#  Geo6D-Lite: Lightweight 6D Pose Estimation on LineMOD

A modular and lightweight implementation of **6D pose estimation** using the [LineMOD dataset](https://bop.felk.cvut.cz/datasets/).  
Built for research, student projects, and fast experimentation in Google Colab.

---

## Project Structure

Geo6D_Lite_LineMOD/
├── config.py # Central configuration (model, training, loss weights)
├── dataset.py # LineMOD dataset loader with cropping + intrinsics correction
├── geo6d_model.py # Main model wrapper (combines backbone + head)
├── main.py # Unified entry point (train / eval)
├── train.py # Minimal training version
├── train_resume.py # Resume training from checkpoint
├── evaluate.py # Standalone evaluation script (optional)
│
├── models/
│ ├── backbone.py # ResNet-based feature extractor
│ ├── pose_head.py # Rotation & translation prediction head
│ ├── losses.py # Loss functions (geodesic, L1)
│ └── rot6d.py # Rotation representation conversion (6D → matrix)
│
├── utils/
│ ├── checkpoint.py # Save & load model checkpoints
│ ├── image_utils.py # Bounding box + intrinsics utilities
│ └── metrics.py # Rotation / translation error metrics
│
└── data/ # (Optional) Dataset preprocessing or augmentations


---

##  Setup Instructions

### 1 Clone the Repository
```bash
git clone https://github.com/Govinda-Bhattarai/Geo6D_Lite_LineMOD.git
cd Geo6D_Lite_LineMOD

## Environment Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

##Install Dependencies

    pip install -r requirements.txt

##Dataset Setup

Download the LineMOD dataset (from BOP Benchmark) and extract it under:

    /content/drive/MyDrive/SharedCheckpoints/datasets/LineMOD/lm/

You can modify this path in config.py or pass it directly via --data_root.

Usage
 Train from Scratch
      python main.py --mode train

 Resume Training
      python main.py --mode train --resume

 Evaluate a Model
      python main.py --mode eval

  Outputs

After running training/evaluation, the following files are automatically saved in:

/content/drive/MyDrive/SharedCheckpoints/geo6d_checkpoints/


| File                          | Description                             |
| ----------------------------- | --------------------------------------- |
| `epoch_*.pth`                 | Model checkpoints per epoch             |
| `train_log.json`              | Average loss history                    |
| `eval_results.json`           | Evaluation metrics                      |
| `evaluation_histograms.png`   | Rotation & translation error histograms |
| `rotation_vs_translation.png` | Scatter plot comparing both errors      |


Model Overview

Backbone: ResNet18 (pretrained on ImageNet)

Head: Fully connected layers for 6D pose regression

Rotation Representation: Ortho6D → 3*3 rotation matrix

Loss: Combined Geodesic + L1 translation

Optimizer: AdamW + Cosine Annealing LR

Example Results

| Metric                 | Description                   |
| ---------------------- | ----------------------------- |
| Mean Rotation Error    | Average angular deviation (°) |
| Mean Translation Error | Average position error (cm)   |

Visualized automatically after evaluation:

    evaluation_histograms.png
    rotation_vs_translation.png

Directory Notes

| Folder                     | Purpose                                                  |
| -------------------------- | -------------------------------------------------------- |
| `models/`                  | Neural network components                                |
| `utils/`                   | Helper functions (checkpointing, metrics, visualization) |
| `data/`                    | (Optional) Dataset preparation or augmentation           |
| `Geo6D_Lite_LineMOD.ipynb` | Colab notebook version (optional)                        |


Checkpoint Example

  /content/drive/MyDrive/SharedCheckpoints/geo6d_checkpoints/geo6d_lite_latest.pth

You can modify this path in:

  main.py argument --checkpoint
  or in your Colab cell directly.


References

BOP Benchmark: LineMOD Dataset

Ortho6D Rotation Representation (Zhou et al., CVPR 2019)

PyTorch Documentation


