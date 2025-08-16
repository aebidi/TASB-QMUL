# SBDI: Side-Branch Detection in IVUS

## Project Structure

sb_baseline/
│
├── Frame_Dataset/ # (To be created by user) Preprocessed JPG frames and annotations.
│ ├── train/
│ ├── val/
│ └── test/
│
├── results/ # Directory for all outputs.
│ ├── best_model.pth # Saved weights of the best performing model.
│ ├── classification_report.txt # Final metrics on the test set.
│ ├── training_log_vX.log # Saved console output from training runs.
│ ├── visualizations_vX/ # Saved images with predicted bounding boxes.
│ └── plots/ # Saved comparison plots.
│
├── attention.py # Defines the custom TemporalFeatureMapAttention module.
├── augmentations.py # Defines the Albumentations pipelines for training and validation.
├── config.py # Central configuration for all hyperparameters and paths.
├── dataset.py # PyTorch Dataset class for loading temporal IVUS sequences.
├── evaluate.py # Script to evaluate the best model on the test set.
├── model.py # (Legacy) Contains earlier model versions (e.g., stacked attention).
├── plot_comparison.py # Script to plot validation learning curves from logs.
├── plot_test_scores.py # Script to plot the final test set bar chart.
├── temporal_attention_model.py # Defines the final, champion FPN Temporal Attention Faster R-CNN.
├── train.py # Main script for training the models.
├── utils.py # Utility functions (collate_fn, EarlyStopping, plotting).
└── visualise.py # Script to generate qualitative results (images with boxes)


---

## Setup and Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support
- Conda (recommended for environment management)

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd sb_baseline
```

### 2. Create and Activate Conda Environment
```bash
conda create -n sbdi_env python=3.10
conda activate sbdi_env
```

### 3. Install Dependencies
Install all required packages using the provided requirements.txt file.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pydicom albumentations matplotlib tqdm pandas numpy torchmetrics
```

### 4. Data Preparation
This project uses a private clinical dataset. The preprocessing pipeline expects the following structure:

DICOMS/: Directory containing raw DICOM files, organized into subfolders per vessel.

IVUS_SB_Labels.csv: CSV file containing annotations and train/val/test splits.

The preprocessing script converts this into the Frame_Dataset/ structure, required for training.

---

## Usage
All scripts are configured via config.py. Key parameters like BATCH_SIZE, EPOCHS, and TEMPORAL_FRAMES_K can be adjusted there.

### 1. Training

Use tmux or nohup for long runs on remote servers. Logs are saved in results/.

To train the final Temporal Attention model (without random augmentations):
```bash
python train.py --no-augment | tee results/training_log_final_model.log
```
To train with random augmentations:
```bash
python train.py | tee results/training_log_augmented.log
```

### 2. Evaluation

After training, a best_model.pth is saved. Run:
```bash
python evaluate.py
```
This prints results and saves a detailed classification_report.txt in results/.

### 3. Visualisation

Generate qualitative results:
```bash
python visualise.py
```
Annotated images will be saved in results/visualizations_V9/.

### 4. Plotting Comparison Graphs

Compare learning curves:
```bash
python plot_comparison.py
```

Compare final test set scores:
```bash
python plot_test_scores.py
```

---

## Results Summary

| Model Version | Key Changes                           | Test mAP@50 |
|---------------|---------------------------------------|-------------|
| V1            | Baseline Single-Frame                 | 0.7981      |
| V4            | Optimized Single-Frame (LR Scheduler) | 0.8184      |
| V7            | FPN Temporal Attention (k=1)          | **0.8365** (Champion) |
| V8            | FPN Temporal Attention (k=3)          | 0.8314      |


### Key Finding
A sophisticated temporal attention architecture (V7) that preserves FPN’s multi-scale features significantly outperforms optimized single-frame models. An optimal temporal window exists, with k=1 (3 frames) performing better than k=3 (7 frames).