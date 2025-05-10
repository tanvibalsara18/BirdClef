# BirdCLEF 2024 - Bird Sound Recognition

## 🐦 About BirdCLEF

BirdCLEF is a part of the LifeCLEF challenge under the CLEF (Conference and Labs of the Evaluation Forum) initiative. This edition focused on identifying bird calls from soundscapes—a more realistic and noisy environment compared to past years. The goal was to build models that can accurately detect bird species from long field recordings containing overlapping sounds and varied background noise.

## 🛠️ Solution Overview

This notebook implements a deep learning pipeline using the **EfficientNet** model for bird audio classification. The approach includes:

- Preprocessing and augmenting audio using `librosa`, `albumentations`, and spectrogram transformations.
- Utilizing PyTorch Lightning for clean training loops and experiment management.
- Applying **mixup** and **Cutout** regularization techniques to improve model robustness.
- Using advanced learning rate schedulers such as `CosineAnnealingLR`, `ReduceLROnPlateau`, and `OneCycleLR`.

## 🧰 Libraries and Tools

The following libraries were used:

- `torch`, `torchvision`, `pytorch_lightning`
- `timm` for pretrained EfficientNet models
- `librosa` for audio processing
- `albumentations` for spectrogram augmentation
- `scikit-learn` for evaluation metrics
- `PIL`, `matplotlib`, `plotly` for visualization
- `wandb` for logging (via `WandbLogger`)

## 📁 Notebook Contents

- **Data Loading & Exploration**: Reads and visualizes metadata and audio waveforms.
- **Audio to Image Conversion**: Converts audio snippets into mel spectrograms.
- **Data Augmentation**: Applies time/frequency masking, mixup, and cutout on spectrogram images.
- **Model Definition**: Uses EfficientNet variants via `timm` with custom classification heads.
- **Training & Validation**: PyTorch Lightning module with training metrics, loss functions, and early stopping.
- **Inference**: Generates predictions and prepares submissions for the Kaggle leaderboard.

## 📈 Model Performance

Evaluation is based on log loss and ranked accuracy on multi-label audio samples. This notebook includes validation performance tracking and can be adapted for ensemble techniques or test-time augmentation.

## 🙏 Acknowledgements

- BirdCLEF & LifeCLEF organizers for the open biodiversity data.
- Kaggle community for baseline sharing and collaborative research.