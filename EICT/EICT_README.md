# Natural Scene Image Classification: Custom CNN vs. Transfer Learning (ResNet-18)

Project submitted for the **Summer Training Programme on Data Science, Machine Learning & Agentic AI**, organised by the **Electronics & ICT Academy (E&ICT), IIT Roorkee**. The project compares a convolutional neural network built from scratch with a pre-trained ResNet-18 on a six-class scene classification task. The report is dated August 11, 2026.

> **Important:** the performance figures in this folder are not produced by the code. See [Status of the Results](#status-of-the-results) before citing any numbers.

---

## Contents

| File | Description |
|------|-------------|
| [`Parul_Report (1).pdf`](./Parul_Report%20%281%29.pdf) | 17-page project report: programme overview, problem statement, dataset, methodology, results, challenges, conclusion, references |
| [`Project_AgenticAI_EICT.ipynb`](./Project_AgenticAI_EICT.ipynb) | PyTorch notebook (11 cells) with setup, augmentations, both model definitions, and printed training and evaluation output |

The notebook is named "AgenticAI" after the programme, but the project itself is image classification. It contains no agentic or LLM components.

---

## Problem Statement

Scene classification must capture global context, background texture and lighting, unlike object-centric classification. Training a deep network from scratch on a medium-sized dataset risks overfitting, while transfer learning from ImageNet features may converge faster and generalise better. The project compares the two approaches on the same data and settings.

---

## Dataset

**Intel Image Classification** from Kaggle (`puneet6008/intel-image-classification`), 150 × 150 RGB images in six classes: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

| Split | Images |
|-------|--------|
| Train (`seg_train`) | 14,034, split in the notebook into 11,929 train and 2,105 validation |
| Test (`seg_test`) | 3,000 |

Classes are roughly balanced (about 16–18% each in the train set).

The dataset is not included in this folder. The notebook expects it at `./data/intel_image_classification/` with `seg_train/seg_train` and `seg_test/seg_test` subfolders.

---

## Methodology

**Preprocessing and augmentation (training only):** resize to 150 × 150, random horizontal flip (p = 0.5), random rotation (±15°), colour jitter (brightness, contrast and saturation 0.2), ImageNet mean/std normalisation. Evaluation uses resize and normalisation only.

**Model 1: Custom CNN.** Four convolution blocks (3→32→64→128→256 channels, 3 × 3 kernels, BatchNorm and ReLU, with max-pooling after blocks 1–3 and adaptive average pooling to 4 × 4 after block 4), then Flatten → Dropout(0.4) → Linear(4096→256) → ReLU → Dropout(0.3) → Linear(256→6).

**Model 2: ResNet-18 (transfer learning).** ImageNet-pre-trained backbone with all weights frozen, and the final layer replaced by Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→6). Only this head is trained (132,870 parameters).

**Training configuration (as described in the report):** cross-entropy loss, AdamW (learning rate 1e-3, weight decay 1e-4), ReduceLROnPlateau (patience 2, factor 0.5), batch size 64, 10 epochs with best-validation checkpointing, seed 42, Google Colab with a Tesla T4 GPU, PyTorch 2.1.0.

---

## Reported Results

These are the figures stated in the report and printed by the notebook. They are **not reproducible from the code in this folder** (see next section).

| Model | Trainable parameters | Training time | Best val. accuracy | Test accuracy | Test loss |
|-------|---------------------|---------------|--------------------|---------------|-----------|
| Custom CNN | 1,438,822 | 4 min 12 s | 83.90% | 82.60% | 0.4810 |
| ResNet-18 (frozen backbone) | 132,870 | 2 min 45 s | 93.40% | 93.10% | 0.2190 |

ResNet-18 per-class test results as reported:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| buildings | 0.92 | 0.90 | 0.91 | 437 |
| forest | 0.98 | 0.99 | 0.98 | 474 |
| glacier | 0.89 | 0.88 | 0.88 | 553 |
| mountain | 0.89 | 0.91 | 0.90 | 525 |
| sea | 0.95 | 0.96 | 0.95 | 510 |
| street | 0.94 | 0.93 | 0.94 | 501 |

The report highlights glacier ↔ mountain and buildings ↔ street as the main confusions.

---

## Status of the Results

The notebook defines the data transforms and both architectures, but **does not train or evaluate anything**:

- It has no dataset loading (`ImageFolder` / `DataLoader`), loss function, optimiser, training loop or evaluation code.
- The epoch-by-epoch "training logs" are lists of strings written directly into the code. The cell is commented *"Simulated execution log"*.
- The classification report is likewise a hard-coded text block, and the notebook has no confusion matrix computation. The confusion matrix in the report is typed text, not a generated plot.
- The notebook reports 1,438,822 trainable parameters for the custom CNN, but by my calculation the layers as defined total 1,439,750. The ResNet-18 head count (132,870) does match.
- The custom CNN's test accuracy and test loss appear only in the report, not in the notebook.

To make this project reproducible, the notebook needs real data loading, a training loop for both models, evaluation on the test set, and a generated confusion matrix, after which the numbers above should be replaced with the measured ones.

---

## Running the Notebook

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow
jupyter notebook Project_AgenticAI_EICT.ipynb
```

As it stands, running it defines the models and prints the stored text. A CUDA GPU is optional; the notebook falls back to MPS or CPU.

---

## Future Scope (from the report)

- Grad-CAM heatmaps to show which image regions drive predictions
- Export to ONNX and a Gradio or Streamlit demo
- Automated hyperparameter search with Ray Tune or Optuna

---

## References

Key sources cited in the report include He et al. (2016) *Deep Residual Learning for Image Recognition*, Paszke et al. (2019) *PyTorch*, Russakovsky et al. (2015) *ImageNet Large Scale Visual Recognition Challenge*, Selvaraju et al. (2017) *Grad-CAM*, and the Intel Image Classification dataset on Kaggle.

---

## Author

**Parul**, Department of Computer Science & Engineering
Summer Training Programme, E&ICT Academy, IIT Roorkee
