# AI600 — Assignment 3: Deep Learning with CNNs and Transfer Learning

## Repository Structure

```
AI600_PA3/
├── README.md
├── requirements.txt
├── .gitignore
│
├── Task1_CNN/
│   ├── mnist_cnn.ipynb              ← MNIST + Colored MNIST (single notebook)
│   ├── plots/
│   │   ├── mnist_loss.png
│   │   ├── mnist_accuracy.png
│   │   ├── mnist_filters.png
│   │   ├── cmnist_samples.png
│   │   ├── cmnist_loss.png
│   │   └── cmnist_accuracy.png
│   └── outputs/
│       ├── mnist_results.json
│       └── cmnist_results.json
│
├── Task2_TransferLearning/
│   ├── resnet18_stl10.ipynb         ← ResNet-18 + GradCAM (single notebook)
│   ├── plots/
│   │   ├── stl10_loss.png
│   │   ├── stl10_accuracy.png
│   │   ├── gradcam_correct_1.png
│   │   ├── gradcam_correct_2.png
│   │   ├── gradcam_wrong_1.png
│   │   └── gradcam_wrong_2.png
│   └── outputs/
│       └── stl10_results.json
│
└── report/
    ├── main.tex
    └── references.bib
```

## Key Results

| Experiment | Metric | Value |
|---|---|---|
| MNIST (custom CNN, 25,034 params) | Test Accuracy | **98.97%** |
| C-MNIST — Biased test | Test Accuracy | **99.36%** |
| C-MNIST — Unbiased test | Test Accuracy | **93.15%** |
| C-MNIST — Accuracy drop | | **6.21 pp** |
| STL-10 (frozen ResNet-18, 5,130 trainable params) | Test Accuracy | **94.73%** |

## Setup

```bash
pip install -r requirements.txt
```

All dependencies (PyTorch, torchvision, matplotlib, numpy, Pillow) are listed
in `requirements.txt`. The notebooks were developed and tested on **Google
Colab with a T4 GPU**.

## Running the Notebooks

Both notebooks are **self-contained** — all helper functions, model
definitions, and utilities are defined directly inside the notebook (no
external `.py` imports). Upload to Google Colab and run top-to-bottom.

### Task 1: `Task1_CNN/mnist_cnn.ipynb`

1. **Part A — MNIST**: trains a custom CNN (≤ 50k params), plots loss/accuracy
   curves, evaluates on test set, visualises first-layer filters.
2. **Part B — C-MNIST**: loads the LMS-provided `.pt` files, trains a
   3-channel variant from scratch, evaluates on biased and unbiased test sets.

**C-MNIST data**: place the three LMS files in `Task1_CNN/data/cmnist/`:
```
data/cmnist/train_biased.pt
data/cmnist/test_biased.pt
data/cmnist/test_unbiased.pt
```

### Task 2: `Task2_TransferLearning/resnet18_stl10.ipynb`

1. **Part A — ResNet-18 fine-tuning**: loads pretrained ResNet-18, freezes all
   conv layers, trains only the FC head (512→10) on STL-10.
2. **Part B — GradCAM**: generates heatmaps for 2 correct and 2 incorrect
   predictions using the final conv block (`layer4[-1]`).

STL-10 downloads automatically (~2.6 GB).

## Compiling the Report

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The report references plot images via relative paths (e.g.
`../Task1_CNN/plots/mnist_loss.png`), so compile from inside the `report/`
directory.
