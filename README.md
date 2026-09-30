# 🧠 Modified CNN for Multi-Class Brain Tumor Classification

**An Efficient, Ultra-Lightweight Deep Learning Approach for MRI-Based Brain Tumor Diagnosis**

[![Python](https://img.shields.io/badge/Python-3.11.7-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-black.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)](#license)

> Master's Thesis Project — Department of Computer Science, College of Education for Pure Sciences, **Wasit University**
> By **Mahmoud Shamran Atheeb** | Supervised by **Asst. Prof. Dr. Baraa Ismeal Farhan**

---

## 📌 Overview

Accurate multi-class classification of brain tumors from Magnetic Resonance Imaging (MRI) is central to timely clinical decision-making — yet most high-performing deep learning models are far too large for deployment in resource-constrained clinical settings (mobile devices, edge hardware, low-resource hospitals).

This project introduces an **ultra-lightweight Modified CNN** that closes that gap: a compact architecture built around **three custom-designed modules**, trained and validated on the **Mendeley Brain Tumor MRI dataset**, and cross-validated on three independent external datasets (Kaggle, Br35H, Figshare) to confirm genuine generalization rather than dataset-specific overfitting.

The result is a model that is simultaneously **more accurate** and **~47× smaller** than comparable architectures such as U-Net, ResNet50, and VGG16 — making real-time, on-device brain tumor screening practical.

> 📦 **Repository size note:** The complete project — including source code, trained model checkpoints, experiment logs/figures, and the full MRI dataset used for training and evaluation — totals **~2.38 GB**. Please ensure sufficient disk space and a stable connection before cloning.

---

## ✨ Key Contributions

The proposed **Modified CNN** integrates three purpose-built modules that address distinct challenges in medical image analysis:

| Module | Full Name | What it does |
|---|---|---|
| **MAFE** | Micro Adaptive Feature Extractor | Reverses the sequential order of CBAM — applying **spatial attention before channel attention** — to better isolate tumor boundaries before channel-wise recalibration. |
| **MMSP** | Micro Multi-Scale Processor | Fuses **three parallel receptive-field pathways** (using depthwise separable convolutions) to capture tumor morphology at multiple resolutions within a single forward pass. |
| **CAIP** | Context-Aware Intelligent Pooling | Replaces static max/average pooling with a **sigmoid-gated, input-adaptive blend**, learned per sample from a global feature summary — instead of a fixed 50/50 ratio. |

These modules are woven into a compact **4-stage hierarchical CNN** (channel progression 32 → 64 → 128 → 256), followed by a lightweight classification head (Global Average Pooling → Dropout → Dense(64) → Dropout → Dense(4, Softmax)).

---

## 📊 Results at a Glance

### Primary Performance (Mendeley Dataset)

| Metric | Value |
|---|---|
| Best Validation Accuracy | **99.22%** |
| Test Accuracy | **97.35%** |
| Reproducibility (3 runs) | 99.12% ± 0.22% (range 98.81%–99.27%) |
| Total Parameters | **659,228** (~0.66M) |
| Parameter-Efficiency Score | **150.51** (Accuracy / Params in M) |
| Model File Size | **7.88 MB** |
| Memory Footprint | **1.8 MB** |
| Inference Latency | **12 ms / image** |

### Cross-Dataset Generalization (Zero-Shot / Fine-Tuned)

| External Dataset | Result |
|---|---|
| Kaggle Brain Tumor MRI | **98.38%** (zero-shot) |
| Br35H Brain Tumor Detection | 90.60% (zero-shot) → **97.62%** (fine-tuned) |
| Figshare Brain Tumor | 89.36% (initial) → **99.35%** (after orientation-artifact correction) |

### Comparison with Recent State-of-the-Art (2024–2025)

| Model | Parameters | Accuracy | Efficiency Score |
|---|---|---|---|
| **Modified CNN (this work)** | **0.66M** | **99.22%** | **150.51** |
| Khan & Auvee (2024) — Resource-efficient CNN | 1.1M | 98.09% | 89.17 |
| Zahoor et al. (2024) — Res-BRNet | 8.2M | 98.22% | — |
| Khan et al. (2025) — Hybrid CE-EEN-B0-ResGANet | ~15M | 99.11% | — |
| Ilani et al. (2025) — U-Net hybrid | ~31M | 98.56% | 3.18 |

Compared to U-Net specifically, the proposed model is **~15.5× smaller in file size**, **~49× lighter in RAM usage**, and **47× fewer parameters**, while achieving higher accuracy.

---

## 🧬 Dataset

| Property | Detail |
|---|---|
| Primary Dataset | Mendeley Brain Tumor MRI Dataset (v4) |
| Total Images | 12,064 |
| Classes | Glioma, Meningioma, Pituitary, No Tumor |
| Split | 7,720 train / 1,930 validation / 2,414 test |
| External Validation Sets | Kaggle Brain Tumor MRI, Br35H Brain Tumor Detection, Figshare Brain Tumor Dataset |
| Total Project + Dataset Size | **~2.38 GB** |

**Preprocessing & Augmentation pipeline:**
- Resize to 128×128
- Random horizontal flip (p=0.40), vertical flip (p=0.30)
- Random rotation (±20°), affine translation (±10%)
- Color jitter (brightness/contrast 0.30, saturation 0.20)
- Gaussian blur (3×3, σ ∈ [0.1, 2.0])
- Channel-wise normalization (ImageNet mean/std)
- MD5-based duplicate detection and integrity verification

---

## 🏗️ Model Architecture

```
Input (128×128×3)
    │
Stage 1 (32ch)  → MAFE → CAIP
    │
Stage 2 (64ch)  → MMSP → MAFE → CAIP
    │
Stage 3 (128ch) → MMSP → MAFE → CAIP
    │
Stage 4 (256ch) → MMSP
    │
Global Average Pooling → Flatten
    │
Dropout(0.3) → Dense(64, ReLU)
    │
Dropout(0.2) → Dense(4, Softmax)
```

**Training configuration:**
- Optimizer: Adam (lr=1e-3, β₁=0.9, β₂=0.999, ε=1e-8)
- Scheduler: Cosine Annealing with Warm Restarts
- Loss: Multi-class Cross-Entropy
- Batch Normalization after every convolution
- Batch size: 96 | Epochs: 300 | Fixed seed: 42 (full reproducibility)
- Training time: ~4h 2min on a single GPU (14,509.9 seconds)

---

## 🖥️ Hardware & Software Environment

| Component | Specification |
|---|---|
| CPU | Intel Core Ultra 9 |
| RAM | 64 GB DDR5 |
| GPU | NVIDIA GeForce RTX 5080 Laptop (15.9 GB VRAM) |
| Framework | PyTorch 2.1.0 |
| CUDA | 12.1 |
| Python | 3.11.7 |

---

## 🌐 Web Application (Demo Interface)

A lightweight local web app was built with **Flask** to demonstrate real-world usage:

- Upload an MRI scan (PNG / JPG / WEBP) via drag-and-drop
- Browse curated sample images by tumor class
- Run real-time inference with the trained model
- Visualize the model's decision-making via **Grad-CAM** heatmaps

Open `http://localhost:5000` in your browser after running the app (see [Installation & Usage](#️-installation--usage)).

---

## 📁 Project Structure

> Run `explore_project.py` (included in this repo) from the project root to
> auto-generate the real folder tree and confirm the ~2.38 GB total size,
> then paste the output here to replace this placeholder.

```bash
python explore_project.py --path . --max-depth 3
```

```
<paste the real tree generated by explore_project.py here>
```

---

## ⚙️ Installation & Usage

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-name>

# 2. Install dependencies
pip install torch torchvision pillow numpy matplotlib seaborn scikit-learn tqdm

# 3. Run the demo web app
python app2.py
```

---

## 🔬 Explainability

Model predictions are interpreted using **Grad-CAM** attention maps combined with Otsu thresholding to localize diagnostically relevant regions, along with focus-ratio and attention-entropy metrics to quantify spatial confidence — supporting clinical trust and transparency.

---

## ⚠️ Limitations & Future Work

- Train/validation split is performed at the **image level** (not patient level), as the Mendeley dataset lacks patient identifiers — a known limitation discussed in detail in the thesis.
- Ablation studies were conducted under a matched 50-epoch budget rather than the full 300-epoch regime; a fully matched-budget ablation is proposed as future work.
- Further clinical validation (e.g., head-to-head comparison with radiologists) is required before real-world deployment.
- Extending CAIP to a spatially-varying (per-region) gating formulation is identified as a promising future direction.

---

## 📚 Citation

If you use this work, please cite the thesis:

```
Atheeb, M. S., & Farhan, B. I. (2026). Lightweight Deep Learning Models for
Brain Tumor Classification. International Conference on Applied Innovations
in IT (ICAIIT), Hochschule Anhalt. https://doi.org/10.25673/124569

Shamran, M., & Ismeal Farhan, B. (2026). Optimized Lightweight Convolutional
Neural Network Architecture for Multi-Class Brain Tumor Classification from
Magnetic Resonance Imaging. Wasit Journal for Pure Sciences, 5(3), 57–69.
https://doi.org/10.31185/wjps.1001
```

---

## 🙏 Acknowledgments

- Wasit University — College of Education for Pure Sciences, Department of Computer Science
- Mendeley Brain Tumor MRI Dataset, Kaggle Brain Tumor MRI Dataset, Br35H Brain Tumor Detection Dataset, and Figshare Brain Tumor Dataset contributors

---

## 📄 License

This project is released for **academic and research purposes**. Please contact the author for any commercial use inquiries.
