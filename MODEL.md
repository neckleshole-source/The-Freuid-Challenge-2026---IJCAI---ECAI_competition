
# Model Architecture & Feature Engineering Specifications

This document outlines the technical details of the classification model, the validation architecture, and the digital forensic feature engineering used to classify document attacks.

---

## 🧠 Model Selection: EfficientNet-B0

The deep learning backbone of the pipeline is **EfficientNet-B0** (pre-trained on ImageNet). EfficientNet was selected for several key reasons:

* **High Texture Sensitivity:** It excels at picking up localized anomalies, boundary artifacts, and JPEG compression disparities introduced by GenAI-driven multimodal edits.
* **Parameter Efficiency:** With only ~5.3M parameters, it fine-tunes quickly on Kaggle GPUs or CPU setups without overfitting on limited training pools.
* **Custom Binary Output Head:** The original classification head has been replaced with:
  
  $$\text{Linear}(in, 1) \rightarrow \text{Sigmoid}(\cdot)$$

  This forces the network to produce a single continuous real-valued float within the range $[0, 1]$, where:
  * **0** represent bona-fide/genuine documents.
  * **1** represents attacks or manipulated fraudulent documents.

---

## 🛠️ Hybrid Forensic Feature Engineering

To combat highly complex "print-and-capture" attacks that suppress digital artifacts (closing the "analog hole"), we combine deep neural representations with classic computer vision forensics.

### 1. Fast Fourier Transform (FFT) Frequency Analysis
Print-and-capture operations often create repeating halftone printing arrays or pixel grids from display screens. We apply a 2D FFT to convert the gray channel to the frequency spectrum:

```python
f_transform = np.fft.fft2(gray)
f_shift = np.fft.fftshift(f_transform)
magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
```
​○ Genuine Documents: Show smooth spectral profiles.
○ ​Re-captured/Screen Documents: Exhibit high-frequency spikes corresponding to physical Moiré patterns.
​2. Laplacian Variance (Edge Sharpness Analysis)
​Natively digital uploads display high-frequency sharp borders. Physical print-and-capture processes degrade image crispness and introduce physical micro-blur. 
We track this using the variance of a Laplacian kernel:
```\sigma^2 = \text{Var}(\nabla^2 I)```
If ```\sigma^2```0 > 150, the file is classified as natively digital (is_digital = 1).
​If ```\sigma^2``` \le 150, it has undergone a physical recapture step (is_digital = 0).

​📈 Training and Validation Pipeline
​Optimization Parameters
​Optimizer: AdamW (Learning Rate: 1\times10^{-4} for stable gradient updates during fine-tuning).
​Loss Function: Binary Cross Entropy Loss (BCELoss).
​Batch Size: 16 (optimized for GPU VRAM limits).

Train / Validation Partitioning
​To ensure our calculated FREUID Score is mathematically sound and reflects general performance, the combined training data is systematically split:

                  [ Combined Input Data ]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [ Training Set (80%) ]           [ Validation Set (20%) ]
            │                                 │
            ▼                                 ▼
    Gradient Descent                   Model Evaluation
  (Update model weights)             (Calculate FREUID metric)

During validation evaluation, prediction outputs are aggregated across the unseen subset, and the DET curve is calculated to measure real production readiness.
