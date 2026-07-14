# Identity Document Fraud & Forgery Detector

An end-to-end computer vision and deep learning pipeline designed to detect bona-fide identity documents from fraudulent ones.

This repository is specifically optimized to identify digital alterations (such as GenAI-driven multimodal edits) and physical-to-digital conversions (such as print-and-capture forgeries that bypass traditional digital watermark analysis).

---

## 📌 Project Overview

This program operates in three sequential phases:
1. **Phase 1 (Specification Extraction):** Scans genuine and fraudulent directories to generate deep structural metadata and frequency domain analysis, outputting `true_specs.json` and `false_specs.json`.
2. **Phase 2 (Model Training & Evaluation):** Trains an image-classification model (EfficientNet-B0) to output real-valued fraud probability scores. Performance is validated locally using a custom **FREUID Metric**.
3.  **Phase 3 (Production Inference):** Processes unseen target files, determines if the document is natively digital or a physical re-capture, extracts the country/type from file naming schemas, and exports results to a submission-ready `submissions.csv`.

---

## 📁 Repository Structure

```text
├── identity_detector.py      # Main pipeline script (Local or Kaggle run)
├── true_documents/           # Directory with authentic IDs / Drivers Licenses
├── false_documents/          # Directory with manipulated/counterfeit documents
├── test_documents/           # Directory with test target files for inference
├── true_labels.csv           # Label mapping for true documents (all 0s)
├── false_labels.csv          # Label mapping for false documents (all 1s)
├── README.md                 # Project Overview (This file)
└── model.md                  # Deep Model and Feature Engineering Spec
```
🔧 Installation & Requirements
​Ensure you have Python 3.8+ installed along with the required processing libraries:
```
pip install torch torchvision opencv-python scikit-learn pandas numpy pillow

```
If running inside a Kaggle Notebook, configure your notebook settings to use a GPU Accelerator (T4 or P100) for optimal training and inference speeds.

​📊 The FREUID Metric
​The model's performance is strictly bound by the custom FREUID Score (where lower is better). The score combines global performance with performance at a strict production operating point.
​

Mathematical Formulation

​AuDET (Area under the Detection Error Trade-off curve): Measures the global trade-off between False Rejection Rates (BPCER) and False Acceptance Rates (APCER).
​APCER @ 1% BPCER: Measures the Attack Presentation Classification Error Rate at a fixed 1\% False Alarm limit.
​We calculate "goodness" scores:

$g_{\text{audet}} = 1 - AuDET$

$$g_{\text{apcer}} = 1.0 - \text{APCER}_{@1\%\text{BPCER}}$$

The final score is the inverted harmonic mean of these values:
$\text{FREUID} = 1 - \frac{2 \cdot g_{\text{audet}} \cdot g_{\text{apcer}}}{g_{\text{audet}} + g_{\text{apcer}}}$

🚀 Execution Guide
​To run the entire pipeline on your local machine:
​Place your datasets inside /kaggle/working/selected_true_cards, /kaggl/working/selected_false_cards/, and /kaggle/input/the-freuid-challenge-2026-ijcai-ecai/private-test/private-test.
​Setup your CSVs mapping filenames to binary classes (0 for genuine, 1 for attack).
​Run the pipeline:
```text

python identity_detector.py
```
Expected Output Files:
`​true_specs.json`: Technical parameters extracted from genuine files.
`​false_specs.json`: Technical parameters extracted from fraudulent files.
​`submissions.csv`: Final inference data containing the following structure:

|image id | image path | label |
|ID_0921 | private_test/US_ID_0921.jpg | 0.984210 |
|ID_0411 | private_test/DE_DL_0411.png | 0.001041 |
