# 🧠 Handwritten Character Recognition using CNN

This project implements a **Convolutional Neural Network (CNN)** to accurately classify handwritten English alphabet characters (A–Z). The model is trained using the **A–Z Handwritten Characters Dataset** provided in CSV format.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🔍 Overview

The goal of this project is to build a deep learning model that can:
- **Classify 26 alphabet characters** from images of handwritten text.
- **Preprocess and visualize** character data from CSV.
- **Train a CNN model** to achieve high accuracy.
- **Evaluate and demonstrate predictions** on test samples.

---

## 📂 Dataset

- **Name**: A–Z Handwritten Alphabets Dataset  
- **Source**: [Kaggle – AZ Handwritten Data](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)  
- **Format**: CSV file with 785 columns:  
  - Column 0: Character label (0 = A, 25 = Z)  
  - Columns 1–784: Pixel values (28x28 grayscale image)

### Sample Snapshot:

| label | pixel1 | pixel2 | ... | pixel784 |
|-------|--------|--------|-----|----------|
|   0   |   0    |   0    | ... |    0     |

---

## 🏗️ Model Architecture

The CNN model is built using the Keras `Sequential()` API:

- **Input Shape**: `(28, 28, 1)`
- **Layers**:
  - `Conv2D` with ReLU activation
  - `MaxPooling2D`
  - `Dropout` for regularization
  - `Flatten`
  - Fully connected `Dense` layers
  - Output layer with `Softmax` activation (26 classes)

### Optimizer & Callbacks:

- **Optimizer**: SGD (Stochastic Gradient Descent)
- **Loss Function**: Categorical Crossentropy
- **Callbacks Used**:
  - `EarlyStopping` to prevent overfitting
  - `ReduceLROnPlateau` to dynamically reduce the learning rate

---

## ✅ Results

- **Training Accuracy**: > 98%
- **Validation Accuracy**: ~95%–97%
- **Loss & Accuracy curves** plotted for model evaluation
- Sample predictions displayed with actual vs predicted labels

---

## ⚙️ Installation & Setup

### Prerequisites

Ensure the following Python packages are installed:

```bash
pip install numpy pandas matplotlib opencv-python scikit-learn keras tensorflow
