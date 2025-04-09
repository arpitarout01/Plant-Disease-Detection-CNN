# Plant Disease Detection using CNN

This repository contains the CNN model code for detecting plant diseases from leaf images using the PlantVillage dataset. The goal is to automate plant disease classification using deep learning.

---

## 📌 Project Overview

- **Dataset**: PlantVillage (38 classes), Link : https://www.kaggle.com/datasets/mohitsingh1804/plantvillage
- **Model**: Custom Convolutional Neural Network (CNN)
- **Framework**: TensorFlow / Keras
- **Image Size**: 128x128
- **Classification Type**: Multiclass (categorical)

---

## 🧪 Methodology

1. **Data Loading & Preprocessing**
   - Images loaded using `image_dataset_from_directory`
   - Resized to 128x128
   - Labels one-hot encoded

2. **CNN Architecture**
   - Multiple Conv2D layers with increasing filters (32→64→128→256)
   - Batch Normalization and MaxPooling
   - Dense layer with Dropout
   - Output layer: 38 softmax units

3. **Training**
   - Optimizer: Adam
   - Loss: Categorical Crossentropy
   - Metrics: Accuracy
   - Callbacks: EarlyStopping & ReduceLROnPlateau

---

## 💻 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/arpitarout01/Plant-Disease-Detection-CNN.git
   cd Plant-Disease-Detection-CNN
