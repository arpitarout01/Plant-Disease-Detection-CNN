# 🌿 Plant Disease Detection using CNN

A deep learning project that detects plant diseases from leaf images using a Convolutional Neural Network (CNN). The model classifies images into 38 different categories based on plant type and disease type.

## 📌 Overview

Plant diseases can drastically affect crop yield and quality. Early and accurate detection can help farmers take timely action. This project utilizes a CNN-based image classification model to identify plant diseases from images of leaves.

## 📂 Dataset

- **Source:** PlantVillage Dataset from kaggle
- **Classes:** 38 (e.g., Apple Scab, Tomato Mosaic Virus, Potato Blight, etc.)
- **Folders:**
  - `train/` - Training data
  - `valid/` - Validation data
- Images are organized in subdirectories named after the plant and disease.

## 🧠 Model Architecture

- Custom CNN with:
  - Convolutional Layers
  - Batch Normalization
  - Max Pooling
  - Dropout Layers
  - Fully Connected Layers
- Loss Function: `categorical loss`
- Optimizer: `Adam`
- Framework: Keras / TensorFlow 
