# 😊 Deep Emotion Facial Expression Recognition Using Attentional CNN

## 📌 Overview

Deep Emotion is a Facial Expression Recognition (FER) system developed using PyTorch and Convolutional Neural Networks (CNNs). The model incorporates a Spatial Transformer Network (STN) to automatically focus on important facial regions before classification, improving the network's ability to recognize human emotions from facial images.

The system is trained on grayscale facial images and classifies them into seven emotion categories. It also supports TensorBoard visualization and class-weighted loss functions to address class imbalance during training.

---

## 🎯 Objectives

* Detect and classify human emotions from facial images.
* Improve feature extraction using Spatial Transformer Networks (STN).
* Train a robust deep learning model using PyTorch.
* Handle imbalanced emotion datasets using weighted loss functions.
* Visualize training performance using TensorBoard.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* TensorBoard
* NumPy
* Pillow (PIL)
* CUDA (GPU Acceleration)
* Convolutional Neural Networks (CNN)
* Spatial Transformer Networks (STN)
* Deep Learning
* Computer Vision

---

## 📂 Project Structure

```text
Deep-Emotion-Facial-Expression-Recognition/
│
├── data_loaders.py      # Dataset loading and preprocessing
├── deep_emotion.py      # CNN + STN model architecture
├── generate_data.py     # Dataset generation and splitting
├── main.py              # Training pipeline with TensorBoard support
├── visualize.py         # Visualization utilities
├── README.md
```

---

## 🧠 Model Architecture

```text
Input Image (48×48 Grayscale)
            │
            ▼
Spatial Transformer Network (STN)
            │
            ▼
Convolution Layer 1
            │
            ▼
Convolution Layer 2
            │
            ▼
Max Pooling
            │
            ▼
Convolution Layer 3
            │
            ▼
Convolution Layer 4
            │
            ▼
Batch Normalization
            │
            ▼
Max Pooling
            │
            ▼
Dropout
            │
            ▼
Fully Connected Layer
            │
            ▼
Output Layer (7 Classes)
```

---

## 😀 Emotion Classes

The model predicts the following emotions:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

---

## ⚙️ Features

* Facial Emotion Recognition using Deep Learning
* Spatial Transformer Network (Attention Mechanism)
* CNN-based Feature Extraction
* TensorBoard Training Visualization
* GPU Training Support (CUDA)
* Class-Weighted Loss for Imbalanced Datasets
* Automatic Dataset Preparation
* Validation Accuracy Monitoring

---

## 📊 Dataset

The project uses a Kaggle facial expression dataset containing grayscale facial images and emotion labels.

Expected files:

```text
dataset/
│
├── train.csv
├── test.csv
```

During preprocessing, the dataset is automatically split into:

```text
train/
val/
test/
```

using the dataset generation utilities.

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/Varunvelagandula/Deep-Emotion-Facial-Expression-Recognition.git
cd Deep-Emotion-Facial-Expression-Recognition
```

### Install Dependencies

```bash
pip install torch torchvision numpy pillow tensorboard
```

---

## ▶️ Dataset Setup

Generate training, validation, and testing images:

```bash
python main.py --setup True --data dataset_path
```

---

## ▶️ Train the Model

Default Training:

```bash
python main.py --train True --data dataset_path
```

Custom Hyperparameters:

```bash
python main.py \
--train True \
--data dataset_path \
--hyperparams True \
--epochs 100 \
--learning_rate 0.005 \
--batch_size 128
```

Training with Class Weights:

```bash
python main.py \
--train True \
--data dataset_path \
--cweights True
```

---

## 📈 TensorBoard Visualization

Launch TensorBoard:

```bash
tensorboard --logdir=runs
```

Open:

```text
http://localhost:6006
```

Track:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy

---

## 💾 Model Output

After training, model weights are saved automatically:

```text
deep_emotion-{epochs}-{batch_size}-{learning_rate}.pt
```

Example:

```text
deep_emotion-100-128-0.005.pt
```

---

## 🎯 Applications

* Human-Computer Interaction
* Smart Surveillance Systems
* Emotion-Aware AI Applications
* Mental Health Monitoring
* Educational Technologies
* Customer Sentiment Analysis
* Social Robotics

---

## 🔮 Future Improvements

* Real-Time Webcam Emotion Detection
* Video-Based Emotion Recognition
* Transfer Learning with ResNet/EfficientNet
* Deployment as a Web Application
* Mobile Application Integration


