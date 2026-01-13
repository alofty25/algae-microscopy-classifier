# algae-microscopy-classifier
Deep learning system for automatic identification and classification of algae species from microscopic images using CNN-based computer vision


## 📌 Project Overview

This project focuses on the **automated classification of algae morphotypes** from microscopic images using deep learning techniques. The goal is to assist biological and environmental analysis by accurately identifying different algae forms through image-based classification.

The system is designed to recognize multiple algae morphotypes such as **spherical, filamentous, colonial**, and others, directly from microscopy images.

### Members 
* Ahmed Mohamed Lotfy 22P0251 - [@alofty25](https://github.com/alofty25)
* Adham Hisham Kandil 22P0217 - [@Kandil122](https://github.com/Kandil122)
---

## 🎯 Key Features

* Automated **multi-class classification** of algae morphotypes
* Supports identification from **microscopic image datasets**
* Uses **Convolutional Neural Networks (CNNs)** with transfer learning
* Built using **PyTorch** for scalability and deployment
* Robust preprocessing and data augmentation for microscopy images

---

## 🧠 Model Approach

* Pretrained CNN architectures (e.g., ResNet, MobileNet, EfficientNet)
* Fine-tuned on labeled algae microscopy images
* Softmax-based multi-class prediction
* Evaluation using accuracy, precision, recall, and confusion matrix

---

### 🗂️  Project Structure

```
algae-microscopy-classifier/
├── configs/
│   └── config.yaml              # Configuration file
├── data/
│   ├── train/
│   │   ├── images/              # Training images
│   │   └── labels/              # Training labels (YOLO format)
│   ├── test/
│   │   ├── images/              # Test images
│   │   └── labels/              # Test labels
│   └── data.yaml                # Dataset configuration
├── src/
│   └── data/
│       ├── __init__.py
│       ├── Data_import.py       # Dataset download utilities
│       ├── algae_dataset.py     # Custom PyTorch Dataset
│       └── dataloader.py         # DataLoader utilities
├── notebooks/
│   ├── exploratary_data_analysis.ipynb
│   ├── feature_importance_fisher.ipynb
│   ├── traditional_ml_classifiers.ipynb
│   ├── train_cnn_classifier.ipynb
│   ├── verify_data_pipeline.ipynb
│   ├── visualize_algae.ipynb
│   └── yolo_based_evaluation.py
├── outputs/
│   ├── models/
│   │   └── best_algae_cnn.pth   # Trained CNN model
│   ├── best_ml_model_xgboost.pkl
│   ├── feature_scaler.pkl
│   ├── extracted_features.csv
│   ├── fisher_scores_all_features.csv
│   ├── recommended_features.csv
│   ├── confusion_matrices.png
│   ├── detection_examples.png
│   └── ml_models_comparison.png
├── main.py                       # Main entry point
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project configuration
└── README.md                     # Project overview
```

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Deep Learning Framework:** PyTorch / TensorFlow
* **Image Processing:** OpenCV
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Python Scripts

---

## 📊 Dataset

* Microscopic algae images with labeled morphotypes
* Images resized and normalized before training
* Data augmentation applied to improve generalization

*(Dataset source and details should be documented here)*

---

## 🚀 Getting Started

1. Clone the repository
2. Install required dependencies
3. Prepare the dataset directory
4. Train or load a pretrained model
5. Run inference on microscopy images

---

## 📈 Results

* Training and validation accuracy plots
* Confusion matrix for morphotype classification
* Sample predictions with visual outputs

---

## 🔍 Limitations & Future Work

* Performance depends on dataset quality and class balance
* Future work may include:

  * Species-level classification
  * Segmentation-based preprocessing
  * Real-time microscope integration

---

## 📄 License

This project is intended for **educational and research purposes**.


