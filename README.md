# Classification-and-Detection-of-Anemia-Using-Synergy-of-Machine-Learning-and-Image-Processing
An automated anemia detection system that combines image processing techniques and machine learning algorithms to classify different types of anemia from microscopic blood smear images. The system extracts morphological and texture-based features from red blood cells and uses an ensemble classifier to provide accurate predictions through a user-friendly GUI.

# Project Overview
Anemia is a common hematological disorder characterized by a deficiency of red blood cells or hemoglobin in the blood. Traditional anemia diagnosis involves manual examination of blood smear images by medical experts, which can be time-consuming and subjective.
This project proposes an automated anemia detection system that uses advanced image processing and machine learning techniques to analyze microscopic blood images and classify anemia types efficiently.
The system processes blood smear images, extracts meaningful morphological and texture features, and applies an ensemble machine learning model to classify the anemia type.

# Key Features
Automated Red Blood Cell (RBC) detection using Fly ROI technique
Image preprocessing using OpenCV
**Shape feature extraction**
 Area
 Perimeter
 Circularity
 Aspect Ratio
**Texture feature extraction**
 Gray Level Co-occurrence Matrix (GLCM)
 Local Binary Pattern (LBP)
**Feature scaling using StandardScaler
Class balancing using SMOTE
Ensemble machine learning model**
 Logistic Regression
 Random Forest
 XGBoost
**Soft Voting Classifier for improved prediction
Graphical User Interface (GUI) for real-time prediction
Visualization of results**
 Confusion Matrix
 ROC Curve
 Classification Report

# System Workflow

The proposed system follows the workflow below:

**1.Dataset Collection**
Microscopic blood smear images are collected and organized into labeled folders representing different anemia types.

**2.Image Preprocessing**
Images are converted to grayscale, smoothed using Gaussian filtering, and segmented using thresholding techniques.

**3.Region of Interest Detection (Fly Model)**
The system detects potential red blood cells by identifying contours and extracting individual regions of interest (ROIs).

**4.Feature Extraction**
For each detected cell, morphological and texture features are extracted using image processing methods.

**5.Feature Scaling and Balancing**
Extracted features are standardized using StandardScaler, and class imbalance is handled using SMOTE.

**6.Model Training**
Three machine learning models are trained:
Logistic Regression
Random Forest
XGBoost

**7.Ensemble Learning**
The predictions of individual models are combined using a VotingClassifier with soft voting to improve classification accuracy.

**8.Model Evaluation**
Performance is evaluated using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
ROC Curve

**9.GUI-based Prediction**
The trained model is integrated into a GUI application where users can upload images and obtain anemia classification results.

# Technologies Used

Python
OpenCV
NumPy
Scikit-learn
XGBoost
Matplotlib
Seaborn
Tkinter
Imbalanced-learn (SMOTE)

# Supported Anemia Types

Depending on the dataset, the system can classify:
**1.Microcytic Anemia**
**2.Macrocytic Anemia**
**3.Normocytic Anemia**
**4.Hemolytic Anemia**
**5.Sickle Cell Anemia**
**6.Non-Anemic**

# Project Structure
Anemia-Detection-ML
│
├── training_model.py
├── gui_code.py
│
├── fly_model.pkl
├── scaler.pkl
├── label_map.pkl
│
├── confusion_matrix.png
├── classification_report.png
│
├── dataset/
│   └── Labeled_Images1
│
└── README.md

# Dataset Availability
The dataset used in this project consists of real microscopic blood smear images obtained from clinical samples. These images were captured by medical professionals with the informed consent of the patients. To ensure patient privacy and prevent any potential misuse of sensitive medical data, the dataset has not been included in this public repository.

The dataset is organized in a folder named Labeled_Images1, which contains six subfolders corresponding to different anemia categories. Each subfolder stores the respective blood cell images required for training and testing the model.

Due to ethical and privacy considerations, the dataset is not publicly shared on GitHub. Researchers or developers who are interested in reproducing the results, modifying the project, or conducting further research may request access to the dataset. Upon a valid request and for academic or research purposes, the dataset can be shared privately.

If you require access to the dataset, please feel free to contact the repository author.

# Running the Training Model
python training_model.py
This will:

Extract features from images
Train the ensemble classifier
Generate evaluation metrics
Save the trained model

# Running the GUI Application
python gui_application.py
Steps in GUI:

Click Select Image
Upload a microscopic blood image
System extracts features
Model predicts anemia type
Results and confidence score are displayed

# Performance Metrics
The model performance is evaluated using:

 Accuracy
 Precision
 Recall
 F1 Score
 Confusion Matrix
 ROC Curve
These metrics help measure how effectively the system classifies different anemia types.
