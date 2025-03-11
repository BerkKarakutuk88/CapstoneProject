# ML Algorithm Selector Application

## Overview
This is a GUI-based machine learning application that allows users to select different machine learning algorithms to train and evaluate on a given dataset. The application is built using `customtkinter` for the UI and integrates machine learning models from `scikit-learn`.

## Features
- Supports four machine learning algorithms:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (K-NN)
  - Random Forest
- Allows users to upload a dataset (CSV format)
- Prepares the data for training and testing
- Displays Confusion Matrix and ROC Curve for model evaluation
- Provides a dark and light theme option

## Requirements
Make sure you have the following dependencies installed before running the application:

```bash
pip install customtkinter scikit-learn pandas numpy matplotlib
```

## How to Use
1. Run the script:
   ```bash
   python Capstone.py
   ```
2. Select a theme (Dark or Light) from the dropdown menu.
3. Click the "Select File" button to upload a dataset (CSV format).
4. Choose a machine learning algorithm from the dropdown menu.
5. Click the "Run Algorithm" button to train and evaluate the selected model.
6. The results (Confusion Matrix and ROC Curve) will be displayed in a new window.

## Dataset Format
- The dataset should be in CSV format.
- The script expects the dataset to contain a target variable in the first column.
- Some columns ('name', 'session', 'Unnamed: 73', 'Unnamed: 74') are automatically dropped.
- Features are standardized using `StandardScaler` before training.

## Notes
- The application processes data asynchronously to prevent UI freezing.
- Uses threading and queues to manage model training and result retrieval.
- Supports binary classification problems for ROC Curve evaluation.


