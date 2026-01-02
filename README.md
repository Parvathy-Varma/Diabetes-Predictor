## 🩺 Diabetes Prediction Using Support Vector Machine (SVM)

This project implements a **machine learning–based diabetes prediction system** using the **PIMA Indians Diabetes Dataset**. The model predicts whether a female patient is diabetic or non-diabetic based on medical and demographic features.

---

### 📌 Project Overview

The goal of this project is to build a reliable **binary classification model** that can predict diabetes using **Support Vector Machine (SVM)** with a linear kernel. The system also includes a **predictive interface** that allows users to input new patient data and receive a real-time prediction.

---

### 📊 Dataset

* **Dataset**: PIMA Indians Diabetes Dataset
* **Source**: UCI Machine Learning Repository / Kaggle
* **Target Variable**: `Outcome`

  * `0` → Non-Diabetic
  * `1` → Diabetic
* **Features Used**:

  * Number of Pregnancies
  * Glucose Level
  * Blood Pressure
  * Skin Thickness
  * Insulin Level
  * Body Mass Index (BMI)
  * Diabetes Pedigree Function
  * Age

> Note: The dataset contains data for female patients only.

---

### ⚙️ Workflow

1. **Import Dependencies**

   * NumPy, Pandas for data handling
   * Scikit-learn for preprocessing, model training, and evaluation

2. **Data Loading & Exploration**

   * Load dataset using Pandas
   * Analyze shape, statistics, and class distribution

3. **Data Preprocessing**

   * Separate features (`X`) and labels (`Y`)
   * Standardize feature values using `StandardScaler`

4. **Train-Test Split**

   * Split data into **80% training** and **20% testing**
   * Use stratification to maintain class balance

5. **Model Training**

   * Train a **Support Vector Classifier (SVC)** with a linear kernel

6. **Model Evaluation**

   * Evaluate accuracy on both training and test datasets
   * Detect possible overfitting

7. **Prediction System**

   * Accepts user input for medical parameters
   * Standardizes input data
   * Predicts diabetes status using the trained model

---

### 🧠 Machine Learning Model

* **Algorithm**: Support Vector Machine (SVM)
* **Kernel**: Linear
* **Why SVM?**

  * Effective for binary classification
  * Works well with standardized numerical data
  * Handles high-dimensional feature spaces efficiently

---

### 📈 Performance

* Accuracy is calculated for:

  * **Training Data**
  * **Testing Data**
* A balanced accuracy score indicates good generalization and minimal overfitting.

---

### 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn

---

### 🚀 Future Improvements

* Use additional models (Logistic Regression, Random Forest, XGBoost)
* Perform hyperparameter tuning
* Add a GUI or web interface
* Handle missing or zero values more effectively

