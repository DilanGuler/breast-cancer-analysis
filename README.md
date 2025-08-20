# Breast Cancer Visualization and Classification

## Abstract
Breast cancer is a leading cause of illness and death worldwide. Early and reliable classification can support timely clinical decisions. This project evaluates classical and modern machine learning approaches on the **Wisconsin Breast Cancer Diagnostic (WBCD)** dataset to classify tumors as **benign** or **malignant**. Models implemented include **Logistic Regression**, **Random Forest**, and a **simple Deep Neural Network (MLP)**. Models are compared using **nested cross-validation**, and **calibration** is assessed (Platt scaling, isotonic regression, Brier score, reliability curves). Clinical utility is further analyzed via **ROC** and **Precision–Recall** curves and **Decision Curve Analysis (DCA)**. Interpretability is provided using **SHAP** (for the tree-based model) and **LIME** (for the neural network), with instance-level explanations. The dataset is loaded directly from `scikit-learn`, ensuring a fully reproducible pipeline without external data files.

## Introduction – A Brief Overview on Breast Cancer
Breast cancer develops when abnormal cells in breast tissue grow uncontrollably and may invade surrounding tissue or metastasize. While many breast lumps are **benign**, distinguishing them from **malignant** tumors is crucial to avoid unnecessary procedures and ensure timely treatment when needed. Algorithmic decision support can complement clinical judgment by providing calibrated risk estimates and transparent, case-level explanations that highlight the most influential features behind each prediction.

## Dataset Description
This project uses the **Wisconsin Breast Cancer Diagnostic (WBCD)** dataset, available via `sklearn.datasets.load_breast_cancer()`. The dataset contains **569** samples and **30** numeric features computed from digitized **fine needle aspirate (FNA)** images of breast masses. The target has two classes: **benign** and **malignant**. For clinical clarity, labels are mapped to `0 = benign` and `1 = malignant` in this repository.

Ten fundamental feature types are computed for each cell nucleus (with mean, standard error, and “worst” values derived for each):
- **radius** (mean distance from center to perimeter points)  
- **texture** (standard deviation of gray-scale values)  
- **perimeter**  
- **area**  
- **smoothness** (local variation in radius lengths)  
- **compactness** \((\text{perimeter}^2 / \text{area} - 1.0)\)  
- **concavity** (severity of concave portions of the contour)  
- **concave points** (number of concave portions of the contour)  
- **symmetry**  
- **fractal dimension** (“coastline approximation” − 1)

**Data access:**  
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target

