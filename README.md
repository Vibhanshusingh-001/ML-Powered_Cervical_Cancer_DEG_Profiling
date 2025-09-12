# 🧬 Cervical-Cancer-DEG-Analysis-Using-ML

---

## 🧠 About Cervical Cancer

Cervical cancer is a malignant tumor of the cervix, primarily caused by **persistent infection with high-risk types of human papillomavirus (HPV)**. It is the **fourth most common cancer in women worldwide**.  
Early detection through **gene expression profiling** and **biomarker discovery** can significantly improve diagnosis and treatment strategies.

---

## 🎯 Project Overview

This project focuses on analyzing gene expression data for **Cervical Cancer** using **microarray data (Affymetrix platform)**.  
It includes:

- RMA normalization  
- Exploratory analysis (PCA, boxplots)  
- DEG identification using `limma`  
- Annotation with gene symbols  
- Visualizations: Volcano plot & heatmap  
- ML-based classification using **kNN**, **SVM**, and **Random Forest**

---

## 📂 Dataset Details

- **Source**: [GEO Database](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63514)  
- **Accession ID**: GSE63514  
- **Platform**: Affymetrix Human Genome U133 Plus 2.0 Array  
- **Samples**: 10 Normal + 10 Cervical Cancer

---

## 🚀 Workflow Summary

### 1. Load Data & Normalize
- Read raw `.CEL` files using `ReadAffy`
- Perform RMA normalization with `affy`
- Export normalized expression matrix (`ExpSet_PostNorm.csv`)

### 2. Visualization
- **Boxplots**: Compare expression distributions pre- and post-normalization
- **PCA Plot**: Visualize group separation (Normal vs. Cancer)

### 3. Differential Gene Expression (DEG) Analysis
- Use `limma` for linear modeling
- Create contrast matrix: `Cervical Cancer - Normal`
- Extract top DEGs by `logFC` and `adjusted p-value`
- Export DEG results (`Result_Table_logFCsorted.csv` and `finalDEGs.csv`)
- Annotate probe IDs to gene symbols using `hgu133plus2.db`

### 4. Heatmap
- Filter highly upregulated/downregulated genes (|logFC| ≥ 2.5)
- Generate expression heatmap using `heatmap.2`

### 5. Machine Learning Models
- Prepare labeled dataset using DEG expression values
- Split into training and testing sets (60/40)
- Apply:
  - **kNN (k-Nearest Neighbors)**
  - **SVM (Support Vector Machine - Radial Kernel)**
  - **Random Forest**
- Evaluate using:
  - Accuracy
  - Confusion matrix
  - Feature importance plots

---
