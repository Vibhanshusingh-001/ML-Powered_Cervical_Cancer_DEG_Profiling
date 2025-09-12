
# 🧬 **Cervical Cancer DEG Analysis using Machine Learning**  

![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)  
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)  
![Python](https://img.shields.io/badge/Python-3.8-blue?style=for-the-badge)  
![R](https://img.shields.io/badge/R-4.2.2-orange?style=for-the-badge)

---

## 💡 **About Cervical Cancer**

Cervical cancer is a **malignant tumor of the cervix**, primarily caused by **persistent infection with high-risk HPV types**.  
🌍 It ranks as the **4th most common cancer in women worldwide**.

🔬 Early detection via **gene expression profiling** and **biomarker discovery** plays a crucial role in enhancing diagnosis, treatment decisions, and survival rates.

---

## 🚀 **Project Goal**

Develop a robust pipeline to analyze **Cervical Cancer gene expression data** and apply **machine learning models** for accurate classification of samples into **Normal vs. Cancer** groups.

---

## ✅ **What’s Inside This Project?**

✔️ Data Preprocessing:  
- RMA normalization  
- Quality control with boxplots & PCA visualization  

✔️ Differential Expression Analysis:  
- Identify DEGs using `limma`  
- Annotate probes to gene symbols  

✔️ Visualization:  
- Volcano plot  
- Expression heatmap  

✔️ Machine Learning Models:  
- k-Nearest Neighbors (kNN)  
- Support Vector Machine (SVM – Radial Kernel)  
- Random Forest Classifier  

---

## 📂 **Dataset Details**

| 📁 **Source** | GEO Database |
|-------------|-------------|
| 🆔 **Accession ID** | GSE63514 |
| 🧱 **Platform** | Affymetrix Human Genome U133 Plus 2.0 Array |
| 👥 **Samples** | 10 Normal + 10 Cervical Cancer |

🔗 [Explore the Dataset on GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63514)

---

## 🧱 **Workflow Summary**

### 1️⃣ Data Loading & Normalization  
- Load `.CEL` files via `ReadAffy`  
- Perform **RMA normalization**  
- Export processed expression data:  
  ➡️ `ExpSet_PostNorm.csv`

### 2️⃣ Exploratory Data Analysis (EDA)  
- 📊 Boxplots: Expression distribution before & after normalization  
- 🎯 PCA plots: Visual group separation (Normal vs. Cancer)

### 3️⃣ DEG Analysis  
- Linear modeling with `limma`  
- Define contrast: **Cervical Cancer – Normal**  
- Filter DEGs by `logFC` & adjusted p-value  
- Export results:  
    - `Result_Table_logFCsorted.csv`  
    - `finalDEGs.csv`  
- Annotate using `hgu133plus2.db`

### 4️⃣ Heatmap Visualization  
- Select highly significant DEGs (|logFC| ≥ 2.5)  
- Plot heatmap: Expression patterns across samples

### 5️⃣ Machine Learning Pipeline  
- Prepare labeled dataset  
- Train/test split (60%/40%)  
- Apply models:  
    - 🧱 **kNN**  
    - ⚛️ **SVM (Radial Kernel)**  
    - 🌳 **Random Forest**  
- Evaluate:  
    - 📈 Accuracy  
    - ✅ Confusion Matrix  
    - ⭐ Feature Importance

---

## 📜 **License**

This project is licensed under the **MIT License** — Feel free to use and modify!  
🔓 Open-source & community-driven.

---

## 🤝 **Contribute**

We ❤️ community contributions!  
Follow these simple steps:  
1. 🍴 Fork the repository  
2. 🌿 Create a feature branch  
3. 🚀 Open a pull request

---

## 👨‍💻 **About Me**

**Vibhanshu Singh**  
🔧 Developer & Maintainer of **MetaOmics-ML**  

📧 [vibhanshusingh78@gmail.com](mailto:vibhanshusingh78@gmail.com)  
🌐 [GitHub: Vibhanshusingh-001](https://github.com/Vibhanshusingh-001)

---

⭐ If this project helps you, consider giving it a star!
