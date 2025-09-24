
#  **ML-Powered_Cervical_Cancer_DEG_Profiling**  

![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)  
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)  
![Python](https://img.shields.io/badge/Python-3.8-blue?style=for-the-badge)  
![R](https://img.shields.io/badge/R-4.2.2-orange?style=for-the-badge)


# ML-Powered Cervical Cancer DEG Profiling

## About Cervical Cancer

Cervical cancer is a **malignant tumor of the cervix**, mainly caused by **persistent infection with high-risk HPV types**.
It ranks as the **4th most common cancer in women worldwide**.

Early detection using **gene expression profiling** and **biomarker discovery** is crucial for improving diagnosis, treatment planning, and survival outcomes.

---

## Project Goal

The aim of this project is to build a **robust and reproducible pipeline** to:

* Process and analyze **cervical cancer microarray gene expression data**.
* Identify **differentially expressed genes (DEGs)** between Normal and Cancer samples.
* Apply **machine learning models** for accurate sample classification.

---

## Workflow

### 1. Data Preprocessing

* Imported microarray datasets (GSE63514 from GEO).
* Applied **RMA (Robust Multi-array Average) normalization**.
* Performed quality control using boxplots and PCA plots.

### 2. Differential Gene Expression Analysis

* Conducted DEG analysis using **limma**.
* Annotated probes to gene symbols.
* Filtered DEGs based on **logFC** and **adjusted p-value** thresholds.

### 3. Visualization

* **Volcano plots** for significant DEGs.
* **Heatmaps** of top-ranked genes.
* **PCA plots** showing group separation.

### 4. Machine Learning Pipeline

* Models implemented:

  * k-Nearest Neighbors (kNN)
  * Support Vector Machine (SVM – Radial Kernel)
  * Random Forest Classifier
  * Logistic Regression
  * Decision Tree

* Steps performed:

  * Train/test split.
  * Model evaluation using accuracy, sensitivity, specificity.
  * Confusion matrix analysis.
  * Feature importance ranking.
  * Learning curve visualization.

---

## Dataset Details

| Source       | GEO Database                                |
| ------------ | ------------------------------------------- |
| Accession ID | **GSE63514**                                |
| Platform     | Affymetrix Human Genome U133 Plus 2.0 Array |
| Samples      | 10 Normal + 10 Cervical Cancer              |

🔗 [View Dataset on GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63514)

---

## Key Outcomes

* Identified **differentially expressed genes** relevant to cervical cancer.
* Built ML models achieving **high classification accuracy**.
* Highlighted **predictive biomarkers** for potential clinical relevance.
* Established a **reusable workflow** for microarray gene expression analysis.

---

## Future Directions

* Extend workflow to **RNA-Seq datasets**.
* Perform **pathway enrichment analysis** on DEGs.
* Explore **deep learning approaches** for improved classification.

---

✨ This repository demonstrates how **integrating genomics with machine learning** can help uncover biomarkers and build predictive models for cervical cancer research.

##  **License**

This project is licensed under the **MIT License** — Feel free to use and modify!  
 Open-source & community-driven.

---

##  **Contribute**

We  community contributions!  
Follow these simple steps:  
1.  Fork the repository  
2.  Create a feature branch  
3.  Open a pull request

---

##  **About Me**

**Vibhanshu Singh**  
 Developer & Maintainer of **ML-Powered_Cervical_Cancer_DEG_Profiling**  

📧 [vibhanshusingh78@gmail.com](mailto:vibhanshusingh78@gmail.com)  
🌐 [GitHub: Vibhanshusingh-001](https://github.com/Vibhanshusingh-001)

---

⭐ If this project helps you, consider giving it a star!
