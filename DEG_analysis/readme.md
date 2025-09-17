# DEG Analysis Workflow 

**Differentially Expressed Genes (DEG) Analysis** 
1. **Data Cleaning & Dimensionality Reduction (PCA)**
2. **DEG Identification**
3. **DEG Visualization (Volcano Plot & Heatmap)**

---

##  Step 0: PCA Plot for Dimensionality Reduction

- **Libraries Used**:  
  `tidyverse`, `factoextra`, `limma`

- **Data Loading**:  
  The expression dataset is loaded from `"ExpSet_PostNorm.csv"`.

- **Prepare Data**:  
  - Transpose expression data (genes as columns, samples as rows).
  - Add group labels:  
    - 10 "Normal" samples  
    - 10 "Cervical_cancer" samples

- **Principal Component Analysis (PCA)**:  
  Applied using `prcomp()` on the dataset (excluding the Group column).
<img width="600" height="600" alt="pca" src="https://github.com/user-attachments/assets/b91810d7-b5eb-4829-83ac-fb48a910ddcf" />

---
## What Is PCA?

  Principal Component Analysis (PCA) is a dimensionality reduction technique.

  It transforms high-dimensional data  into a small number of components (PC1, PC2, …), capturing the most important variation.
Here is a **detailed explanation in Markdown format** of how the entire DEGs analysis workflow works:

---

# **Differentially Expressed Genes (DEGs) Identification**

##  **1. Model Matrix Design**

First, a **design matrix** is created to represent the sample group labels:

* 10 samples belong to the "Normal" group.
* 10 samples belong to the "Cervical\_cancer" group.
  This matrix encodes group information in a way suitable for linear modeling.

Then, column names of the matrix are set to "Normal" and "Cervical\_cancer" for clarity.

---

##  **2. Linear Model Fitting**

A linear model is fit for **each gene** in the normalized gene expression dataset.
The model estimates how gene expression depends on the condition (Normal vs Cervical Cancer).

---

##  **3. Contrast Matrix Design**

A **contrast matrix** is defined to specify the comparison of interest:

* Cervical Cancer vs Normal.
  This helps focus the analysis specifically on the difference between the two groups.

---

##  **4. Apply Contrast and Model Optimization**

The contrast matrix is applied to the fitted model, effectively comparing Cervical Cancer vs Normal.

Purpose:
**Empirical Bayes moderation** helps improve the statistical estimation of variances, especially for genes with low expression levels, where variance estimates tend to be unreliable.

* How It Works:
Instead of relying solely on the sample variance for each gene, the method shrinks the gene-wise variances towards a common (global) value. This stabilizes the variance estimates and improves the accuracy of differential expression testing.
---

##  **5. Extract Top Differentially Expressed Genes**

The `topTable()` function extracts the top genes based on:

* Log Fold Change (logFC).
* adjust.method = "BH":
Applies the Benjamini-Hochberg (BH) method to control the false discovery rate (FDR) when adjusting p-values for multiple testing.

 The function returns a table with:

* Gene names

* Log fold-change (logFC)

* Average expression (AveExpr)

* Moderated t-statistic (t)

* Raw p-value (P.Value)

* Adjusted p-value (adj.P.Val)

* B-statistic (log-odds
---

##  **6. Filter Significant DEGs**

From the saved DEGs table:

* Genes with adjusted P-value < 0.05
* And logFC > 2 or logFC < -2
  are considered significant and saved in a new file `finalDEGs.csv`.

---

##  **7. Gene Annotation (Adding Gene Symbols)**

* The final DEGs are annotated by mapping probe IDs to gene symbols using the `hgu133plus2.db` database.
* The probe IDs and corresponding gene symbols are combined with the DEG data and saved as `DEGs_Annotated.csv`.

---

##  **8. Visualization: Volcano Plot**

A Volcano Plot is generated to visualize DEGs:

* X-axis: log Fold Change (logFC).
* Y-axis: -log10(P-value).
* Red dots represent upregulated genes (logFC > 2, P < 0.05).
* Green dots represent downregulated genes (logFC < -2, P < 0.05).

This plot highlights significant DEGs.

<img width="600" height="600" alt="VolcanoPlot_FC_2" src="https://github.com/user-attachments/assets/9ea2a217-2748-43a2-a25b-0ec531a1ccf6" />


---

##  **9. Probe/Gene Annotation for Full Expression Dataset**

The normalized expression dataset is also annotated:

* Probe IDs are mapped to gene symbols.
* The annotated expression dataset is saved as `ExpSet_PostNorm_Annotated.csv`.

---

##  **10. Prepare Heatmap Data**

* Select genes that are highly upregulated (logFC ≥ 2.5) or highly downregulated (logFC ≤ -2.5).
* Extract corresponding gene expression values from the annotated expression dataset.
* Remove duplicate gene symbols.
* Prepare the dataset (`heatmap_data.csv`) for visualization.

---

##  **11. Visualization: Heatmap of DEG Expression**

A heatmap is created to show expression patterns of selected DEGs across samples:

* Red indicates low expression.
* Blue indicates high expression.
* Both row and column clustering are performed to visualize gene co-expression patterns.

The heatmap helps identify expression clusters and visualize up/downregulated genes clearly.
<img width="1300" height="1300" alt="heatmap_exp_deg_cluster" src="https://github.com/user-attachments/assets/14f43f70-5592-4d1d-903f-55b53c3a7c47" />


