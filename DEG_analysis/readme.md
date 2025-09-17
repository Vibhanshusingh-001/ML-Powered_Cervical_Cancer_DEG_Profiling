# DEG Analysis Workflow

**Differentially Expressed Genes (DEG) Analysis Steps**
1. Data Cleaning & Dimensionality Reduction (PCA)  
2. DEG Identification  
3. DEG Visualization (Volcano Plot & Heatmap)  



## Step 0: PCA Plot for Dimensionality Reduction

**Libraries Used**  
`tidyverse`, `factoextra`, `limma`

**Data Loading**  
Expression dataset is loaded from:  
`"ExpSet_PostNorm.csv"`

**Prepare Data**  
- Transpose expression data (genes → columns, samples → rows).  
- Add group labels:  
  - 10 "Normal" samples  
  - 10 "Cervical_cancer" samples  

**Principal Component Analysis (PCA)**  
Applied using `prcomp()` on the dataset (excluding Group column).

<img width="600" height="600" alt="pca" src="https://github.com/user-attachments/assets/5f4ef32d-1301-42eb-9f29-81f6009bf2c6" />



# Differentially Expressed Genes (DEGs) Identification

## 1. Model Matrix Design

- Create a **design matrix** to represent the sample group labels:  
  - 10 samples in the "Normal" group  
  - 10 samples in the "Cervical_cancer" group  
- Column names are set to "Normal" and "Cervical_cancer" for clarity.



## 2. Linear Model Fitting

A linear model is fit for **each gene** to estimate how gene expression depends on the condition (Normal vs Cervical Cancer).



## 3. Contrast Matrix Design

Define a **contrast matrix** to specify the comparison:  
Cervical Cancer vs Normal.  
This focuses the analysis specifically on the difference between the two groups.



## 4. Apply Contrast and Model Optimization

The contrast matrix is applied to the fitted model to compare Cervical Cancer vs Normal.

**Empirical Bayes moderation** improves statistical variance estimation, especially for genes with low expression.  
- Instead of relying only on sample variance per gene, the method shrinks variances towards a global value.  
- This stabilizes the estimates and improves differential expression testing accuracy.



## 5. Extract Top Differentially Expressed Genes

Use `topTable()` function to extract top genes based on:  
- log Fold Change (logFC)  
- Benjamini-Hochberg (BH) adjustment for multiple testing.

The result table includes:  
- Gene name  
- logFC  
- Average Expression (AveExpr)  
- Moderated t-statistic (t)  
- Raw p-value (P.Value)  
- Adjusted p-value (adj.P.Val)  
- B-statistic (log-odds)



## 6. Filter Significant DEGs

From the DEGs table, select genes where:  
- Adjusted P-value < 0.05  
- logFC > 2 or logFC < -2  

Save significant DEGs to `finalDEGs.csv`.



## 7. Gene Annotation (Add Gene Symbols)

- Map probe IDs to gene symbols using the `hgu133plus2.db` database.  
- Save the annotated DEGs as `DEGs_Annotated.csv`.


## 8. Visualization: Volcano Plot

- X-axis: log Fold Change (logFC)  
- Y-axis: -log10(P-value)  
- Red dots: upregulated genes (logFC > 2, P < 0.05)  
- Green dots: downregulated genes (logFC < -2, P < 0.05)

![Volcano Plot](https://github.com/user-attachments/assets/9ea2a217-2748-43a2-a25b-0ec531a1ccf6)



## 9. Probe/Gene Annotation for Full Expression Dataset

- Map probe IDs to gene symbols.  
- Save as `ExpSet_PostNorm_Annotated.csv`.



## 10. Prepare Heatmap Data

- Select genes that are:  
    - Highly upregulated (logFC ≥ 2.5)  
    - Highly downregulated (logFC ≤ -2.5)  
- Extract corresponding expression values from annotated expression dataset.  
- Remove duplicate gene symbols.  
- Save as `heatmap_data.csv`.



## 11. Visualization: Heatmap of DEG Expression

- Red color indicates low expression.  
- Blue color indicates high expression.  
- Perform row and column clustering to show gene co-expression patterns.

<img width="1000" height="1000" alt="heatmap_exp_deg_cluster" src="https://github.com/user-attachments/assets/37782768-8c5b-43ca-96e6-1fdcd39ccaff" />

