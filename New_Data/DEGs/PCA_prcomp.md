

## 1. What is PCA?

Principal Component Analysis (PCA) is a statistical technique used to **simplify large and complex datasets** while retaining most of the important information.

In gene expression analysis, PCA reduces **thousands of genes** into a few new variables called **Principal Components (PCs)** that capture the major patterns in the data.

---

## 2. Why PCA is needed in gene expression analysis

Gene expression data usually has:
- Thousands of genes (high dimensionality)
- Fewer samples
- High correlation between genes
- Noise and redundancy

PCA helps to:
- Reduce data complexity
- Visualize samples in 2D or 3D
- Identify biological patterns
- Detect outliers
- Detect batch effects
- Check separation between conditions (e.g., Normal vs Disease)

---

## 3. What is a Principal Component (PC)?

A Principal Component is a **new axis** created from the original gene expression data.

Key points:
- PC1 captures the **maximum variation**
- PC2 captures the **second highest variation**
- PC3 captures the next highest variation
- Each PC is independent (orthogonal) to others
- Each PC is a combination of all genes

---

## 4. What do we mean by a “new axis”?

In the original dataset:
- Each gene represents one axis
- Thousands of genes mean thousands of axes

PCA **does not use these original gene axes**.  
Instead, it creates **new axes** that better describe how the data varies.

These new axes are:
- Principal Component 1 (PC1)
- Principal Component 2 (PC2)
- Principal Component 3 (PC3), and so on

---

## 5. Intuitive idea behind a new axis

Imagine plotting samples using only two genes:
- X-axis = Gene 1
- Y-axis = Gene 2

If the points lie along a **diagonal direction**:
- The diagonal explains the data better than X or Y alone
- PCA rotates the coordinate system to align with this direction

That diagonal direction becomes **PC1**.

So:
- PC1 is a **rotated axis**
- It captures the strongest pattern in the data

---

## 6. Rule used by PCA to find PC1

PCA follows one main rule:

> **Find the direction along which the data variance is maximum**

This means:
- Consider all possible directions
- Measure how spread out the data is along each direction
- Choose the direction with the largest spread

That direction becomes **Principal Component 1 (PC1)**.

---

## 7. How PCA finds new axes mathematically

### Step 1: Mean-centering the data
- Calculate the mean expression for each gene
- Subtract the mean from each value
- Centers the data around zero

Purpose:
- PCA focuses on variation, not absolute expression levels

---

### Step 2: Construct the covariance matrix
The covariance matrix answers:
- Which genes vary together?
- How strongly do they vary together?


   Gene1  Gene2
```

Gene1    4      4
Gene2    4      4

```

This indicates strong co-variation between genes.

---

### Step 3: Eigen decomposition
From the covariance matrix, PCA computes:
- **Eigenvectors** → directions (new axes)
- **Eigenvalues** → amount of variance explained

Important:
- Each eigenvector corresponds to one PC
- The eigenvector with the largest eigenvalue becomes PC1

---

## 8. Why eigenvectors are ideal new axes

Eigenvectors have special properties:
- Point in directions of maximum variance
- Are independent of each other
- Are perpendicular (orthogonal)

This ensures:
- No redundancy
- Each PC captures unique information

---

## 9. How PC2 and other PCs are found

After PC1:
- PCA finds the next direction with maximum remaining variance
- This direction must be perpendicular to PC1
- That direction becomes PC2

The same process continues for PC3, PC4, and so on.

---

## 10. What are PCA scores?

PCA scores represent the **coordinates of each sample** on the new axes.

How they are calculated:
- Each sample is projected onto each PC axis
- The projection value is the PCA score

Interpretation:
- Samples with similar scores have similar expression patterns
- Large positive or negative scores indicate strong differences

Example:

| Sample     | PC1  | PC2 |
|------------|------|-----|
| Normal_1   | -6.2 | 1.3 |
| Cancer_1   | 7.1  | -2.4 |

---

## 11. What are PCA loadings?

PCA loadings show **how much each gene contributes** to a PC.

- High loading → gene strongly influences that PC
- Loadings help identify genes driving group separation

---

## 12. Simple numeric example

Sample points:
```

(2, 3), (3, 4), (4, 5)

````

These points lie along a diagonal.

- Best direction ≈ (1, 1)
- Normalized direction ≈ (0.707, 0.707)

This direction becomes PC1.

Each sample is projected onto this direction to obtain PC1 scores.

---

## 13. PCA plots explained

### Axes
- X-axis (PC1): maximum variation
- Y-axis (PC2): second highest variation
- Percentages show variance explained

### Points
- Each point represents one sample
- Distance between points reflects similarity

### Colors and ellipses
- Colors indicate groups (used only for interpretation)
- Ellipses show group spread and variability

---

## 14. Interpretation of PCA results

### Clear group separation
- Strong biological signal
- Meaningful gene expression differences

### Overlapping groups
- Biological heterogeneity
- Early disease stages or noise

### Outliers
- Potential technical issues
- Extreme biological samples
- Should be checked in QC

---

## 15. Importance of PCA before machine learning

PCA helps to:
- Validate data quality
- Check sample separability
- Detect batch effects
- Build confidence before ML modeling

Clear PCA separation often leads to better ML performance.

---

## 16. What `prcomp()` does in R

```r
prcomp(data)
````

Automatically:

1. Centers the data
2. Computes the covariance matrix
3. Finds eigenvectors and eigenvalues
4. Orders PCs by variance explained
5. Calculates PCA scores

---

## 17. Key idea 

> PCA finds new axes by rotating the coordinate system so that the first axis captures the maximum variance, and each subsequent axis captures the next highest variance while remaining perpendicular to previous axes.

---

## 18. One-line  summary

PCA reduces high-dimensional data by finding new orthogonal axes that capture maximum variance, using eigenvectors of the covariance matrix.

```
```

