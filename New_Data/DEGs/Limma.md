


# Differential Gene Expression Analysis using **limma**


---

## 1. Objective

To identify **Differentially Expressed Genes (DEGs)** between:
- **Normal**
- **Cervical cancer**

using the **limma** package, which combines:
- Linear models
- Empirical Bayes statistics

---

## 2. Expression Data Structure

- Rows → Genes  
- Columns → Samples  

Example:

| Gene | Sample1 | Sample2 | ... | SampleN |
|-----|--------|--------|-----|--------|
| Gene1 | 5.2 | 5.4 | ... | 7.1 |
| Gene2 | 3.8 | 4.0 | ... | 6.2 |

---

## 3. Design Matrix (Experimental Design)

### Purpose
Encodes sample group information numerically so it can be used in linear regression.

### Example Groups
- 10 Normal samples
- 10 Cervical cancer samples

### Conceptual Design Matrix

| Sample Type | Normal | Cervical_cancer |
|-----------|--------|----------------|
| Normal | 1 | 0 |
| Cancer | 0 | 1 |

### Why Needed
- Tells limma which samples belong to which condition
- Enables estimation of group-wise mean expression

---

## 4. Linear Model Fitting (`lmFit`)

### Model (per gene)

Each gene is modeled independently:

Expression = Design_matrix × Coefficients + Error

Where:
- Coefficients represent mean expression per condition
- Error captures biological and technical noise

### What `lmFit()` Does
- Fits a linear model for **each gene**
- Estimates:
  - Mean expression per group
  - Residual variance per gene

---

## 5. Contrasts (Biological Comparisons)

### Purpose
Defines **what comparison to test**.

### Example Contrast
Cervical_cancer − Normal

This produces:
- logFC (log2 fold change)
- Positive logFC → upregulated in cancer
- Negative logFC → downregulated in cancer

### Why Use Contrasts
- Separates model fitting from hypothesis testing
- Allows flexible reuse of the same model

---

## 6. The Variance Problem in Gene Expression

- Each gene has its own variance
- Small sample sizes cause noisy variance estimates
- Classical t-tests become unstable and unreliable

---

## 7. Empirical Bayes Moderation (`eBayes`)

### Core Idea
Borrow information **across all genes** to stabilize variance estimates.

### How It Works
- Combines:
  - Gene-specific variance
  - Global variance estimated from all genes
- Produces a **shrunken (moderated) variance**

### Benefits
- Increased statistical power
- Reduced false positives
- Better performance with small sample sizes

---

## 8. Moderated t-Statistics

### Classical t-statistic
Uses gene-specific variance only → unstable

### Moderated t-statistic (limma)
Uses shrunken variance → stable and robust

### Result
- More reliable p-values
- Better DEG detection

---

## 9. Statistical Outputs from limma

For each gene, limma reports:

| Column | Meaning |
|------|--------|
| `logFC` | Log2 fold change |
| `AveExpr` | Average expression |
| `t` | Moderated t-statistic |
| `P.Value` | Raw p-value |
| `adj.P.Val` | FDR-adjusted p-value |
| `B` | Log-odds of being differentially expressed |

---

## 10. B-statistic (Log-Odds of DE)

### Definition (plain text)

B = log( P(DE) / (1 − P(DE)) )

Where:
- P(DE) = probability that the gene is differentially expressed

### Interpretation
- B = 0 → 50% probability of DE
- B > 0 → higher probability of DE
- Larger B → stronger evidence

---

## 11. Multiple Testing Correction

### Why Needed
- Thousands of genes tested simultaneously
- Raw p-values inflate false positives

### Benjamini–Hochberg (BH) Method
Controls the **False Discovery Rate (FDR)**

Plain-text formula:

q = (p × total_number_of_tests) / rank_of_p_value

---

## 12. Extracting Differentially Expressed Genes (`topTable`)

### Purpose
- Extracts DEG results
- Applies FDR correction
- Ranks genes by significance or fold change

### Typical DEG Filtering Criteria

```r
abs(logFC) > 1 & adj.P.Val < 0.05
````

Meaning:

* At least 2-fold change
* Statistically significant after FDR correction

---

## 13. Exporting Results

* DEGs are sorted (e.g., by logFC)
* Results are saved as CSV files
* Used for:

  * Pathway analysis
  * Visualization
  * Reporting

---

## 14. Why limma Works So Well

| Aspect              | Classical Tests | limma                  |
| ------------------- | --------------- | ---------------------- |
| Variance estimation | Gene-wise only  | Shrinkage across genes |
| Small sample sizes  | Poor            | Excellent              |
| Statistical power   | Lower           | Higher                 |
| False positives     | More            | Fewer                  |

---



---

## 15.  Summary

> limma fits gene-wise linear models and applies empirical Bayes variance shrinkage to compute moderated t-statistics, enabling robust and powerful identification of differentially expressed genes even with small sample sizes.


