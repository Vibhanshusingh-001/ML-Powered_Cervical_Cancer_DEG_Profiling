
# Machine Learning (ML)

## Model Development for Gene Expression Analysis

This workflow builds machine learning models for classifying **Normal vs Cancer** samples using gene expression data. It covers preprocessing, visualization, training multiple ML models, evaluation, and comparison.

---

## Step 1: Data Preparation

1. **Files used**:

   * **Post-normalized annotated file** (gene expression matrix after preprocessing).
   * **DEG annotated file** (list of differentially expressed genes).

2. **Filtering and Cleaning**:

   * Copy gene names from the DEG file.
   * Map these genes in the post-normalized file.
   * Remove:

     * **NA values** (missing values).
     * **Duplicate entries** (for genes appearing multiple times).

3. **Creating input dataset**:

   * Keep only **DEGs** and their expression values.
   * Save the dataset as **Short\_table.csv**.
   * Place it into a new folder named **ML\_Analysis** for further work.

---

## Step 2: Loading Required Packages

* Packages are installed and loaded to support:

  * **Data handling** → `readxl`, `tidyverse`, `dplyr`, `sjmisc`, `tibble`.
  * **Machine Learning** → `caret`, `glmnet`, `randomForest`, `xgboost`, `e1071`, `rpart`, `kernlab`.
  * **Visualization** → `ggplot2`, `cowplot`, `grid`.
  * **Evaluation** → `pROC`.

This ensures the environment is ready for ML model development.

---

## Step 3: Loading Data

* The file **Short\_table.csv** is loaded.
* The dataset is examined by:

  * **Structure (`str`)** → to see data types.
  * **Dimensions (`dim`)** → to check number of samples and features.
  * **First few rows (`head`)** → to view sample data.

---

## Step 4: Data Pre-processing & Exploratory Data Analysis (EDA)

### 4.1 Handling Missing Values

* Rows with **NA values** are removed.

### 4.2 Handling Duplicate Genes

* For duplicate gene names, the **mean expression value** is calculated and used.

### 4.3 Restructuring Dataset

* Gene names are re-inserted as a column called **Symbols**.
* The dataset is **transposed** so that:

  * Rows represent samples.
  * Columns represent gene expression values.

### 4.4 Cleaning Sample Labels

* Sample names (like `GSM1234_NORMAL.CEL`) are cleaned to extract the condition (Normal / Cancer).
* These labels are converted into categorical factors.

### 4.5 Final Dataset Check

* The dataset now has:

  * **First column** → Condition (Normal or Cancer).
  * **Remaining columns** → Expression values of selected DEGs.

---

## Step 5: Data Visualization

1. **Box and Whisker Plots**

   * Created for each gene to see the spread and variability in expression.

2. **Sample Matrix Plot**

   * A quick check of sample distribution across Normal and Cancer groups.

These visualizations help confirm the dataset quality and structure before ML training.

---

## Step 6: Data Splitting

* The dataset is divided into:

  * **Training set (60%)** → used to build ML models.
  * **Testing set (40%)** → used to evaluate performance.

* **Cross-validation setup**:

  * **10-fold cross-validation** is used.
  * Ensures robust training and prevents overfitting.
  * Accuracy is chosen as the evaluation metric.

---

## Step 7: Building Machine Learning Models

Multiple models are trained on the same dataset for comparison.

### 7.1 k-Nearest Neighbors (kNN)

* Model predicts class by checking the majority label of nearest neighbors.
* Outputs:

  * Feature importance (which genes contribute most).
  * Learning curve (training vs testing performance).
  * Confusion matrix (classification performance).

### 7.2 Support Vector Machine (SVM)

* Uses **radial basis function kernel** for separating data.
* Outputs: feature importance, learning curve, confusion matrix.

### 7.3 Random Forest

* Builds multiple decision trees and combines them (ensemble learning).
* Identifies the **most important genes** for classification.
* Outputs: feature importance plots, confusion matrix, learning curve.

### 7.4 Logistic Regression

* Fits a logistic model to separate Normal vs Cancer groups.
* Outputs: important genes, confusion matrix, learning curve.

### 7.5 Decision Tree

* Builds a single decision tree classifier.
* Easy to interpret but less robust compared to Random Forest.
* Outputs: feature importance, confusion matrix, learning curve.

### 7.6 XGBoost (Extreme Gradient Boosting)

* Uses boosting technique to combine weak learners into a strong model.
* Generally provides high accuracy.
* Outputs: feature importance, confusion matrix, learning curve.

---

## Step 8: Model Evaluation

For each model:

* **Confusion Matrix** → Shows classification results (True Positive, True Negative, False Positive, False Negative).
* **Feature Importance** → Highlights genes most influential in classification.
* **Learning Curves** → Shows how accuracy changes with training size or parameter tuning.
* **Accuracy Value** → Overall model performance metric.

All results are saved as images (`.png`) for records.

---

## Step 9: Model Comparison

1. Collect accuracy values of all models into a single table.
2. Create a **barplot** comparing accuracies.

   * Models listed on one axis.
   * Accuracy values displayed as bars.
   * Labels show numerical accuracy values.
3. This helps in identifying the **best performing model** for cancer classification.

---

## Step 10: Output

* Saved plots:

  * Confusion matrices.
  * Learning curves.
  * Feature importance.
  * Model comparison barplot.
* Printed results:

  * Table of accuracies for all models.

---

## Workflow Summary

1. **Input data** → Post-normalized expression + DEG list.
2. **Filtering** → Keep DEGs, remove NA and duplicates.
3. **Preprocessing** → Restructure dataset, clean labels.
4. **EDA** → Visualize distributions and sample groups.
5. **Splitting** → 60% training, 40% testing.
6. **Modeling** → Train kNN, SVM, RF, Logistic Regression, Decision Tree, XGBoost.
7. **Evaluation** → Confusion matrix, feature importance, learning curve, accuracy.
8. **Comparison** → Side-by-side accuracy comparison.
9. **Best Model** → Select the model with highest accuracy for final use.

