


# Data Preprocessing

## Step 1: Load Required Libraries (Packages)

This step loads all R packages required for data handling, preprocessing, analysis, and visualization.

---

## Step 2: Set Up Your Workspace

Prepare the working environment (e.g., set working directory, clear workspace, set seed if needed).

---

## Step 3: Load the Data

Import the raw dataset into R for further preprocessing and analysis.

---

## Step 4: Clean and Explore the Data (Preprocessing & EDA)

This block performs **data cleaning, transformation, and exploratory data analysis (EDA)**.  
EDA (Exploratory Data Analysis) involves inspecting the data to identify inconsistencies, missing values, and structural issues before modeling.

---

### 4.1 Check Basic Data Structure

```r
str(df)  # Displays data types (numeric, character, factor, etc.)
dim(df)  # Shows dimensions: rows × columns (e.g., genes × samples)
````

---

### 4.2 Remove Missing Values

```r
df <- df[complete.cases(df), ]  # Removes rows containing any NA values
head(df)
dim(df)  # Re-check dimensions after cleaning
```

**Why?**
Missing values can distort statistical analysis and machine learning models.

---

### 4.3 Average Duplicate Genes (If Present)

```r
x <- df  # Create a copy of the dataset

x <- do.call(
  rbind,
  lapply(
    lapply(split(x, x$DEGs), `[`, 2:ncol(x)),
    colMeans
  )
)

dim(x)
```

**What this does:**

* Splits the dataset by gene names in the `DEGs` column
* Computes the column-wise mean for duplicated genes
* Recombines them into a single table

**Why this is needed:**

* Microarray datasets often contain multiple probes per gene
* Averaging ensures **one row per gene**

> **Note:** After this step, gene names become row names instead of a column.

---

### 4.4 Convert Row Names to a Column

```r
x <- data.frame(x)
x <- tibble::rownames_to_column(x, var = "Symbols")

head(x)
dim(x)

df <- x  # Update the main dataset
```

**Purpose:**
Makes gene identifiers explicit by converting row names into a visible column (`Symbols`).

---

### 4.5 Transpose the Dataset (Genes → Columns, Samples → Rows)

```r
library(sjmisc)

df_t <- rotate_df(df, cn = TRUE)  # Transpose the data
Symbols <- colnames(df[-1])       # Extract original column names

df_t <- cbind(Symbols, df_t)      # Add them as a new column
write.csv(df_t, "transposed_table.csv", row.names = FALSE)

df_t <- read.csv("transposed_table.csv", header = TRUE)

dim(df_t)
df_t[1]
```

**What this does:**

* Original format:

  * Rows = genes
  * Columns = samples
* New format:

  * Rows = samples
  * Columns = genes

Saving and reloading helps avoid formatting inconsistencies after transposition.

---

### 4.6 Clean Sample Labels

```r
df_t$Symbols <- sub(
  ".*_(.*?)\\.CEL",
  "\\1",
  df_t$Symbols
)

df_t[1]
```

**What this does:**

* Uses regular expressions to extract clean class labels
* Example:
  `"GSM1234_NORMAL.CEL"` → `"NORMAL"`

This simplifies downstream classification tasks (e.g., Normal vs Cancer).

---

### 4.7 Convert Class Labels to Factors

```r
df_t[1] <- factor(df_t$Symbols)
str(df_t)

df <- df_t  # Final cleaned dataset
```

**Why?**

* Factors are required for classification models in R
* Ensures R treats labels as categorical, not text

---

### 4.8 Final Inspection

```r
str(df)
df <- data.frame(df)
head(df)
```

---

### Exploratory Plots

<p align="center">
  <img src="https://github.com/user-attachments/assets/409f5c65-ae37-4e66-b8a1-72f2bee6d258" width="450" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b981e9cf-cb0d-4102-bc9e-cdc815f18f6e" width="450" />
</p>

---

# Building & Comparing ML Models

## Importing Required Libraries

```r
library("readxl")  
library("tidyverse")
library("caret")
library("glmnet")
library(kernlab)
library(randomForest)
library(e1071)
library(rpart)
library(xgboost)
library(pROC)
library(cowplot)
library(ggplot2)
library(dplyr)
library(sjmisc)
library(tibble)
```

### Purpose of each library

| Package        | Why it’s used                                           |
| -------------- | ------------------------------------------------------- |
| `readxl`       | Read Excel files                                        |
| `tidyverse`    | Data manipulation (`dplyr`) + visualization (`ggplot2`) |
| `caret`        | **Core ML framework** (training, CV, evaluation)        |
| `glmnet`       | Regularized regression (LASSO, Ridge, Elastic Net)      |
| `kernlab`      | Kernel-based models (used internally by caret)          |
| `randomForest` | Random Forest algorithm                                 |
| `e1071`        | SVM and other ML utilities                              |
| `rpart`        | Decision Trees                                          |
| `xgboost`      | Gradient boosting models                                |
| `pROC`         | ROC curves and AUC                                      |
| `cowplot`      | Combine multiple ggplots                                |
| `sjmisc`       | Helper functions                                        |
| `tibble`       | Modern data frames                                      |

Even though you train **kNN here**, these libraries allow **multiple models** in the same script.

---

## Cross-Validation Setup

```r
control <- trainControl(method = "cv", number = 8)
metric <- "Accuracy"
```

### What this does

* `trainControl()` defines **how models are trained and validated**
* `method = "cv"` → **k-fold cross-validation**
* `number = 8` → data split into **8 folds**

**Why cross-validation?**

* Reduces overfitting
* Gives more reliable performance estimates

---

## Inspecting the Dataset

```r
head(df_t)
```

Displays the **first 6 rows** of `df_t` to:

* Check column names
* Verify data types
* Ensure response variable exists

*(Likely `Symbols` is your class label)*

---

## Storage for Model Results

```r
model_accuracies <- list()
```

Creates an **empty list** to store accuracy values for:

* kNN
* Random Forest
* SVM
* XGBoost (later)

---

## Function to Plot & Save Confusion Matrix

```r
plot_save_cm <- function(cm_table, model_name) {
```

This function:

* Takes a **confusion matrix table**
* Produces a **publication-quality heatmap**
* Saves it as a PNG file

---

## Function to Plot Learning Curve

```r
plot_save_learning_curve <- function(fit_model, model_name)
```

This function:

* Uses `caret::plot()`
* Shows **model performance vs training set size**
* Helps diagnose:

  * Underfitting
  * Overfitting

---

## k-Nearest Neighbors Model (Model 1)

### Set Random Seed

```r
set.seed(7)
```

Ensures **reproducibility**.

---

### Train kNN Model

```r
fit.knn <- train(
  Symbols ~ ., 
  data = train, 
  method = "knn", 
  metric = metric, 
  trControl = control
)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/e1172a9c-4323-40ed-b9f1-1fe9b17d48bb" width="650" />
</p>

---

### Feature Importance

```r
varImp(fit.knn)
```

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/e05af6f3-a30a-46c2-b81f-24dbcec06937" width="650" />
</p>

---

## Model Evaluation

```r
cm_knn <- confusionMatrix(pred.knn, test$Symbols, positive = "Cancer")
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/4506f1ff-a22e-4eba-8909-1511fcd2fde7" width="600" />
</p>

---

## Evaluation Outputs

### Confusion Matrix

<p align="center">
  <img src="https://github.com/user-attachments/assets/db5d61f2-8a3d-4488-a809-8cb0f1ab4450" width="700" />
</p>

---

### Learning Curve

<p align="center">
  <img src="https://github.com/user-attachments/assets/06bd2310-a4a3-4363-b7e2-de6d69059df8" width="650" />
</p>
```


