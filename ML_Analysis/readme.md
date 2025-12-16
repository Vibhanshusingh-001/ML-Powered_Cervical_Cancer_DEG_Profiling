#  Data preprocessing

## Step 1: Load Required Libraries (Packages)

This step loads all R packages required for data handling, preprocessing, analysis, and visualization.



## Step 2: Set Up Your Workspace

Prepare the working environment (e.g., set working directory, clear workspace, set seed if needed).



## Step 3: Load the Data

Import the raw dataset into R for further preprocessing and analysis.



## Step 4: Clean and Explore the Data (Preprocessing & EDA)

This block performs **data cleaning, transformation, and exploratory data analysis (EDA)**.
EDA (Exploratory Data Analysis) involves inspecting the data to identify inconsistencies, missing values, and structural issues before modeling.



### 4.1 Check Basic Data Structure

```r
str(df)  # Displays data types (numeric, character, factor, etc.)
dim(df)  # Shows dimensions: rows × columns (e.g., genes × samples)
```



### 4.2 Remove Missing Values

```r
df <- df[complete.cases(df), ]  # Removes rows containing any NA values
head(df)
dim(df)  # Re-check dimensions after cleaning
```

**Why?**
Missing values can distort statistical analysis and machine learning models.



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



### 4.7 Convert Class Labels to Factors

```r
df_t[1] <- factor(df_t$Symbols)
str(df_t)

df <- df_t  # Final cleaned dataset
```

**Why?**

* Factors are required for classification models in R
* Ensures R treats labels as categorical, not text



### 4.8 Final Inspection

```r
str(df)
df <- data.frame(df)
head(df)
```


<img width="480" height="480" alt="box_and_whisker_plots" src="https://github.com/user-attachments/assets/409f5c65-ae37-4e66-b8a1-72f2bee6d258" />


<img width="480" height="480" alt="data" src="https://github.com/user-attachments/assets/b981e9cf-cb0d-4102-bc9e-cdc815f18f6e" />








#  Building & Comparing ML Models



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



## Cross-Validation Setup

```r
control <- trainControl(method="cv", number=8)
metric <- "Accuracy"
```

### What this does

* `trainControl()` defines **how models are trained and validated**
* `method="cv"` → **k-fold cross-validation**
* `number=8` → data split into **8 folds**

 **Why cross-validation?**

* Reduces overfitting
* Gives more reliable performance estimates

### Metric

```r
metric <- "Accuracy"
```

This tells `caret` to **optimize Accuracy** when selecting the best model.


## Inspecting the Dataset

```r
head(df_t)
```

Displays the **first 6 rows** of `df_t` to:

* Check column names
* Verify data types
* Ensure response variable exists

*(Likely `Symbols` is your class label)*



## Storage for Model Results

```r
model_accuracies <- list()
```

Creates an **empty list** to store accuracy values for:

* kNN
* Random Forest
* SVM
* XGBoost (later)

This allows **easy comparison of models**.



# 6. Function to Plot & Save Confusion Matrix

```r
plot_save_cm <- function(cm_table, model_name) {
```

This function:

* Takes a **confusion matrix table**
* Produces a **publication-quality heatmap**
* Saves it as a PNG file



### Step-by-step inside the function

#### Convert table to data frame

```r
table <- data.frame(cm_table)
```

Caret confusion matrices are tables → converted to data frame for plotting.



#### Mark correct vs incorrect predictions

```r
mutate(goodbad = ifelse(table$Prediction == table$Reference, "high", "low"))
```

* **Correct prediction** → `"high"`
* **Incorrect prediction** → `"low"`

Used later for color coding.



#### Compute proportions

```r
group_by(Reference) %>%
mutate(prop = Freq/sum(Freq))
```

Calculates **class-wise proportions**, useful for imbalanced datasets.



#### Plot confusion matrix

```r
ggplot(...) +
geom_tile() +
geom_text(aes(label = Freq)) +
scale_fill_manual(...)
```

* `geom_tile()` → heatmap cells
* `geom_text()` → numbers in cells
* Green (`#009194`) → correct
* Orange (`#FF9966`) → incorrect



#### Save plot

```r
ggsave(paste0("confusion_matrix_", model_name, ".png"))
```

Each model gets its own file:

```
confusion_matrix_knn.png
```



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

```r
png(...)
print(p)
dev.off()
```

Saves the learning curve as an image.



## k-Nearest Neighbors Model (Model 1)

###  Set Random Seed

```r
set.seed(7)
```

Ensures **reproducibility**:

* Same CV splits
* Same results every run



### Train kNN Model

```r
fit.knn <- train(Symbols~., 
                 data=train, 
                 method="knn", 
                 metric=metric, 
                 trControl=control)
```

### Breakdown

| Component           | Meaning                                        |
| ------------------- | ---------------------------------------------- |
| `Symbols ~ .`       | Predict `Symbols` using **all other features** |
| `data=train`        | Training dataset                               |
| `method="knn"`      | k-Nearest Neighbor algorithm                   |
| `metric="Accuracy"` | Optimize accuracy                              |
| `trControl=control` | 8-fold CV                                      |

 `caret` automatically:

* Tunes `k`
* Chooses best `k` using CV



### Feature Importance

```r
varImp(fit.knn)
```

* Computes **importance scores** based on how features affect prediction
* For kNN, importance is **distance-based**



### Plot feature importance

```r
plot(varImp(fit.knn), top = 30)
```

Shows **top 30 features** contributing most to classification.

 Useful for:

* Biomarker discovery
* Feature selection
* Biological interpretation



### Combine Plots

```r
plot_grid(p1, p2)
```

Uses `cowplot` to place:

* Feature importance
* Model performance
  side by side.



###  Save Feature Importance Plot

```r
png("feature_importance_knn.png")
plot(varImp(fit.knn), top = 30)
dev.off()
```



## Prediction on Test Data

```r
pred.knn <- predict(fit.knn, newdata = test)
```

* Applies trained kNN model
* Generates **class predictions** for unseen data



## Model Evaluation

```r
cm_knn <- confusionMatrix(pred.knn, test$Symbols, positive = "Cancer")
```

### What this computes

* Confusion matrix
* Accuracy
* Sensitivity
* Specificity
* Precision
* F1-score

 `positive = "Cancer"`
Treats **Cancer** as the positive class (important for biomedical tasks).



### Store accuracy

```r
model_accuracies$knn <- cm_knn$overall["Accuracy"]
```

Now you can later compare:

```r
model_accuracies
```



## Plot & Save Evaluation Outputs

### Confusion Matrix

```r
plot_save_cm(c1, "knn")
```

Creates:

```
confusion_matrix_knn.png
```



### Learning Curve

```r
plot_save_learning_curve(fit.knn, "knn")
```

Creates:

```
learning_curve_knn.png
```


