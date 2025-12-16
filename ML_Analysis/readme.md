


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


# Machine Learning Models: Training, Evaluation, and Comparison



# Model 2: Support Vector Machine (SVM – Radial Kernel)

## Model Training

```r
set.seed(101)
fit.svm <- train(
  Symbols ~ ., 
  data = train, 
  method = "svmRadial", 
  metric = metric, 
  trControl = control
)
fit.svm
````
<img width="826" height="422" alt="fit_svm" src="https://github.com/user-attachments/assets/2230a39e-2062-4fb3-9204-1ec35dfc90af" />

### Explanation

* `set.seed(101)` ensures reproducibility
* `svmRadial` uses a **Radial Basis Function (RBF) kernel**
* SVM finds an **optimal separating hyperplane** in a high-dimensional space
* The radial kernel allows **non-linear decision boundaries**
* Hyperparameters (`C`, `sigma`) are tuned automatically by `caret`

---

## Feature Importance (SVM)

```r
varImp(fit.svm)
```

* Measures how much each feature contributes to classification
* For SVM, importance is estimated indirectly via model sensitivity

```r
plot(varImp(fit.svm), top = 30)
```

###  Feature Importance Plot (SVM)

> <img width="800" height="600" alt="feature_importance_svm" src="https://github.com/user-attachments/assets/0ee2234b-8690-4eed-8cdf-9a4048ea31c7" />


---

## Model Prediction

```r
pred.svm <- predict(fit.svm, newdata = test)
```

* Applies the trained SVM model to unseen test data
* Outputs predicted class labels

---

## Model Evaluation

```r
cm_svm <- confusionMatrix(pred.svm, test$Symbols, positive = "Cancer")
print(cm_svm)
```
<img width="655" height="547" alt="Evaluation_SVM" src="https://github.com/user-attachments/assets/b946d177-e2f4-45c0-aea2-0a84e04540ae" />

### Metrics Computed

* Accuracy
* Sensitivity (Recall)
* Specificity
* Precision
* F1-score

```r
model_accuracies$svm <- cm_svm$overall["Accuracy"]
```

---

## Confusion Matrix (SVM)

> <img width="1800" height="1200" alt="confusion_matrix_svm" src="https://github.com/user-attachments/assets/0cfe4d37-06bf-40a2-85d6-05ce35c92d40" />


---

## Learning Curve (SVM)

> <img width="800" height="600" alt="learning_curve_svm" src="https://github.com/user-attachments/assets/7800aee0-8f3e-44d8-a9c8-cad37594d9c2" />

---

# Model 3: Random Forest

## Model Training

```r
set.seed(123)
fit.rf <- train(
  Symbols ~ ., 
  data = train, 
  method = "rf",
  metric = metric,
  trControl = control
)
fit.rf
```
<img width="767" height="392" alt="Fit_RF" src="https://github.com/user-attachments/assets/5a4f814b-f7ed-4543-bc25-8d0afe8b74cc" />

### Explanation

* Random Forest is an **ensemble of decision trees**
* Each tree is trained on a bootstrap sample
* Feature randomness reduces overfitting
* Final prediction is based on **majority voting**

---

## Feature Importance (Random Forest)

```r
varImp(fit.rf)
```

* RF computes importance using:

  * Mean Decrease in Accuracy
  * Mean Decrease in Gini

```r
plot(varImp(fit.rf), top = 30)
```

###  Feature Importance Plot (RF)

> <img width="800" height="600" alt="feature_importance_rf" src="https://github.com/user-attachments/assets/ce618fc7-dd8f-4331-a6de-ddefc2e29c7b" />


---

## Prediction and Evaluation

```r
pred.rf <- predict(fit.rf, newdata = test)
cm_rf <- confusionMatrix(pred.rf, test$Symbols, positive = "Cancer")
model_accuracies$rf <- cm_rf$overall["Accuracy"]
```
<img width="557" height="561" alt="Evaluation_rf" src="https://github.com/user-attachments/assets/41f58aca-117d-4e17-a8ba-49ac49174bff" />

---

## Confusion Matrix (RF)

> <img width="1800" height="1200" alt="confusion_matrix_rf" src="https://github.com/user-attachments/assets/59504fc1-dbc0-47ae-a9e1-1e9e45d3daab" />


---

## Learning Curve (RF)

> <img width="800" height="600" alt="learning_curve_rf" src="https://github.com/user-attachments/assets/125345c6-665c-4876-9dc4-aa953775b773" />


---

# Model 4: Logistic Regression

## Model Training

```r
set.seed(101)
fit.lr <- train(
  Symbols ~ ., 
  data = train, 
  method = "glm", 
  family = "binomial",
  metric = metric,
  trControl = control
)
fit.lr
```
<img width="557" height="312" alt="fit_lr" src="https://github.com/user-attachments/assets/cb4d5993-4149-477d-ac91-b30229f8adcd" />
<img width="415" height="256" alt="results_lr" src="https://github.com/user-attachments/assets/911befea-389f-4576-a749-6c88c97a0a46" />

### Explanation

* Logistic regression models **probability of class membership**
* Uses a **logit (sigmoid) function**
* Coefficients represent log-odds contribution of each feature
* Suitable as a **baseline interpretable model**

---

## Feature Importance (Logistic Regression)

```r
varImp(fit.lr)
plot(varImp(fit.lr), top = 10)
```

### 📌 Feature Importance Plot (LR)

> <img width="800" height="600" alt="feature_importance_lr" src="https://github.com/user-attachments/assets/e4bd6497-428a-49d7-bf9e-b33ca511d2c3" />


---

## Prediction and Evaluation

```r
pred.lr <- predict(fit.lr, newdata = test)
cm_lr <- confusionMatrix(pred.lr, test$Symbols, positive = "Cancer")
model_accuracies$lr <- cm_lr$overall["Accuracy"]
```
<img width="602" height="558" alt="Evaluation_LR" src="https://github.com/user-attachments/assets/0a9e79db-524f-4eab-951b-cc0520fcae37" />

---

## Confusion Matrix (LR)

> <img width="1800" height="1200" alt="confusion_matrix_lr" src="https://github.com/user-attachments/assets/6ae55376-6d8e-4573-a480-3026b5ed8194" />


---

## Custom Learning Curve (Logistic Regression)

### Purpose

* `caret` does not provide learning curves for `glm`
* Custom function evaluates performance across increasing training sizes

```r
results_lr <- custom_learning_curve(train, test, Symbols ~ .)
```

###  Learning Curve Plot (LR)

> *(Paste learning curve image here)*

---

# Model 5: Decision Tree

## Model Training

```r
set.seed(101)
fit.dt <- train(
  Symbols ~ ., 
  data = train,
  method = "rpart",
  metric = "Accuracy",
  trControl = control,
  tuneGrid = expand.grid(cp = seq(0.0001, 0.1, 0.005)),
  control = rpart.control(minsplit = 2, minbucket = 1)
)
fit.dt
```
<img width="852" height="727" alt="fit_DT" src="https://github.com/user-attachments/assets/9433b08c-b0d1-4d92-bb6b-614a16faed90" />

### Explanation

* Decision trees split data using **if–else rules**
* `cp` controls tree complexity (pruning)
* Lower `minsplit` allows deeper trees

---

## Feature Importance (Decision Tree)

```r
varImp(fit.dt)
plot(varImp(fit.dt), top = 30)
```

### 📌 Feature Importance Plot (DT)

> <img width="800" height="600" alt="feature_importance_DT" src="https://github.com/user-attachments/assets/0cf21dbe-efc6-47ca-a2ce-85829206df43" />


---

## Prediction and Evaluation

```r
pred.dt <- predict(fit.dt, newdata = test)
cm_dt <- confusionMatrix(pred.dt, test$Symbols, positive = "Cancer")
model_accuracies$dt <- cm_dt$overall["Accuracy"]
```
<img width="541" height="548" alt="Evaluation_dt" src="https://github.com/user-attachments/assets/b69dea08-0372-4ced-8c2d-b919a2334bd1" />

---

## Confusion Matrix (DT)

> <img width="1800" height="1200" alt="confusion_matrix_dt" src="https://github.com/user-attachments/assets/42d65a9d-c4dc-4f27-8eec-6e91c2f21f54" />


---

## Learning Curve (DT)

> <img width="800" height="600" alt="learning_curve_dt" src="https://github.com/user-attachments/assets/def82392-f7d8-483f-89c0-b403ca7924bb" />


---

# Model 6: XGBoost

## Model Training

```r
set.seed(101)
fit.xgb <- train(
  Symbols ~ ., 
  data = train, 
  method = "xgbTree",
  metric = metric,
  trControl = control
)
fit.xgb
```
<p align="center">
<img width="50" height="500" alt="fit_xgb" src="https://github.com/user-attachments/assets/ac3d76b1-b031-4eed-89c0-9dc8d5b19fc1" />
<img width="500" height="500" alt="XGB_fit" src="https://github.com/user-attachments/assets/b2dc2654-f602-46ca-a508-e7adf474d18c" />
</p>
### Explanation

* XGBoost is a **gradient boosting algorithm**
* Sequentially corrects errors of previous trees
* Strong performance on high-dimensional omics data
* Includes regularization to prevent overfitting

---

## Feature Importance (XGBoost)

```r
varImp(fit.xgb)
plot(varImp(fit.xgb), top = 30)
```

###  Feature Importance Plot (XGB)
<p align="center">
> <img width="300" height="200" alt="feature_importance_XGB" src="https://github.com/user-attachments/assets/4f299fbc-5742-460d-a7a2-91b4bb12bbe5" />
</p>

---

## Prediction and Evaluation

```r
pred.xgb <- predict(fit.xgb, newdata = test)
cm_xgb <- confusionMatrix(pred.xgb, test$Symbols, positive = "Cancer")
model_accuracies$xgb <- cm_xgb$overall["Accuracy"]
```
<p align="center">
<img width="587" height="547" alt="Evaluation_xgb" src="https://github.com/user-attachments/assets/048a0787-ef58-48df-8e4c-a4641682ae48" />
</p>
---

## Confusion Matrix (XGB)
<p align="center">
> <img width="500" height="700" alt="confusion_matrix_xgb" src="https://github.com/user-attachments/assets/440d8ae3-95dc-4a1a-834d-024ad0ca191b" />
</p>

---

## Learning Curve (XGB)
<p align="center">
> <img width="500" height="700" alt="learning_curve_xgb" src="https://github.com/user-attachments/assets/e10a1a73-33a9-4012-900c-021907e08349" />
</p>

---

# Model Comparison

## Accuracy Table

```r
acc_df <- data.frame(
  Model = names(model_accuracies),
  Accuracy = unlist(model_accuracies)
)
print(acc_df)
```

---

## Model Accuracy Comparison Plot

```r
ggsave(
  "model_accuracy_comparison.png", 
  plot = p, 
  width = 8, 
  height = 6, 
  dpi = 300
)
```

###  Accuracy Comparison Bar Plot



<p align="center">
  <img width="700" height="500" alt="model_accuracy_comparison" src="https://github.com/user-attachments/assets/296a73b9-156b-4527-a1b5-c6afb9f73378" />
</p>

---


