


#### Step 1: Load Required Libraries (Packages)

#### Step 2: Set Up Your Workspace

#### Step 3: Load the Data

#### Step 4: Clean and Explore the Data (Preprocessing & EDA)
This big block `{ ... }` checks, cleans, and reshapes the data. EDA means "Exploratory Data Analysis"—just looking around to spot issues.

- **Check the basics**:
  ```r
  str(df)  # Shows data types (e.g., numbers, text)
  dim(df)  # Rows x columns (e.g., 1000 genes x 20 samples)
  ```
  
- **Remove missing values**:
  ```r
  df <- df[complete.cases(df), ]  # Drops rows with any NA (empty cells)
  head(df); dim(df)  # Peek and check size again
  ```
 
- **Average duplicate genes** (if genes repeat):
  ```r
  x <- df  # Copy data
  x <- do.call(rbind, lapply(lapply(split(x, x$DEGs), `[`, 2:ncol(x)), colMeans))  # Groups by gene (DEGs column), averages values across duplicates
  dim(x)
  ```
  - **What it does**: Splits data by gene name (in "DEGs" column), averages the numeric columns for duplicates, then stacks them back.
  - **Why?**: Microarray data often has repeated genes; averaging cleans it to one row per gene.
  - **Simple tip**: Output now has genes as rows (no "DEGs" column—it's now row names).

- **Fix row names** (turn hidden row labels into a visible column):
  ```r
  x <- data.frame(x)
  x <- tibble::rownames_to_column(x, var="Symbols")  # Makes row names a new "Symbols" column
  head(x); dim(x)
  df <- x  # Update main table
  ```

- **Transpose (flip) the table** (genes to columns, samples to rows):
  ```r
  library(sjmisc)
  df_t <- rotate_df(df, cn=T)  # Flips rows/columns; keeps column names
  Symbols <- colnames(df[-1])  # Grabs old column names (now rows)
  df_t <- cbind(Symbols, df_t)  # Adds them as a new column
  write.csv(df_t, "transposed_table.csv", row.names=F)  # Saves flipped version
  df_t <- read.csv("transposed_table.csv", h=T)  # Reloads it
  dim(df_t); df_t[1]  # Check size and first column
  ```
  - **What it does**: Original: Rows=genes, Columns=samples. Now: Rows=samples, Columns=genes. Saves/ reloads to fix any glitches.

- **Clean sample labels** (e.g., from "GSM1234_NORMAL.CEL" to "NORMAL"):
  ```r
  df_t$Symbols <- sub(".*_(.*?)\\.CEL", "\\1", df_t$Symbols)  # Extracts text between last "_" and ".CEL" (e.g., "NORMAL" or "CANCER")
  df_t[1]  # Check first column
  ```
  - **What it does**: Uses regex (pattern matching) to strip file extras, keeping just the label (Normal/Cancer).
  

- **Convert labels to factors** (categorical data):
  ```r
  df_t[1] <- factor(df_t$Symbols)  # Turns text labels into categories R understands
  str(df_t)
  df <- df_t  # Final update
  ```

- **Final peek**:
  ```r
  str(df); df <- data.frame(df); head(df)
  ```
  - Just double-checks the cleaned table.

#### Step 5: Visualize the Data (Simple Plots)
Plots help spot outliers or imbalances (e.g., more Normal samples?).

- **Box-and-Whisker Plots** (for each gene/feature):
  ```r
  png("box_and_whisker_plots.png")  # Saves plot as image
  par(mfrow=c(2,4))  # Arranges 2 rows x 4 columns of mini-plots
  for(i in 2:9) {  # Loops over first 8 genes (columns 2-9)
    boxplot(x[,i], main=names(df)[i], col="blue")  # Box plot per gene; shows spread, median, outliers
  }
  dev.off()  # Closes and saves
  ```
  - **What it does**: Creates side-by-side box plots for the first few genes. Each box shows data spread (whiskers=range, box=most data, line=average).
 <img width="480" height="480" alt="box_and_whisker_plots" src="https://github.com/user-attachments/assets/418f6c66-284d-47c5-bca5-ef81e8148712" />

- **Bar Plot of Sample Labels** (e.g., count of Normal vs. Cancer):
  ```r
  library(ggplot2); library(caret)
  x <- df[,2:ncol(df)]  # Inputs: All gene columns (features)
  y <- df[,1]           # Output: Labels (Normal/Cancer)
  y <- as.factor(y)     # Ensure it's categorical
  plot(y, col="blue")   # Basic bar chart of label counts
  ```
<img width="500" height="450" alt="data" src="https://github.com/user-attachments/assets/b981e9cf-cb0d-4102-bc9e-cdc815f18f6e" />




###  Building & Comparing ML Models



---

#### Step 1: Setup for Training
```r
control <- trainControl(method="cv", number=8)  # 8-fold cross-validation: Split train data into 8 parts, train on 7/test on 1, repeat
metric <- "Accuracy"  # Score models by % correct predictions
head(df_t)  # Peek at data (samples x genes + Symbols label)
```
- **What**: Sets rules for fair model training (CV prevents "memorizing" data). Accuracy = correct guesses / total.
- **Why**: CV makes models reliable; accuracy is simple for classification.
- **Tip**: For imbalanced data (e.g., more Normal), swap metric to "Balanced Accuracy". Add `summaryFunction = twoClassSummary` for ROC/AUC if needed.

#### Step 2: Helper Functions (Reusable Plotting Tools)
These save time—define once, use for all models.

##### Confusion Matrix Plotter
```r
plot_save_cm <- function(cm_table, model_name) {  # cm_table = raw counts, model_name = e.g., "knn"
  library(ggplot2); library(dplyr)
  table <- data.frame(cm_table)  # Turn matrix into table
  plotTable <- table %>%
    mutate(goodbad = ifelse(Prediction == Reference, "high", "low")) %>%  # Color: Green=correct, Orange=wrong
    group_by(Reference) %>%  # Group by true label
    mutate(prop = Freq / sum(Freq))  # % of predictions per true class
  
  p <- ggplot(plotTable, aes(x = Reference, y = Prediction, fill = goodbad, alpha = Freq)) +  # Heatmap-style
    geom_tile() +  # Boxes
    geom_text(aes(label = Freq), vjust = 0.5, fontface = "bold", alpha = 1) +  # Numbers in boxes
    scale_fill_manual(values = c(high = "#009194", low = "#FF9966")) +  # Colors
    theme_bw() + xlim(rev(levels(table$Reference))) +  # Flip x-axis for readability
    ggtitle(paste("Confusion Matrix -", model_name))
  
  ggsave(paste0("confusion_matrix_", model_name, ".png"), plot = p, width = 6, height = 4, dpi = 300)  # Save PNG
}
```
- **What**: Turns a confusion matrix (table of predictions vs. reality) into a pretty heatmap PNG.
- **Why**: Visualizes errors (e.g., false positives = predicted Cancer but Normal). Diagonal = correct.
- **Tip**: Example output: Green boxes on diagonal (high accuracy), orange off-diagonal (mistakes).

##### Learning Curve Plotter
```r
plot_save_learning_curve <- function(fit_model, model_name) {
  p <- plot(fit_model, main = paste("Learning Curve -", model_name))  # Caret's built-in plot: Accuracy vs. model params
  png(paste0("learning_curve_", model_name, ".png"), width = 800, height = 600)
  print(p)
  dev.off()  # Saves PNG
}
```
- **What**: Plots how model performance changes with tweaks (e.g., more neighbors in kNN).
- **Why**: Shows if model improves or overfits (e.g., train accuracy high, test low = bad).
- **Tip**: For Logistic Regression, it uses a custom version later— this is for others.

#### Step 3: Build the 6 Models
Each follows: **Train → Importance Plot → Predict → Evaluate → Save Plots**. Uses `train()` from caret (easy wrapper). Assumes binary labels ("Normal"/"Cancer").

##### 1. k-Nearest Neighbors (kNN) – Simple "Vote by Neighbors"
```r
set.seed(7)  # Random seed for reproducibility
fit.knn <- train(Symbols ~ ., data = train, method = "knn", metric = metric, trControl = control)  # Train: Predict label from all genes
fit.knn  # Shows best k (neighbors) and CV accuracy

# Feature Importance (top genes)
varImp(fit.knn)
p1 <- plot(varImp(fit.knn), top = 30, main = "kNN")  # Bar plot of influential genes
p2 <- plot(fit.knn, main = "kNN")  # Performance plot
plot_grid(p1, p2)  # Side-by-side (needs cowplot)

png("feature_importance_knn.png", width = 800, height = 600)
plot(varImp(fit.knn), top = 30, main = "Feature Importance - kNN")
dev.off()  # Save PNG

# Predict & Evaluate
pred.knn <- predict(fit.knn, newdata = test)
cm_knn <- confusionMatrix(pred.knn, test$Symbols, positive = "Cancer")  # Table + stats (sensitivity, etc.)
print(cm_knn)
model_accuracies$knn <- cm_knn$overall["Accuracy"]  # Store score
plot_save_cm(cm_knn$table, "knn")  # Save CM plot
plot_save_learning_curve(fit.knn, "knn")  # Save curve
```
- **What**: Classifies by averaging "k" similar samples (e.g., k=5: vote of 5 closest).
- **Why**: Fast, intuitive—no assumptions about data shape.
- **Tip**: Good for gene data; tune k via CV. If accuracy low, scale genes first (`preProcess = c("center", "scale")` in train).
<img width="600" height="500" alt="confusion_matrix_knn" src="https://github.com/user-attachments/assets/c85d0273-3630-4a90-95f5-68dce8ffaf5e" />

<img width="400" height="300" alt="feature_importance_knn" src="https://github.com/user-attachments/assets/964657d8-08b2-4a3a-aa64-a8134c0300c6" />
<img width="475" height="200" alt="fit_knn" src="https://github.com/user-attachments/assets/35175ca9-ede6-4925-803d-4b3c5e488861" />
<img width="337" height="257" alt="knn_evaluation" src="https://github.com/user-attachments/assets/33811bed-8c56-4f36-81e3-075909b7537f" />
<img width="400" height="300" alt="learning_curve_knn" src="https://github.com/user-attachments/assets/5f64e514-a5d8-49ed-9556-8f56e6335ba0" />
<img width="367" height="245" alt="VIMP_knn" src="https://github.com/user-attachments/assets/eb7ab35a-23e9-49ee-a663-4b54a260e941" />

##### 2. Support Vector Machine (SVM) – Radial Kernel
Similar structure:
```r
set.seed(101)
fit.svm <- train(Symbols ~ ., data = train, method = "svmRadial", metric = metric, trControl = control)
# ... (importance plot, predict, cm_svm, store accuracy, save plots like above)
```
- **What**: Draws a "boundary" to separate classes; radial = handles curvy patterns.
- **Why**: Robust to outliers; great for high-dimensional gene data.
- **Tip**: `svmRadial` tunes cost/sigma. Add `tuneLength=10` for more tweaks.

##### 3. Random Forest (RF) – Ensemble of Trees
Similar:
```r
set.seed(123)
fit.rf <- train(Symbols ~ ., data = train, method = "rf", metric = metric, trControl = control)
# ... (importance, predict, cm_rf, etc.)
```
- **What**: Builds many decision trees, averages votes.
- **Why**: Handles interactions between genes; built-in importance ranking.
- **Tip**: Slow on big data—add `tuneGrid = expand.grid(mtry = c(2,4,6))` to tune.

##### 4. Logistic Regression (LR) – Simple Linear Classifier
Similar, but with custom learning curve:
```r
set.seed(101)
fit.lr <- train(Symbols ~ ., data = train, method = "glm", family = "binomial", metric = metric, trControl = control)
# ... (importance plot for top 10 genes, predict, cm_lr, etc.)

# Custom Learning Curve (simulates adding more training data)
custom_learning_curve <- function(train, test, formula, sizes = seq(0.1, 1.0, by = 0.1), seed = 80, filename = "learning_curve_lr.png") {
  set.seed(seed); accuracies <- c()
  for (s in sizes) {
    idx <- sample(1:nrow(train), floor(s * nrow(train)))  # Subset train data
    train_subset <- train[idx, ]
    fitlr <- glm(formula, data = train_subset, family = binomial)  # Fit GLM
    probs <- predict(fitlr, test, type = "response")  # Probabilities
    preds <- ifelse(probs > 0.5, "Cancer", "Normal")  # Threshold to labels
    cm <- confusionMatrix(as.factor(preds), test$Symbols, positive = "Cancer")
    accuracies <- c(accuracies, cm$overall["Accuracy"])
  }
  results <- data.frame(TrainingSize = sizes * 100, Accuracy = accuracies)
  png(filename, width = 800, height = 600)
  plot(results$TrainingSize, results$Accuracy, type = "b", pch = 19,  # Line + points
       xlab = "Training Set Size (%)", ylab = "Accuracy", main = "Custom Learning Curve - LR")
  dev.off()
  return(results)
}
results_lr <- custom_learning_curve(train, test, Symbols ~ .)  # Run & print
```
- **What**: Fits a line to probabilities (sigmoid curve).
- **Why**: Interpretable (coefficients = gene effects); baseline for comparison.
- **Tip**: Assumes linear relationships—use if genes are pre-scaled. Custom curve shows if more data helps.

##### 5. Decision Tree (DT)
```r
set.seed(101)
fit.dt <- train(Symbols ~ ., data = train, method = "rpart", metric = "Accuracy", trControl = control,
                tuneGrid = expand.grid(cp = seq(0.0001, 0.1, 0.005)),  # Tune complexity (cp: prune threshold)
                control = rpart.control(minsplit = 2, minbucket = 1))  # Allow small splits
# ... (importance, predict, cm_dt, etc.)
```
- **What**: Tree of yes/no splits (e.g., "If GeneA > 5, then Cancer?").
- **Why**: Visual (plot tree with `fancyRpartPlot`), easy to explain.
- **Tip**: Pruning prevents overfitting; low minsplit for small datasets.

##### 6. XGBoost – Advanced Boosting
Similar:
```r
set.seed(101)
fit.xgb <- train(Symbols ~ ., data = train, method = "xgbTree", metric = metric, trControl = control)
# ... (importance—fix bug: change plot(varImp(fit.dt)) to fit.xgb, predict, cm_xgb, etc.)
```
- **What**: Builds trees sequentially, fixing previous errors.
- **Why**: Often top performer on tabular data like genes; handles missing values.
- **Tip**: Fast; add `tuneGrid` for eta/max_depth. Bug in code: Use `fit.xgb` in plot.

#### Step 4: Compare Models
```r
# List of accuracies (from earlier)
acc_df <- data.frame(Model = names(model_accuracies), Accuracy = unlist(model_accuracies))

# Bar Plot
p <- ggplot(acc_df, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  ylim(0, 1) + geom_text(aes(label = round(Accuracy, 3)), vjust = -0.5, size = 5) +  # Labels on bars
  ggtitle("Model Accuracy Comparison") + xlab("Models") + ylab("Accuracy") +
  theme_bw() + theme(...) + coord_flip()  # Horizontal bars for readability

ggsave("model_accuracy_comparison.png", plot = p, width = 8, height = 6, dpi = 300)
print(acc_df)  # Console table
```
- **What**: Bar chart + table of % accuracy per model.
- **Why**: Quick winner (e.g., XGBoost often highest). Example table:
  
  | Model | Accuracy |
  |-------|----------|
  | xgb   | 0.95    |
  | rf    | 0.92    |
  | ...   | ...     |

- **Tip**: Sort by Accuracy descending. For ties, check sensitivity (recall for Cancer).

### Overall: What You Get & Next Steps
- **Outputs**: 6x (CM PNG + Learning Curve PNG + Importance PNG) + Comparison PNG + Console stats.
- **Typical Results**: 80-95% accuracy on gene data; RF/XGB shine.
- **Tips to Run/Improve**:
  - Add train/test split if missing (see above).
  - Scale data: Add `preProcess = c("center", "scale")` in train() for distance-based models (kNN/SVM).
  - More metrics: Use `pROC` for ROC curves (`roc(test$Symbols ~ as.numeric(pred.prob))`).
  - Overfitting? If train acc >> test, increase CV folds.
  - Shorter? Wrap models in a loop: `models <- c("knn", "svmRadial", ...); for(m in models) { fit <- train(... method=m); ... }`.

This turns data into predictions—great for biomarker discovery! If errors (e.g., no train/test), share output. 🚀

  
