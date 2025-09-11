
### ✅ 1. `library(tidyverse)`

* **What is it?**
  The **`tidyverse`** is a powerful and popular collection of R packages designed for data science tasks, built around the philosophy of "tidy data."

* **Key Features:**

  * Provides tools for data manipulation, visualization, and analysis.
  * Uses a consistent and readable syntax based on **`dplyr`** (for data manipulation) and **`ggplot2`** (for visualization).
  * Makes data workflows easier and more intuitive.

* **Commonly Included Packages:**

  * `ggplot2` → Visualization
  * `dplyr` → Data manipulation
  * `readr` → Reading data files
  * `tidyr` → Data tidying
  * `purrr` → Functional programming tools
  * `tibble` → Modern data frames

* **Why use it?**
  Simplifies common data science tasks with elegant syntax, promotes readable code, and helps process data in a structured way.

---

### ✅ 2. `library(factoextra)`

* **What is it?**
  The **`factoextra`** package is designed to easily visualize and interpret the results of multivariate data analyses, such as **PCA (Principal Component Analysis)**, **clustering**, and **correspondence analysis**.

* **Key Features:**

  * Helps visualize results from exploratory data analysis (EDA).
  * Functions for creating high-quality plots of PCA, clustering results (e.g., k-means, hierarchical), and other multivariate analyses.
  * Easy-to-use functions like `fviz_pca_ind()`, `fviz_cluster()`, and `fviz_dend()`.

* **Why use it?**
  Great for interpreting complex multidimensional data visually, making it easy to explain patterns, groupings, and relationships.

---

### ✅ 3. `library(limma)`

* **What is it?**
  The **`limma`** package stands for **Linear Models for Microarray Data** but is now widely used for analyzing differential expression in both microarray and RNA-seq data.

* **Key Features:**

  * Fits linear models to expression data.
  * Performs differential expression analysis using empirical Bayes moderation to improve statistical power.
  * Useful for identifying genes that are significantly up- or down-regulated between experimental conditions.

* **Why use it?**
  Powerful and fast for analyzing gene expression data, especially when dealing with high-dimensional data and multiple comparisons.

---

### ✅ Summary Table

| Package      | Purpose                                      | Key Functions                       |
| ------------ | -------------------------------------------- | ----------------------------------- |
| `tidyverse`  | Data manipulation and visualization          | `filter()`, `mutate()`, `ggplot()`  |
| `factoextra` | Visualizing multivariate analyses results    | `fviz_pca_ind()`, `fviz_cluster()`  |
| `limma`      | Differential expression analysis in genomics | `lmFit()`, `eBayes()`, `topTable()` |

