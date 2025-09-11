

## ✅ What is the **R.utils** Package?

The **R.utils** package provides various utility functions to facilitate programming in R, including:

* File and directory manipulation
* Argument parsing
* Logging
* System operations
* Other useful helper functions
---

## 🔧 Common Features of R.utils

| Feature              | Example Function                           |
| -------------------- | ------------------------------------------ |
| File operations      | `copyFile()`, `moveFile()`, `deleteFile()` |
| Directory operations | `copyDirectory()`, `moveDirectory()`       |
| Logging              | `catAndRun()`, `verbose()`                 |
| System calls         | `syscommand()`                             |
| Argument handling    | `commandArgs(asValues = TRUE)`             |
| Timer utilities      | `timerStart()`, `timerStop()`              |

---


### ✅ 1. **R.utils**

* Provides utility functions to ease R programming.
* Common uses:

  * File manipulation: `copyFile()`, `deleteFile()`
  * Directory management: `copyDirectory()`, `moveDirectory()`
  * Logging & verbose output: `verbose()`
  * Command-line argument parsing: `commandArgs(asValues = TRUE)`
  * Timer utilities: `timerStart()`, `timerStop()`
* Useful when building large pipelines or automating workflows in R.

---

### ✅ 2. **limma**

* **Purpose**: Linear Models for Microarray Data.
* Primarily used for analyzing gene expression data, especially from microarrays.
* Key functions:

  * Differential expression analysis using linear models.
  * `lmFit()`, `eBayes()`, `topTable()`
* Example use case:

  * Comparing gene expression between experimental conditions.

---

### ✅ 3. **affy**

* **Purpose**: Analysis of Affymetrix microarray data.
* Key features:

  * Reading `.CEL` files (raw microarray data).
  * Data preprocessing: background correction, normalization.
  * Expression measure summarization.
* Important functions:

  * `ReadAffy()`: Read raw data.
  * `rma()`: Robust Multi-array Average normalization.
* Workflow: Raw `.CEL` → Preprocessing → Expression Matrix.

---

### ✅ 4. **affyPLM**

* **Purpose**: Probe-level modeling for quality assessment of Affymetrix microarrays.
* Focus:

  * Model-based quality metrics.
  * Generation of diagnostic plots (e.g., NUSE and RLE plots).
* Common functions:

  * `fitPLM()`: Fit probe-level model.
  * `NUSE()`, `RLE()`: Normalized unscaled standard errors, relative log expression.
* Used to assess microarray data quality.

---

### ✅ 5. **hgu133plus2.db**

* **Purpose**: Annotation package for the Affymetrix Human Genome U133 Plus 2.0 Array.
* Provides mappings:

  * Probe IDs ↔ Gene Symbols
  * Probe IDs ↔ Entrez Gene IDs
  * Probe IDs ↔ Chromosomal locations
* Example functions:

  * `select()`, `mapIds()` to retrieve annotations for probe sets.
* Important when interpreting expression data.

---

### ✅ 6. **hgu133plus2cdf**

* **Purpose**: Chip Definition File (CDF) package for Affymetrix Human Genome U133 Plus 2.0.
* Provides detailed probe layout of the microarray:

  * Which probes belong to which probe sets.
* Enables redefinition of probe sets or use of custom CDF.
* Example use case:

  * Use alternative probe set definitions to improve analysis accuracy.

---

### ✅ 7. **IRanges**

* **Purpose**: Efficient manipulation of genomic intervals and ranges.
* Data structure: `IRanges` object holds start/end positions of genomic features.
* Key uses:

  * Overlap detection.
  * Range arithmetic.
  * Genomic feature modeling.
* Common functions:

  * `IRanges(start, end)`: Create range objects.
  * `findOverlaps()`: Identify overlaps between genomic intervals.
* Widely used in Bioconductor genomic workflows.

---

### ✅ 8. **RColorBrewer**

* **Purpose**: Provides color palettes for data visualization.
* Features:

  * Predefined palettes suitable for different purposes:

    * Sequential, Diverging, Qualitative color schemes.
* Common functions:

  * `display.brewer.all()`: Show all palettes.
  * `brewer.pal(n, "Set1")`: Generate n colors from a given palette.
* Helps create publication-quality plots.

---
