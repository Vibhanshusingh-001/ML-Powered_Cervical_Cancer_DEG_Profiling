

# Affymetrix `.CEL` File Format

A **CEL file** is the raw data output from **Affymetrix GeneChip microarrays**.
It contains **probe-level intensity measurements** along with metadata about the chip and scanning process.
These files are crucial for **preprocessing, normalization, and quality control** in microarray analysis.



## 1. Versions of CEL Format

* **CEL v3 (ASCII / Text-based)**

  * Human-readable format
  * Data stored as plain text sections

* **CEL v4 (Binary)**

  * Compact and efficient
  * Not human-readable (appears as random symbols in a text editor)
  * Standard format in most modern datasets (e.g., GEO CEL files)

---

## 2. Structure of CEL Files

### (A) Header Section

Contains metadata about the array layout, grid geometry, and analysis parameters.

**Example (ASCII v3):**

```plaintext
[CEL]
Version=3
Cols=1164
Rows=1164
TotalX=1164
TotalY=1164
GridCornerUL=228 214
GridCornerUR=8422 230
GridCornerLR=8406 8426
GridCornerLL=213 8411
Algorithm=Percentile
AlgorithmParameters=Percentile:75;CellMargin:2;OutlierHigh:1.500;OutlierLow:1.004
```

---

### (B) Intensity Section

Contains raw probe intensity values with associated statistics.

**Example (ASCII v3):**

```plaintext
[INTENSITY]
CellHeader=X Y MEAN STDV NPIXELS
0   0   456.7   12.3   9
1   0   467.2   13.1   9
2   0   460.5   11.8   9
```

**Columns:**

* **X, Y** → Probe coordinates
* **MEAN** → Raw signal intensity
* **STDV** → Standard deviation of pixel intensities
* **NPIXELS** → Number of pixels used in the measurement

In **Binary CEL v4**, these values are stored in binary blocks (not human-readable).

---

### (C) Masks Section

Lists probes excluded due to physical defects.

```plaintext
[MASKS]
CellHeader=X Y
45  103
46  103
```

---

## 3. Example Dummy CEL (ASCII v3)

```plaintext
[CEL]
Version=3
Cols=5
Rows=5
Algorithm=Percentile
AlgorithmParameters=Percentile:75;CellMargin:2

[INTENSITY]
CellHeader=X Y MEAN STDV NPIXELS
0 0 456.7 12.3 9
1 0 467.2 13.1 9
2 0 460.5 11.8 9
0 1 489.3 12.0 9
1 1 478.6 13.7 9

[MASKS]
CellHeader=X Y
1 2

[OUTLIERS]
CellHeader=X Y
2 2
```

---

## 4. ASCII vs Binary CEL (Comparison)

| **ASCII CEL v3** (Human-readable) | **Binary CEL v4** (Not readable) |
| --------------------------------- | -------------------------------- |
| \`\`\`plaintext                   | \`\`\`plaintext                  |
| \[CEL]                            | ^@^D^D^D^@ ... random symbols    |
| Version=3                         | Cols=1164                        |
| Cols=5                            | Rows=1164                        |
| Rows=5                            | GridCornerUL=228 214             |
| \[INTENSITY]                      | ...                              |
| 0 0 456.7 12.3 9                  | Binary-encoded intensities       |
| 1 0 467.2 13.1 9                  | (not human-readable)             |
| ...                               |                                  |
| \`\`\`                            | \`\`\`                           |

---

## 5. Reading CEL Files 

### In **R (Bioconductor)**

```r
library(affy)   # or 'oligo'
data <- ReadAffy(filenames = "sample.CEL")
exprs(data)     # Extract intensity matrix
```

### In **Python (Biopython)**

```python
from Bio.Affy import CelFile

with open("sample.CEL") as handle:
    c = CelFile.read(handle)
    print(c.ncols, c.nrows)   # Array dimensions
```

