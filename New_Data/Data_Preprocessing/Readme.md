

**`ReadAffy()`** is a function from the **affy** package in R used to **load raw Affymetrix microarray data** from **.CEL files**.

### **What it does**

* Reads raw probe intensities from CEL files
* Combines them into an **AffyBatch object**
* Stores:

  * sample information
  * probe-level intensities
  * chip annotation
  * metadata

### **Why it is used**

To prepare Affymetrix microarray data for:

* background correction
* normalization (like RMA)
* summarization
* quality control

### **Example**

```r
data <- ReadAffy(filenames = targets$FileName)
```
<img width="578" height="202" alt="image" src="https://github.com/user-attachments/assets/38d8b395-f9a4-4c06-88a1-b9b473c81fa3" />

