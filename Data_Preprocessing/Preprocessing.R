
#  Set working directory to raw CEL files folder
setwd("e:/Microarray/Cervical_Cancer_data/")
list.files()
getwd()


#install.packages("R.utils")
library(R.utils)

# list all .CEL.gz files

cel_files <- list.files(pattern = "\\.CEL\\.gz$", full.names = TRUE)
head(cel_files)

# Unzip all .CEL.gz files (keep originals)

for (file in cel_files) {
  gunzip(file, remove = FALSE)  # Set remove = TRUE to delete .gz after unzipping
}


# verify the extracted .CEL files

cel_unzipped <- list.files(pattern = "\\.CEL$", full.names = TRUE)
length(cel_unzipped)
head(cel_unzipped)

# Rename the .CEL files with proper names

# Create a new folder for renamed files
new_folder <- "Cervical_Cancer_data"
dir.create(new_folder, showWarnings = FALSE)

# Rename each file with GSM ID and condition
for (old_file in cel_unzipped) {
  # Extract just the filename
  old_name <- basename(old_file)
  
  # Extract GSM ID (e.g., GSM123456)
  gsm_id <- sub("^(GSM\\d+).*", "\\1", old_name)
  
  # Extract condition (_Normal or _Cancer)
  condition <- sub(".*(_Normal|_Cancer).*", "\\1", old_name)
  
  # Create new filename
  new_file_name <- paste0(gsm_id, condition, ".CEL")
  
  # Define full path to new file
  new_file_path <- file.path(new_folder, new_file_name)
  
  # Copy and rename the file
  file.copy(from = old_file, to = new_file_path)
}

# set working directory to the renamed files folder
setwd(file.path(getwd(), new_folder))
list.files()

getwd()
list.files()

# Install packages
#if (!requireNamespace("BiocManager", quietly = TRUE))
  #install.packages("BiocManager")

#BiocManager::install("affy")
#BiocManager::install("affyPLM")
#BiocManager::install("limma")
#BiocManager::install("hgu133plus2.db")
#BiocManager::install("hgu133plus2cdf")


# Load packages
library(limma)
library(affy)
library(affyPLM)
library(hgu133plus2.db)
library(hgu133plus2cdf)
library(IRanges)
library(RColorBrewer)


getwd()

targets <- readTargets("target.txt") 
targets

#Read CEL Files
data <- ReadAffy(filenames = targets$FileName)
data

# RMA Normalization
eset <- rma(data)
normset <- exprs(eset)

write.csv(normset, "ExpSet_PostNorm.csv", quote = F)

#   Box Plot
# Set up a 1x2 plotting layout
par(mfrow = c(1, 2))

# Boxplot Before Normalization
png(file = "Boxplot_Pre-Normalization.png", bg = "transparent", width = 600, height = 500, res = 100)
par(mar = c(10, 5, 4, 2) + 0.1, bg = "white", family = "sans") # Adjusted margins and font
boxplot(data,
        col = brewer.pal(8, "Set2")[1], # Light coral color from RColorBrewer
        main = "Boxplot Pre-Normalization",
        xlab = "", ylab = "Intensities",
        las = 2, cex.axis = 0.8, cex.lab = 1.2, cex.main = 1.4, # Refined font sizes
        whiskcol = "gray40", staplecol = "gray40", boxcol = "gray40", # Subtle whisker/staple colors
        outpch = 21, outbg = "red", outcex = 0.7) # Outlier customization
title(xlab = "Sample Array", line = 9, cex.lab = 1.2)
grid(col = "gray80", lty = "dotted") # Add subtle gridlines
dev.off()

# Boxplot After Normalization
png(file = "Boxplot_Post-Normalization.png", bg = "transparent", width = 600, height = 500, res = 100)
par(mar = c(10, 5, 4, 2) + 0.1, bg = "white", family = "sans") # Consistent styling
boxplot(normset,
        col = brewer.pal(8, "Set2")[2], # Light blue color from RColorBrewer
        main = "Boxplot Post-Normalization",
        xlab = "", ylab = "Intensities",
        las = 2, cex.axis = 0.8, cex.lab = 1.2, cex.main = 1.4,
        whiskcol = "gray40", staplecol = "gray40", boxcol = "gray40",
        outpch = 21, outbg = "blue", outcex = 0.7)
title(xlab = "Sample Array", line = 9, cex.lab = 1.2)
grid(col = "gray80", lty = "dotted") # Add subtle gridlines
dev.off()
