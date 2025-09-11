
#-----------------------#
# DEG Analysis - Part 2
#-----------------------#
# 1. Data Cleaning - Dimensionality Reduction (PCA)
# 2. Differentially Expressed Genes (DEGs) Identification 
# 3. DEGs Visualization (Volcano plot & Heatmap)


#=====================#
#   PCA Plot
#=====================#
install.packages("factoextra")
library(tidyverse)
library(factoextra)
library(limma)


data <- read.csv("ExpSet_PostNorm.csv")
nrow(data)  # Check the number of rows in your data
length(c(rep("Normal", 10), rep("Cervical_cancer", 10)))  # Check the length of your group labels


# Adjust Group Labels
data_t <- t(data[,-1 ])  # Exclude the first col (gene names) during transpose
data_t <- as.data.frame(data_t)
data_t$Group <- c(rep("Normal", 10), rep("Cervical_cancer", 10))
data_t$Group

# Perform PCA
pca_res <- prcomp(data_t[, -ncol(data_t)])

# view all PC scores 
head(pca_res$x)



# Save PCA plot as high-resolution TIFF
png("pca.png", width = 2000, height = 2000, res = 300)

# Generate PCA plot
fviz_pca_ind(pca_res,
             geom.ind = c("point", "text"),
             col.ind = data_t$Group,
             palette = c("red", "blue"),
             addEllipses = TRUE,
             ellipse.type = "confidence",
             legend.title = "Group",
             labelsize = 2  # increased label size for better clarity
)
dev.off()

#=======================#
# DEGs Identification
#=======================#
# Model Matrix Design
# Let's create a model matrix using the factor() function to represent the condition labels ("Normal" and "Cervical_cancer")
design <- model.matrix(~factor(c("Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal" , "Cervical_cancer","Cervical_cancer","Cervical_cancer","Cervical_cancer","Cervical_cancer","Cervical_cancer","Cervical_cancer","Cervical_cancer","Cervical_cancer","Cervical_cancer")))
# Now assign names ("Normal" and "Cervical_cancer") to the columns of the model matrix
colnames(design) <- c("Normal", "Cervical_cancer")

# Fits a linear model for each gene based on the given series of arrays
# It estimates the relationship between gene expression and conditions

fit <- lmFit(normset[,1:ncol(normset)], design) 
fit

# Contrast Matrix Design
# Define the specific comparison between conditions you want to analyze.
cont.matrix = makeContrasts(Cervical_cancer - Normal, levels=design)
cont.matrix

# Fitting model with Contrasts(2 Groups), so apply the defined contrast to the previously fitted model (fit)
fit2 <- contrasts.fit(fit, cont.matrix)

# Model optimization / Empirical Bayes Moderation
# Improves the estimation of variances for genes with low expression
# Computes moderated t-statistics and log-odds (B-stats) of differential expression by empirical Bayes shrinkage of the standard errors towards a common value
# contrast-specific information from fit2 is incorporated into the fit object, which is then passed to eBayes()
fit2 <- eBayes(fit)  
fit2

# Result Top Table
topTable(fit2, coef = 2, adjust.method = "BH") 

DEGs <- topTable(fit2, coef=2, adjust="BH", sort.by="logFC", number=100000); #inf
DEGs
write.csv(DEGs, "Result_Table_logFCsorted.csv", quote = F, row.names = TRUE)

#==================================================#
# Filter & Save final DEGs based on Pvalue & logFC
#==================================================#
# Read data of topTable
DEGs <- read.csv("Result_Table_logFCsorted.csv", header = TRUE)

# Filter & Save DEGs
final_DEGs <- DEGs[DEGs$P.Value < 0.05 & (DEGs$logFC > 2 | DEGs$logFC < -2), ]
write.csv(final_DEGs,"finalDEGs.csv", quote = F, row.names = F)



#=============================================#
# Annotate(getting Gene Symbols) filtered DEGs
#=============================================#
#BiocManager::install("hgu133plus2.db")
library("hgu133plus2.db")

DEGs <- read.csv("finalDEGs.csv", header = TRUE)
head(DEGs)


#open the the finalDEGs file manually and write the heading of affimetric IDs as Probe_IDs
probes=DEGs$Probe_ID
head(probes)
Symbols = unlist(mget(probes, hgu133plus2SYMBOL, ifnotfound=NA))
head(Symbols)

# Combine gene annotations with raw data
deg_anno = cbind(probes,Symbols, DEGs)
write.csv(deg_anno, "DEGs_Annotated.csv", quote = F, row.names = F)



#================================================#
# DEG Viz (Volcano plot)
#================================================#
install.packages("gdata")
install.packages("gplots")
library(gdata)
library(gplots)

DEGs <- read.csv("Result_Table_logFCsorted.csv", h=T)
head(DEGs)

png(filename = "VolcanoPlot_FC_2.png")
with(DEGs, plot(logFC, -log10(P.Value), pch=20, main="Volcano plot"))
with(subset(DEGs, P.Value < 0.05 & logFC > 2 ), points(logFC, -log10(P.Value), pch=20, col="red"))
with(subset(DEGs, P.Value < 0.05 & logFC < -2), points(logFC, -log10(P.Value), pch=20, col="green"))
dev.off()



#========================#
# Probe/Gene Annotation
#========================#
BiocManager::install("hgu133plus2.db")
library("hgu133plus2.db")

normset <- read.csv("ExpSet_PostNorm.csv", h=TRUE)
head(normset)

# Match probe IDs and retrieve Gene SYMBOLS
probes=normset$X
head(probes)
Symbols = unlist(mget(probes, hgu133plus2SYMBOL, ifnotfound=NA))

# Combine gene annotations with raw data
normset_anno = cbind(probes,Symbols,normset)
write.csv(normset_anno, "ExpSet_PostNorm_Annotated.csv", quote = F, row.names = F)


#open DEGs_annotated file and ExpSet_PostNorm_annotated file.
#In exp_postnorm annotated file, probe id will be repeated, son delete any one probe id and rename empty column as Genes
#A heatmap can effectively show upto 50- 70 genes, hence we need to consider only highly upregulated or highly downregulated genes.
#Go to DEGs_annotated file and filter logfc value greater than equal to 2.5 (which was initially 2) 
#and less than or equal to -2.5(which was initially -2)
#select the gene symbols and paste it in new sheet, and remove duplicates. select column- then data- then select remove duplicates
#Go to Exp_postNorm_annotated file and selcet the second row of Genes coluumn
#type =VLOOKUP(slect the fist gene symbol on adjuscent row comma go to the sheet where we have pasted filtered genes and 
#select column comma 1 (because it is 1st column) comma 0 (for exact matching))
#when press enter it will show #NA on first row, doble clich on box, it will autofill column
#copy this entire column and paste it in the 1st row which is heading Genes (while pating select paste as values i.e 123)
#filter the column and remove #NA and NA values
#now we have got the result, just copy pate the genes and expression values in another sheet and name as heatmap_data


#====================================#
# DEG Viz (Heatmap DEG Expression)
#====================================#
data <- read.csv(file = "Heatmap_data.csv", h=T)
head(data)


#to know the smallest values of gene expression , write =min(select the all 10 columns with expression data)
#for highest =max(select columns ,close bracket, enter)
#based on which give values in the col_breaks
#low exp red, high blue

rnames <- data[,1]
mat_data <- data.matrix(data[,2:ncol(data)])
rownames(mat_data) <- rnames
my_palette <- colorRampPalette(c("red", "blue"))(n = 299)
col_breaks = c(seq(0,5,length=100), # for red
               seq(5.1,10,length=100), # combo of red & blue
               seq(10.1,15.5,length=100)) # for blue
png("heatmap_exp_deg_cluster.png",     
     width = 6*300,        # 5 x 300 pixels
     height = 6*300,
     res = 300,            # 300 pixels per inch
     pointsize = 8)        # smaller font size

heatmap.2(mat_data,
          main = "Heatmap", # heat map title
          density.info="none",  # turns off density plot inside color legend
          trace="none",         # turns off trace lines inside the heat map
          margins =c(12,9),     # widens margins around plot
          col=my_palette,       # use on color palette defined earlier
          breaks=col_breaks, 
          dendrogram="both",     # only draw a row dendrogram
          Colv="T" ,         # turn off column clustering
          lhei = c(1,7)         # Key size width adjustment
)            
dev.off()
