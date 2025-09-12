
# ML (Machine Learning) 
# Model Development for Gene Expression Analysis

#open Postnorm annotated and DEG annotated file
#copy gene names from GEGs annotated file and paste in another sheet
#map these genes in postnorm annotated file , copy paste as values, and remove NA values, duplicate genes
#take only DEGs and its expression values in another csv file name it as Short_table for ML analysis
#move this file to a new folder inside this called ML_Analysis

setwd("e:/Microarray/Cervical_Cancer_data/ML_analysis/")

# Load the required packages
install.packages("xlsx")
install.packages("caret")
install.packages("glmnet")

# Importing required libraries
library("readxl")  
library("tidyverse") # data manipulation and visualization package
library("caret")  # machine learning library package
library("glmnet")

#======================#
## load Data
#======================#
df <- read.csv("Short_table.csv", header = T) #if it was workbook then we have to use df<-read_excel("short_table.xlsx", sheet = 1)
head(df)

#============================================#
# Data Pre-processing |   EDA(exploratory data analysis) 
#============================================#

{  # Check the structure of the dataset
  
  str(df)
  
  dim(df)
  
  
  # remove rows that contain NA values
  df <- df[complete.cases(df), ]
  head(df)
  dim(df)
  
  
  #Calculate Mean of duplicate genes
  x <- df
  x <- data.frame(x)
  x <- do.call(rbind,lapply(lapply(split(x,x$DEGs),`[`,2:ncol(x)),colMeans))
  dim(x)
  
  #Convert rownames as a 1st column with header Symbols -> which became rownames after previous operation
  library(tibble) # from tidyverse
  x <- data.frame(x)
  x <- tibble::rownames_to_column(x, var="Symbols")
  head(x)
  dim(x) 
  
  df <- x
  
  # Transpose table 
  install.packages("sjmisc")
  library(sjmisc)  
  df_t <- rotate_df(df, cn=T)
  Symbols <- colnames(df[-1])
  df_t <- cbind(Symbols, df_t)
  write.csv(df_t, "transposed_table.csv", row.names=F)
  
  df_t <- read.csv("transposed_table.csv", h=T)
  dim(df_t)
  df_t[1]
  
  # Healthy_1.CEL to N/T
  #df_t[,1] <- gsub("_.*$", "", df_t[,1])#if the files are present like normal_1-s1.CEL
  #df_t[1]
  
  #My files are like GSM1234_NORMAL.CEL
  #hence i need to extract values between _ and . i.e Normal or Cancer
  #sub() → Substitutes the first match of a regex.
  
  #"\\1" → Returns only the captured group (i.e., what's between _ and .CEL)
  #.*_ → Matches everything up to the last underscore.
  #(.*?) → Captures the shortest string between _ and .CEL → e.g., "Normal" or "Cancer"
  #\\.CEL → Matches .CEL (the . is escaped with \\)
  
  
  df_t$Symbols <- sub(".*_(.*?)\\.CEL", "\\1", df_t$Symbols)
  df_t[1]
  
  # convert Healthy/DCIS from char to factor
  df_t[1] <- factor(df_t$Symbols)
  str(df_t)
  
  # df_t -> df
  df <- df_t # df is df_NT
  
}

# view transformed data
str(df)
df <- data.frame(df)
head(df)

#===============================================#
# Step 3.  Visualize Dataset - Figures - Plots 
#===============================================#
#####  Box and Whisker Plots  ##### 
# Given that the input variables are numeric, we can create box and whisker plots of each
png("box_and_whisker_plots.png")
par(mfrow=c(2,4))
for(i in 2:9) {
  boxplot(x[,i], main=names(df)[i], col="blue")
}
dev.off()


#####  Sample matrix  ##### 
library(ggplot2)
library(caret) 
# split input and output
x <- df[,2:ncol(df)]  # x -  inputs attributes
y <- df[,1]     # y -  outputs attributes
y<- as.factor(y)
plot(y, col="blue")


#======================#
# Step 2. Data Splitting
#======================#
# create 60%/40% for training and testing dataset
library(caret)
set.seed(101)
split <- createDataPartition(df$Symbols, p=0.60, list=FALSE)  # Return the row indices as a matrix/vector, not as a list.
train <- df[split,]
test <- df[-split,]

# dimensions of dataset, train, test
dim(df)
dim(train)
dim(test)


#set cross-validation control for training
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

head(df_t)


#=========================#
# Build ML Models   
#=========================#

# ML (Machine Learning) 
# Model Development for Gene Expression Analysis - [Part 2]


# Install and Load the required packages
install.packages("caret")
library("caret")


# 1... kNN(k-Nearest Neighbor) - [Model 1]
#-------------------------------------------
set.seed(7)
fit.knn <- train(Symbols~., 
                 data=train, 
                 method="knn", 
                 metric=metric, 
                 trControl=control)
fit.knn


install.packages("cowplot")     # Only once
library(cowplot)                # Load every time you use plot_grid()

# check important variables
varImp(fit.knn)
p1 <- plot(varImp(fit.knn), top = 30, main="kNN")
p2 <- plot(fit.knn, main="kNN")
plot_grid(p1, p2)

# make predictions using trained model on new/test
pred.knn <- predict(fit.knn, newdata = test)

# Model Evaluation
confusionMatrix(pred.knn, test$Symbols, positive = "Cancer")
c1 <- confusionMatrix(pred.knn, test$Symbols, positive = "Cancer")$table

#========================#
# plot Confusion Matrix
#========================#
library(ggplot2)
library(dplyr)
table <- data.frame(c1)
plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "high", "low")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

# fill alpha relative to sensitivity/specificity by proportional outcomes within reference groups (see dplyr code above as well as original confusion matrix for comparison)
ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = Freq)) + # alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(high = "#009194", low="#FF9966")) +
  #scale_fill_gradient(low="white", high="#009194") +
  theme_bw() +
  xlim(rev(levels(table$Reference)))



# 2.SVM model     [model-2]
#-----------------------------------
# For bioinformatics tasks like gene expression classification:
# Use svmLinear when classes are clearly separable (e.g., PCA shows clusters),
# Use svmRadial for complex, non-linear patterns (common in omics data).

install.packages("kernlab")
library(kernlab)
set.seed(101)
fit.svm <- train(Symbols~., data=train, method="svmRadial", metric=metric, trControl=control)
fit.svm

# Feature Importance
varImp(fit.svm)

p1 <- plot(varImp(fit.svm), top = 30, main="Support Vector Machines with Radial Basis")
p2 <- plot(fit.svm)
plot_grid(p1, p2)


# Make Predictions # Confusion Matrix
pred.svm <- predict(fit.svm, newdata = test)

# Model Evaluation
confusionMatrix(pred.svm, test$Symbols, positive = "Cancer")
c1 <- confusionMatrix(pred.svm, test$Symbols, positive = "Cancer")$table



library(ggplot2)
library(dplyr)
table <- data.frame(c1)
plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "high", "low")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

# fill alpha relative to sensitivity/specificity by proportional outcomes within reference groups (see dplyr code above as well as original confusion matrix for comparison)
ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = Freq)) + # alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(high = "#009194", low="#FF9966")) +
  #scale_fill_gradient(low="white", high="#009194") +
  theme_bw() +
  xlim(rev(levels(table$Reference)))




# 3 .Random Forest Model - [Model 3]
#----------------------------------------
install.packages("randomForest")
library(randomForest)

set.seed(123)
fit.rf <- train(Symbols~.,
                data=train,
                method="rf",
                metric=metric,
                trControl=control)
fit.rf

# view important genes
varImp(fit.rf)



# visualize the important genes
plot(varImp(fit.rf), top = 30)

# make predictions using trained model on new/test
pred.rf <- predict(fit.rf, newdata = test)

# Model Evaluation
confusionMatrix(pred.rf, test$Symbols, positive = "Cancer")
c1 <- confusionMatrix(pred.rf, test$Symbols, positive = "Cancer")$table



#========================#
# plot Confusion Matrix
#========================#
library(ggplot2)
library(dplyr)
table <- data.frame(c1)
plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "high", "low")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

# fill alpha relative to sensitivity/specificity by proportional outcomes within reference groups (see dplyr code above as well as original confusion matrix for comparison)
ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = Freq)) + # alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(high = "#009194", low="#FF9966")) +
  #scale_fill_gradient(low="white", high="#009194") +
  theme_bw() +
  xlim(rev(levels(table$Reference)))



#generating models comaprision plot
acc_knn <- confusionMatrix(pred.knn, test$Symbols)$overall["Accuracy"]
acc_rf  <- confusionMatrix(pred.rf, test$Symbols)$overall["Accuracy"]
acc_svm <- confusionMatrix(pred.svm, test$Symbols)$overall["Accuracy"]

library(ggplot2)

accuracy_df <- data.frame(
  Model = c("kNN", "Random Forest", "SVM"),
  Accuracy = c(acc_knn, acc_rf, acc_svm)  # ✅ properly closed
)

p <- ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  ylim(0, 1) +
  geom_text(aes(label = round(Accuracy, 3)), vjust = -0.5, size = 5) +
  ggtitle("Model Accuracy Comparison") +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white", color = NA)
  )


# Save the barplot to PNG
ggsave("model_accuracy_comparison.png", plot = p, width = 6, height = 4, dpi = 300)







