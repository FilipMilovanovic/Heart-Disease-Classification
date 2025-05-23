# Load the dataset 
data <- read.csv("../data/heart.csv", stringsAsFactors = FALSE)

# Check for missing values using various patterns 
apply(data, MARGIN = 2, FUN = function(x) sum(is.na(x)))
apply(data, MARGIN = 2, FUN = function(x) sum(x == "-", na.rm = TRUE))
apply(data, MARGIN = 2, FUN = function(x) sum(x == "", na.rm = TRUE))
apply(data, MARGIN = 2, FUN = function(x) sum(x == " ", na.rm = TRUE))

# Convert categorical columns to factors
data$ChestPainType <- as.factor(data$ChestPainType)
data$Sex <- as.factor(data$Sex)
data$RestingECG <- as.factor(data$RestingECG)
data$ExerciseAngina <- as.factor(data$ExerciseAngina)
data$ST_Slope <- as.factor(data$ST_Slope)

# Encode target variable as factor with levels "No" and "Yes"
data$HeartDisease <- as.factor(data$HeartDisease)
levels(data$HeartDisease) <- c("No", "Yes")

# Quick check on FastingBS — binary feature
length(unique(data$FastingBS))




# Visualize the relationship between each predictor and the target variable using ggplot2.
# This step helps assess the discriminative power of input features by comparing their distribution across the target classes.
# Features showing distinguishable patterns between classes are likely to be more informative and valuable for the model.
# If a feature shows a clear visual separation between the classes of the target variable,
# it is more likely to contribute effectively to classification performance.
# Features with overlapping or uniform distributions across classes may provide limited predictive value.
library(ggplot2)

# Numerical distributions
ggplot(data, aes(x = Age, fill = HeartDisease)) + geom_density(alpha = 0.5)
ggplot(data, aes(x = RestingBP, fill = HeartDisease)) + geom_density(alpha = 0.5)
ggplot(data, aes(x = Cholesterol, fill = HeartDisease)) + geom_density(alpha = 0.5)
ggplot(data, aes(x = FastingBS, fill = HeartDisease)) + geom_density(alpha = 0.5)
ggplot(data, aes(x = MaxHR, fill = HeartDisease)) + geom_density(alpha = 0.5)
ggplot(data, aes(x = Oldpeak, fill = HeartDisease)) + geom_density(alpha = 0.5)

# Categorical distributions
ggplot(data, aes(x = Sex, fill = HeartDisease)) + geom_density(alpha = 0.5)
ggplot(data, aes(x = ChestPainType, fill = HeartDisease)) + geom_bar(position = 'dodge')
ggplot(data, aes(x = RestingECG, fill = HeartDisease)) + geom_bar(position = 'dodge')
ggplot(data, aes(x = ExerciseAngina, fill = HeartDisease)) + geom_bar(position = 'dodge')
ggplot(data, aes(x = ST_Slope, fill = HeartDisease)) + geom_bar(position = 'dodge')

# Detect outliers using boxplot stats
apply(data[,c(1,4,5,8,10)], 2, function(x) length(boxplot.stats(x)$out))

# Check for normality (used later to decide how to scale the data)
apply(data[,c(4,5,8,10)], 2, shapiro.test)

# Since none of the selected variables follow normal distribution,
# we standardize using the median and IQR
data.std <- as.data.frame(apply(data[,c(4,5,8,10)], 2, function(x) scale(x, center = median(x), scale = IQR(x))))

# Normalization for Age 
data$Age <- (data$Age - min(data$Age)) / (max(data$Age) - min(data$Age)) 

# Append relevant categorical and normalized features into one dataset
data.std$Sex <- as.numeric(data$Sex)
data.std$FastingBS <- as.numeric(data$FastingBS)
data.std$ChestPainType <- as.numeric(data$ChestPainType)
data.std$RestingECG <- as.numeric(data$RestingECG)
data.std$Age <- as.numeric(data$Age)
data.std$ExerciseAngina <- as.numeric(data$ExerciseAngina)
data.std$ST_Slope <- as.numeric(data$ST_Slope)
data.std$HeartDisease <- as.factor(data$HeartDisease)

# Inspect structure and summary of the processed data
str(data.std)
summary(data.std)

# Load caret for data partitioning
library(caret)

# Split the dataset into training and test sets using stratified sampling
# It's essential that this split is random to avoid bias. 
# For example, if data were collected over time, earlier observations may differ from later ones.
# Stratification ensures that both training and test sets preserve the original class distribution of the target variable.
# This helps the model generalize better, as learning from one distribution and predicting on another can degrade performance.
# A seed is set to ensure reproducibility of the random data split
set.seed(1010)
indexes <- createDataPartition(data.std$HeartDisease, p = 0.8, list = FALSE)
train.data <- data.std[indexes, ]
test.data <- data.std[-indexes, ]

# Setup for 10-fold cross-validation
numFolds <- trainControl(method = "cv", number = 10)

# Define the range of K values to test (odd numbers only)
# Using odd K avoids ties between classes
kGrid <- expand.grid(.k = seq(from = 3, to = 25, by = 2))

# Train KNN model using caret with cross-validation
library(e1071)
knn.cv <- train(x = train.data[,-12],
                y = train.data$HeartDisease,
                method = "knn",
                trControl = numFolds,
                tuneGrid = kGrid)

# Output performance results and optimal K
knn.cv
plot(knn.cv)

# Extract the best value for K from the CV results
best_k <- knn.cv$bestTune$k
best_k
# K = 13 was selected as the optimal number of neighbors based on cross-validation accuracy.
# This value balances bias and variance effectively, avoiding overfitting from low K or underfitting from high K.




# Predict using KNN with best K on the test set
library(class)
knn.pred <- knn(train = train.data[,-12],
                test = test.data[,-12],
                cl = train.data$HeartDisease,
                k = best_k)

# Confusion matrix comparing predicted vs actual classes
knn.cm <- table(true = test.data$HeartDisease, predicted = knn.pred)
knn.cm

# Custom evaluation function to compute accuracy, precision, recall, and F1 score
getEvaluationMetrics <- function(cm){
  TP <- cm[2,2]
  TN <- cm[1,1]
  FP <- cm[1,2]
  FN <- cm[2,1]

  accuracy <- sum(diag(cm)) / sum(cm)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1 <- (2 * precision * recall) / (precision + recall)

  c(Accuracy = accuracy, 
    Precision = precision, 
    Recall = recall, 
    F1 = F1)
}
# Confusion matrix interpretation:
# TP: actual positives correctly predicted as positives (true positives)
# TN: actual negatives correctly predicted as negatives (true negatives)
# FP: actual negatives incorrectly predicted as positives (false positives)
# FN: actual positives incorrectly predicted as negatives (false negatives)

# Evaluate the model's predictive performance
knn.eval <- getEvaluationMetrics(knn.cm)
knn.eval

# Results:
# Accuracy ~86.89% – correctly predicted 158 out of 183 test cases
# Precision ~88.12% – of all predicted positives, this percent were truly positive
# Recall ~88.12% – of all actual positives, this percent were correctly identified
# F1 Score ~88.12% – harmonic mean of precision and recall
