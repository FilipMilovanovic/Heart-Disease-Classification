# Load the dataset and disable automatic factor conversion
data <- read.csv("../data/heart.csv", stringsAsFactors = FALSE)

# Check for missing values using multiple indicators
apply(data, MARGIN = 2, FUN = function(x) sum(is.na(x)))
apply(data, MARGIN = 2, FUN = function(x) sum(x == "-", na.rm = T))
apply(data, MARGIN = 2, FUN = function(x) sum(x == "", na.rm = T))
apply(data, MARGIN = 2, FUN = function(x) sum(x == " ", na.rm = T))

# Convert relevant categorical variables into factors
length(unique(data$Sex))
data$Sex <- as.factor(data$Sex)
length(unique(data$ChestPainType))
data$ChestPainType <- as.factor(data$ChestPainType)

data$FastingBS <- as.factor(data$FastingBS)
data$RestingECG <- as.factor(data$RestingECG)
data$ExerciseAngina <- as.factor(data$ExerciseAngina)
data$ST_Slope <- as.factor(data$ST_Slope)

# Convert the target variable to a labeled factor
library(ggplot2)
data$HeartDisease <- as.factor(data$HeartDisease)
levels(data$HeartDisease) <- c("No", "Yes")

# Visualize the relationship between each predictor and the target variable using ggplot2.
# This step helps assess the discriminative power of input features by comparing their distribution across the target classes.
# Features showing distinguishable patterns between classes are likely to be more informative and valuable for the model.
# If a feature shows a clear visual separation between the classes of the target variable,
# it is more likely to contribute effectively to classification performance.
# Features with overlapping or uniform distributions across classes may provide limited predictive value.
ggplot(data, aes(x = Age, fill = HeartDisease)) + geom_density(alpha = 0.5)
ggplot(data, aes(x = Sex, fill = HeartDisease)) + geom_density(alpha = 0.5)
ggplot(data, aes(x = ChestPainType, fill = HeartDisease)) + geom_density(alpha = 0.5)
ggplot(data, aes(x = RestingBP, fill = HeartDisease)) + geom_density(alpha = 0.5)
ggplot(data, aes(x = Cholesterol, fill = HeartDisease)) + geom_density(alpha = 0.5)
ggplot(data, aes(x = FastingBS, fill = HeartDisease)) + geom_bar(position = 'dodge')
ggplot(data, aes(x = RestingECG, fill = HeartDisease)) + geom_bar(position = 'dodge')
ggplot(data, aes(x = MaxHR, fill = HeartDisease)) + geom_density(alpha = 0.5)
ggplot(data, aes(x = ExerciseAngina, fill = HeartDisease)) + geom_bar(position = 'dodge')
ggplot(data, aes(x = Oldpeak, fill = HeartDisease)) + geom_density(alpha = 0.5)
ggplot(data, aes(x = ST_Slope, fill = HeartDisease)) + geom_bar(position = 'dodge')

# Load caret for data partitioning
library(caret)

# Split the dataset into training and test sets using stratified sampling
# It's essential that this split is random to avoid bias. 
# For example, if data were collected over time, earlier observations may differ from later ones.
# Stratification ensures that both training and test sets preserve the original class distribution of the target variable.
# This helps the model generalize better, as learning from one distribution and predicting on another can degrade performance.
# A seed is set to ensure reproducibility of the random data split
set.seed(1010)
indexes <- createDataPartition(data$HeartDisease, p = 0.8, list = FALSE)
train.data <- data[indexes, ]
test.data <- data[-indexes, ]

# Train a classification tree model on the training data
library(rpart)
tree1 <- rpart(HeartDisease ~ .,
               data = train.data,
               method = "class")

# Visualize the decision tree
library(rpart.plot)
rpart.plot(tree1, extra = 104)

# Generate predictions and confusion matrix
tree1.pred <- predict(tree1, newdata = test.data, type = "class")
tree1.cm <- table(true = test.data$HeartDisease, predicted = tree1.pred)
tree1.cm

# Define a function to compute common evaluation metrics from a confusion matrix.
# These metrics (accuracy, precision, recall and F1) help compare model performance beyond raw classification counts.
getEvaluationMetrics <- function(cm) {
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


# Evaluate the first decision tree model
tree1.eval <- getEvaluationMetrics(tree1.cm)
tree1.eval

# Summary of performance for the first decision tree model:
# Accuracy = 86.338%: correctly predicted 158 out of 183 observations
# Precision = 88.775%: proportion of predicted positives that are actually positive
# Recall = 86.138%: proportion of actual positives correctly identified
# F1 = 87.437%: harmonic mean of precision and recall

# Load required libraries for cross-validation
library(e1071)
library(caret)

# Perform 10-fold cross-validation to tune the complexity parameter (cp)
# Cross-validation helps estimate model performance more reliably by rotating through training/testing splits
numFolds <- trainControl(method = "cv", number = 10) 

# Define a grid of candidate cp values to evaluate
cpGrid <- expand.grid(.cp = seq(from = 0.001, to = 0.05, by = 0.001))

# Train models using cross-validation and select the optimal cp value
set.seed(1010)
crossvalidation <- train(x = train.data[,-12],
                         y = train.data$HeartDisease,
                         method = "rpart", 
                         trControl = numFolds, 
                         tuneGrid = cpGrid) 

# Inspect cross-validation results
crossvalidation
plot(crossvalidation)

# Extract the best-performing cp value based on cross-validation
bestCp <- crossvalidation$bestTune$cp
bestCp

# Train a new decision tree model using the optimized cp value
tree2 <- rpart(HeartDisease ~ ., 
               data = train.data,
               method = "class", 
               control = rpart.control(cp = bestCp))

# Predict and evaluate the optimized model
tree2.pred <- predict(tree2, newdata = test.data, type = "class")
tree2.cm <- table(true = test.data$HeartDisease, predicted = tree2.pred) 
tree2.cm
tree2.eval <- getEvaluationMetrics(tree2.cm)

#Accuracy = 86.885%
#Precision = 88.118%
#Recall = 88.118%
#F1 = 88.118%

# Compare performance metrics between the initial and optimized models
data.frame(rbind(tree1.eval, tree2.eval), row.names = c("Initial decision tree model", "Optimized decision tree model"))

# The optimized model yields a slightly higher F1 score, suggesting a more balanced performance.
# While the first model had slightly better precision, the second model performs better overall by achieving
# improved recall and F1, making it a more reliable choice for detecting positive cases.
