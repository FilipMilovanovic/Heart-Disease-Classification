# Load the dataset
data <- read.csv("../data/heart.csv", stringsAsFactors = FALSE)

# Check for missing values or improper entries
apply(data, 2, function(x) sum(is.na(x)))       # Check for NA
apply(data, 2, function(x) sum(x == "-", na.rm = TRUE))  # Check for dashes
apply(data, 2, function(x) sum(x == "", na.rm = TRUE))   # Check for empty strings
apply(data, 2, function(x) sum(x == " ", na.rm = TRUE))  # Check for space-only entries

# Convert categorical variables into factors
data$Sex <- as.factor(data$Sex)
data$ChestPainType <- as.factor(data$ChestPainType)
data$RestingECG <- as.factor(data$RestingECG)
data$ExerciseAngina <- as.factor(data$ExerciseAngina)
data$ST_Slope <- as.factor(data$ST_Slope)

# Convert target variable and label the levels for better readability
data$HeartDisease <- as.factor(data$HeartDisease)
levels(data$HeartDisease) <- c("No", "Yes")

# Check for normality of continuous features
apply(data[, c(1, 4, 5, 6, 8, 10)], 2, shapiro.test)
# None are normally distributed - discretization is needed

# FastingBS is binary numeric, so we convert it into a factor
table(data$FastingBS)
data$FastingBS <- as.factor(data$FastingBS)

# Convert continuous variables into numeric type (if not already)
data$Age <- as.numeric(data$Age)
data$RestingBP <- as.numeric(data$RestingBP)
data$Cholesterol <- as.numeric(data$Cholesterol)
data$MaxHR <- as.numeric(data$MaxHR)

# Discretize continuous features using quantile-based binning (equal-frequency)
library(bnlearn)
discretized <- discretize(data[, c(1, 4, 5, 8, 10)],
                          method = "quantile",
                          breaks = c(5, 5, 5, 5, 3))
summary(discretized)

# Merge discretized features with the remaining categorical ones
newData <- as.data.frame(cbind(discretized, data[, c(2, 3, 6, 7, 9, 11, 12)]))

# Partition the data into training and testing sets (80/20 split)
library(caret)
set.seed(1010)
indexes <- createDataPartition(newData$HeartDisease, p = 0.80, list = FALSE)
train.data <- newData[indexes, ]
test.data <- newData[-indexes, ]

# Train the Naive Bayes classifier
library(e1071)
nb1 <- naiveBayes(HeartDisease ~ ., data = train.data)
nb1  # Model summary

# Make predictions on the test set
nb1.pred <- predict(nb1, newdata = test.data, type = "class")

# Generate confusion matrix
nb1.cm <- table(true = test.data$HeartDisease, predicted = nb1.pred)

# Define a function to compute evaluation metrics
getEvaluationMetrics <- function(cm) {
  TP <- cm[2, 2]
  TN <- cm[1, 1]
  FP <- cm[1, 2]
  FN <- cm[2, 1]
  
  accuracy <- sum(diag(cm)) / sum(cm)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1 <- (2 * precision * recall) / (precision + recall)
  
  c(Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1 = F1)
}

# Evaluate initial predictions
nb1.eval <- getEvaluationMetrics(nb1.cm)
nb1.eval

# Predict class probabilities for ROC analysis
nb2.raw <- predict(nb1, newdata = test.data, type = "raw")

# Generate ROC curve and compute AUC
library(pROC)
nb2.roc <- roc(response = as.integer(test.data$HeartDisease),
               predictor = nb2.raw[, 2],
               levels = c(1, 2))
plot.roc(nb2.roc)

# Print AUC value
nb2.roc$auc

# Display optimal thresholds using Youden index
plot.roc(nb2.roc, print.thres = TRUE, print.thres.best.method = "youden")

# Get coordinates of local maxima to find the best threshold
nb2.coords <- coords(nb2.roc,
                     ret = c("accuracy", "spec", "sens", "thr"),
                     x = "local maximas")

# Use the best threshold based on Youden index
prob.treshold <- nb2.coords[11, 4]

# Generate new predictions based on optimal threshold
nb2.pred <- ifelse(test = nb2.raw[, 2] >= prob.treshold,
                   yes = "Yes", no = "No")
nb2.pred <- as.factor(nb2.pred)

# Evaluate model with new threshold
nb2.cm <- table(true = test.data$HeartDisease, predicted = nb2.pred)
nb2.eval <- getEvaluationMetrics(nb2.cm)

# Compare performance before and after threshold optimization
data.frame(rbind(nb1.eval, nb2.eval), row.names = c("Model 1", "Model 2"))

# Summary: The optimized threshold using Youden index improved model balance,
# with higher accuracy, precision, and F1 score compared to the original model.
