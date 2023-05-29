library(caret)
library(pROC)
library(mltools)
library(magrittr)
library(randomForest)
library(glmnet)

# Set the seed for reproducibility
set.seed(123)

chooseProblem <- function() {
  problem_name <- readline("Please enter the binary problem name (ADCTL, ADMCI, MCICTL): ")

  while (problem_name != "ADCTL" && problem_name != "ADMCI" && problem_name != "MCICTL") {
    cat("Invalid input. Please enter one of the following: ADCTL, ADMCI, MCICTL.\n")
    problem_name <- readline("Please enter the binary problem name (ADCTL, ADMCI, MCICTL): ")
  }

  predicted_file <- paste0(problem_name, "/", "50411_castrechini_",problem_name ,"res.csv")
  test_file <- paste0(problem_name, "/", problem_name, "test_wl.csv")
  predicted_data <- read.csv(predicted_file)
  test_data <- read.csv(test_file)

  return(list(predicted_data = predicted_data, test_data = test_data, problem_name = problem_name))
}

# Load the training and test datasets
data <- chooseProblem()
predicted_data <- data$predicted_data
test_data <- data$test_data


# Define the label
label_var_predicted <- names(predicted_data)[2]
label_var_test <- names((test_data)[ncol(test_data)])

predicted_label <- predicted_data[, label_var_predicted]
original_labels <- test_data[, label_var_test]


if(data$problem_name == "ADCTL") {
  predicted_label <- ifelse(predicted_label == "AD", 1, 0)
  original_labels <- ifelse(original_labels == "CTL", 0, 1)
} else if (data$problem_name == "ADMCI"){
  predicted_label <- ifelse(predicted_label == "AD", 1, 0)
  original_labels <- ifelse(original_labels == "MCI", 0, 1)
} else {
  predicted_label <- ifelse(predicted_label == "MCI", 1, 0)
  original_labels <- ifelse(original_labels == "CTL", 0, 1)
}


accuracy <- sum(predicted_label == original_labels) / length(predicted_label)


true_positives <- sum(predicted_label == 1 & original_labels == 1)
false_negatives <- sum(predicted_label == 1 & original_labels == 0)
true_negatives <- sum(predicted_label == 0 & original_labels == 0)
false_positives <- sum(predicted_label == 0 & original_labels == 1)

sensitivity <- true_positives / (true_positives + false_negatives)
specificity <- true_negatives / (true_negatives + false_positives)

mcc <- mltools::mcc(as.vector(predicted_label), as.vector(original_labels))
auc <- roc(predicted_label, original_labels)$auc


precision <- true_positives / (true_positives + false_positives)
f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
balanced_accuracy <- (sensitivity + specificity) / 2
