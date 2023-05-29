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

actual_labels <- predicted_data[, label_var_predicted]
original_labels <- test_data[, label_var_test]

mcc <- mltools::mcc(as.vector(actual_labels), as.vector(original_labels))
auc <- pROC::roc(as.vector(actual_labels), as.vector(original_labels))$auc

