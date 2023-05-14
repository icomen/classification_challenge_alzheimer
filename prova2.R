library(caret)
library(pROC)
library(mltools)

# Load the training and test datasets
ADCTLtrain <- read.csv("ADCTL/ADCTLtrain.csv")
ADMCItrain <- read.csv("ADMCI/ADMCItrain.csv")
MCICTLtrain <- read.csv("MCICTL/MCICTLtrain.csv")

ADCTLtest <- read.csv("ADCTL/ADCTLtest.csv")
ADMCItest <- read.csv("ADMCI/ADMCItest.csv")
MCICTLtest <- read.csv("MCICTL/MCICTLtest.csv")

# Define the binary classification problems
binary_problems <- list(
  list(train = ADCTLtrain, test = ADCTLtest, label = "AD vs CTL"),
  list(train = ADMCItrain, test = ADMCItest, label = "AD vs MCI"),
  list(train = MCICTLtrain, test = MCICTLtest, label = "MCI vs CTL")
)

# Define the classifiers to be used
classifiers <- list(
  logistic = list(method = "glm", family = "binomial"),
  knn = list(method = "knn"),
  svm = list(method = "svmLinear"),
  random_forest = list(method = "rf")
)

# Define the feature selection methods to be used
feature_selection_methods <- list(
  variance_threshold = list(method = "nearZeroVar", freqCut = 10),
  correlation_based = list(method = "cor", cutoff = 0.9),
  recursive_feature_elimination = list(method = "rfe", sizes = 5)
)

# Define the featureSelection function
featureSelection <- function(train_data, label_var, method_name) {
  # Get the feature selection method
  method_params <- feature_selection_methods[[method_name]]

  if (method_name == "variance_threshold") {
    # Variance threshold method
    selected_feature_vars <- names(train_data)[!caret::nearZeroVar(train_data[, -which(names(train_data) == label_var)], freqCut = method_params$freqCut)]

  } else if (method_name == "correlation_based") {
    # Correlation-based method
    cor_mat <- cor(train_data[, -which(names(train_data) == label_var)])
    high_cor_vars <- findCorrelation(cor_mat, cutoff = method_params$cutoff, verbose = FALSE)
    selected_feature_vars <- names(train_data)[-which(names(train_data) == label_var)][high_cor_vars]

  } else if (method_name == "recursive_feature_elimination") {
    # Recursive feature elimination method
    method <- caret::rfe(
      train_data[, -which(names(train_data) == label_var)],
      train_data[, label_var],
      sizes = method_params$sizes,
      method = method_params$method,
      verbose = FALSE
    )
    selected_feature_vars <- names(train_data)[method$optVariables]

  } else {
    # Unsupported method
    stop(paste("Unsupported feature selection method:", method_name))
  }

  # Return the selected feature variables
  return(selected_feature_vars)
}



# Define the performance metrics to be used
metrics <- c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "AUC", "MCC", "Balanced Accuracy")

# Perform binary classification and evaluation for each problem
for (problem in binary_problems) {

  # Get the train and test data
  train_data <- problem$train
  test_data <- problem$test

  # Define the label and feature variables
  label_var <- names(train_data)[ncol(train_data)]
  train_data_tmp <- train_data[, -c(1, ncol(train_data))]
  feature_vars <- names(train_data_tmp)

  # Preprocess the data
  preproc <- preProcess(train_data[, feature_vars], method = c("center", "scale"))
  train_data[, feature_vars] <- predict(preproc, train_data[, feature_vars])
  test_data[, feature_vars] <- predict(preproc, test_data[, feature_vars])

  # Split the data into training and validation sets for cross-validation
  train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE)

  # Perform binary classification and feature selection for each classifier and feature selection method
  for (classifier_name in names(classifiers)) {
    for (fs_name in names(feature_selection_methods)) {

      classifier_params <- classifiers[[classifier_name]]
      fs_params <- feature_selection_methods[[fs_name]]

      # Perform feature selection
      selected_features <- featureSelection(train_data[, feature_vars], train_data[, label_var], method = fs_params$method)
      selected_feature_vars <- names(train_data)[selected_features]

      # Train the classifier
      classifier_model <- train(train_data[, selected_feature_vars], train_data[, label_var], method = classifier_params$method, family = classifier_params$family)

      # Evaluate the performance on the training data using cross-validation
      cv_results <- train(train_data[, selected_feature_vars], train_data[, label_var], method = classifier_params$method, family = classifier_params$family, trControl = train_control, tuneGrid = classifier_params$tune_grid, metric = classifier_params$metric)

      # Print the cross-validation results
      cat("\n", classifier_name, "with", fs_name, "feature selection (training):")
      print(cv_results$results)

      # Evaluate the performance on the test data
      test_predictions <- predict(classifier_model, newdata = test_data[, selected_feature_vars])
      test_results <- evaluate(test_predictions, test_data[, label_var], positive = "AD")

      # Print the test results
      cat("\n", classifier_name, "with", fs_name, "feature selection (test):")
      print(test_results)

      # Save the results to the output dataframe
      output_df <- rbind(output_df, data.frame(
        problem = problem_name,
        classifier = classifier_name,
        feature_selection = fs_name,
        auc = test_results$auc,
        mcc = test_results$mcc
      ))
    }
  }
}

# Print the final results
cat("\nFinal results:")
print(output_df)




