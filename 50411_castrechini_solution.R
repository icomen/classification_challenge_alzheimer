# Load the required libraries
library(caret)
library(pROC)
library(mltools)
library(stats)

# Set the seed for reproducibility
set.seed(123)

# Define the classifiers and their hyperparameters
classifiers <- list(
  # "Logistic Regression" = list(
  #   method = "glm",
  #   family = "binomial",
  #   metric = "Accuracy"
  # ),
  # "Decision Tree" = list(
  #   method = "rpart",
  #   cp = 0.01,
  #   maxdepth = 4,
  #   metric = "Accuracy"
  # ),
  "Random Forest" = list(
    method = "rf",
    ntree = 500,
    mtry = 3,
    metric = "Accuracy"
  )
)

# Define the feature selection methods and their hyperparameters
feature_selection_methods <- list(
  "Variance Threshold" = list(
    method = "varianceThreshold",
    threshold = 0.01
  ),
  "Correlation Threshold" = list(
    method = "correlationThreshold",
    threshold = 0.7
  )
)

# Define a function to perform feature selection
featureSelection <- function(x, y, method, threshold) {
  switch(
    method,
    varianceThreshold = {
      selector <- caret::nearZeroVar(x, saveMetrics = TRUE)
      selector$nzv[selector$nzv] # Return the indices of the non-zero variance features
    },
    correlationThreshold = {
      selector <- caret::findCorrelation(cor(x), threshold, verbose = FALSE)
      xnames <- colnames(x)
      xnames[-selector]
    },
    stop("Invalid feature selection method.")
  )
}



# Load the training and test datasets for each binary problem
problems <- list(
  "ADCTL" = list(
    train = read.csv("ADCTL/ADCTLtrain.csv"),
    test = read.csv("ADCTL/ADCTLtest.csv")
  ),
  "ADMCI" = list(
    train = read.csv("ADMCI/ADMCItrain.csv"),
    test = read.csv("ADMCI/ADMCItest.csv")
  ),
  "MCICTL" = list(
    train = read.csv("MCICTL/MCICTLtrain.csv"),
    test = read.csv("MCICTL/MCICTLtest.csv")
  )
)




# Define a function to evaluate the performance of a classifier on a dataset
evaluateClassifier <- function(model, data, label_var) {
  preds <- predict(model, data[, -which(names(data) == label_var)], type = "class")
  probs <- predict(model, data[, -which(names(data) == label_var)], type = "prob")
  auc <- roc(data[, label_var], probs[, label_var])$auc
  #auc <- roc(data[, label_var], probs[, "AD"])$auc
  mcc <- mcc(data[, label_var], preds)
  return(list(auc = auc, mcc = mcc))
}

# Perform binary classification and feature selection for each binary problem
for (problem_name in names(problems)) {
  problem_params <- problems[[problem_name]]

  #train_data <- problem$train
  train_data <- problem_params$train
  test_data <- problem_params$test
  #test_data <- problem$test

  # Define the label and feature variables
  #label_var <- names(train_data)[1]
  label_var <- names(train_data)[ncol(train_data)]
  #feature_vars <- names(train_data)[-1]
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
      selected_features <- featureSelection(train_data[, feature_vars], train_data[, label_var], method = fs_params$method, threshold = fs_params$threshold)
      selected_feature_vars <- names(train_data)[selected_features]

      #print(selected_feature_vars)

      # Train the classifier
      classifier_model <- train(train_data[, selected_feature_vars], train_data[, label_var],
                                method = classifier_params$method,
                                metric = classifier_params$metric)

      #family = classifier_params$family

      # Evaluate the performance on the training data using cross-validation
      cv_results <- train(train_data[, selected_feature_vars], train_data[, label_var],
                          method = classifier_params$method, family = classifier_params$family,
                          trControl = train_control, tuneGrid = classifier_params$tune_grid, metric = classifier_params$metric)

      # Print the cross-validation results
      cat("\n", classifier_name, "with", fs_name, "feature selection (training):")
      print(cv_results$results)


      # Evaluate the performance on the test data
      test_predictions <- predict(classifier_model, newdata = test_data[, selected_feature_vars])
      test_results <- evaluate(classifier_model, test_predictions, label_var)
      #test_data[, label_var]

      # Print the test results
      cat("\n", classifier_name, "with", fs_name, "feature selection (test):")
      print(test_results)

      output_df <- data.frame(problem = character(), classifier = character(), feature_selection = character(), auc = numeric(), mcc = numeric())

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


