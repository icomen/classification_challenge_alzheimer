library(caret)
library(pROC)
library(mltools)
library(magrittr)
library(randomForest)

# Set the seed for reproducibility
set.seed(123)

# Load the training and test datasets
train_data <- read.csv("ADMCI/ADMCItrain.csv")
test_data <- read.csv("ADMCI/ADMCItest.csv")

# Define the label and feature variables
label_var <- names(train_data)[ncol(train_data)]
train_data_tmp <- train_data[, -c(1, ncol(train_data))]
feature_vars <- names(train_data_tmp)

# Preprocess the data
preproc <- preProcess(train_data[, feature_vars], method = c("center", "scale"))
train_data[, feature_vars] <- predict(preproc, train_data[, feature_vars])
test_data[, feature_vars] <- predict(preproc, test_data[, feature_vars])

# Logistic Regression
logi_grid <- expand.grid(parameter=c(0.001, 0.01, 0.1, 1,10,100, 1000))

# Gradient Boosting Machine
gbm_grid <- expand.grid(
  interaction.depth = c(1, 3, 5, 9, 11),
  #n.trees = c(50, 100, 200),
  n.trees = seq(50, 1000, length.out = 10),
  shrinkage = c(0.01, 0.1, 0.3, 0.5),
  n.minobsinnode = c(10, 20, 30)
)

# Naive Bayes
nb_grid <- data.frame(laplace = seq(0.0, 1.0, length.out = 10), usekernel = c(TRUE, FALSE), adjust = c(TRUE, FALSE))

# Decision Tree
dt_grid <- data.frame(cp = seq(0.001, 0.1, length.out = 10))

# Random Forest
rf_grid <- data.frame(mtry = seq(2, floor(sqrt(ncol(train_data))), by = 1))

# Support Vector Machine
svm_grid <- data.frame(C = c(0.001, 0.01, 0.1, 1, 10, 100, 1000))

# k-Nearest Neighbour
knn_grid <- expand.grid(k = 1:10)

# Define the feature selection methods to be used
feature_selection_methods <- list(
  no_feature_selection = list(
    method = "none",
    vars = NULL
  )
  # random_forest_variable_importance = list(
  #   method = "varImp",
  #   vars = NULL
  # ),
  # recursive_feature_elimination = list(
  #   method = "rfe",
  #   vars = NULL,
  #   sizes = c(2:4),
  #   rfeControl = rfeControl(functions = caretFuncs, method = "cv", number = 10)
  # ),
  # principal_component_analysis = list(
  #   method = "pca",
  #   vars = NULL,
  #   n_components = 10
  # )
)


classifiers <- list(
  logistic_regression = list(
    method = "glm",
    family = "binomial",
    tune_grid = logi_grid,
    metric = "Accuracy"
  ),
  decision_tree = list(
    method = "rpart",
    family = "binomial",
    tune_grid = dt_grid,
    metric = "Accuracy"

  ),
  random_forest = list(
    method = "rf",
    family = "binomial",
    tune_grid = rf_grid,
    metric = "Accuracy"
  ),
  gradient_boosting_machine = list(
    method = "gbm",
    family = "binomial",
    tune_grid = gbm_grid,
    metric = "Accuracy"
  ),
  naive_bayes = list(
    method = "naive_bayes",
    family = "binomial",
    tune_grid = nb_grid,
    metric = "Accuracy"
  ),
  svm = list(
    method = "svmLinear",
    family = "binomial",
    tune_grid = svm_grid,
    metric = "Accuracy"
  ),
  knn = list(
    method = "knn",
    family = "binomial",
    tune_grid = knn_grid,
    metric = "Accuracy"
  )
)

# Define function for feature selection
featureSelection <- function(train_data, label_var, feature_selection_method) {
  # Get the feature selection method
  method_params <- feature_selection_methods[[feature_selection_method]]

  if (feature_selection_method == "none") {
    selected_feature_vars <- feature_vars
  }
  # else if (feature_selection_method == "pca") {
  #   pca_model <- preProcess(train_data[, feature_vars], method = method_params$method, pcaComp = method_params$n_components)
  #   train_data_pca <- predict(pca_model, train_data[, feature_vars])
  #   #selected_feature_vars <- names(train_data_pca)
  #   selected_feature_vars <- train_data_pca
  # }
  else {
    # Unsupported method
    stop(paste("Unsupported feature selection method:", feature_selection_method))
  }

  # Return the selected feature variables
  return(selected_feature_vars)
}

# Split the data into training and validation sets for cross-validation
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE)

# evaluate <- function(test_predictions, test_data, positive = "AD") {
#   # Extract the predicted probabilities for the positive class
#   predicted_probabilities <- test_predictions$prob[, positive]
#
#   # Create a binary vector of true labels based on the predicted probabilities
#   true_labels <- as.integer(predicted_probabilities > 0.5)
#
#   # Compute the AUC and MCC based on the true labels and the positive class
#   auc <- pROC::auc(true_labels, predicted_probabilities, curve = FALSE)
#   mcc <- caret::mcc(true_labels, predicted_probabilities > 0.5)
#
#   # Return the AUC and MCC as a named list
#   return(list(auc = auc, mcc = mcc))
# }


# Perform binary classification and feature selection for each classifier and feature selection method
for (classifier_name in names(classifiers)) {
       for (fs_name in names(feature_selection_methods)) {
         classifier_params <- classifiers[[classifier_name]]
         fs_params <- feature_selection_methods[[fs_name]]

         # Perform feature selection
         selected_features <- featureSelection(train_data[, feature_vars], train_data[, label_var], fs_params$method)
         #selected_features_vars <- names(train_data)[selected_features]
         selected_features_vars <- selected_features

         if (classifier_name == "logistic") {
           classifier_model <- train(train_data[, selected_features_vars], train_data[, label_var],
                                     method = classifier_params$method,
                                     family = classifier_params$family)

           # Evaluate the performance on the training data using cross-validation
           cv_results <- train(train_data[, selected_features_vars], train_data[, label_var],
                               method = classifier_params$method,
                               family = classifier_params$family,
                               trControl = train_control,
                               tuneGrid = classifier_params$tune_grid,
                               metric = classifier_params$metric)
         }
         else {
           classifier_model <- train(train_data[, selected_features_vars], train_data[, label_var],
                                     method = classifier_params$method)

           # Evaluate the performance on the training data using cross-validation
           cv_results <- train(train_data[, selected_features_vars], train_data[, label_var],
                               method = classifier_params$method,
                               trControl = train_control,
                               tuneGrid = classifier_params$tune_grid,
                               metric = classifier_params$metric)
         }

         # Print the cross-validation results
         cat("\n", classifier_name, "with", fs_name, "feature selection (training):", "\n")
         print(cv_results$results)

         # Evaluate the performance on the test data
         test_predictions <- predict(classifier_model, newdata = test_data[, selected_features_vars])
         # Predict probabilities on test data
         test_probs <- predict(classifier_model, newdata = test_data[, selected_features_vars], type="prob")
         # auc <- pROC::roc(predictor = test_predictions, response = test_data[, label_var])$auc
         # mcc <- mltools::mcc(test_predictions, test_data[, label_var])$mcc
         # test_results <- list(auc = auc, mcc = mcc)
         # roc_obj <- pROC::roc(predictor = test_probs, response = NULL, auc = TRUE)
         # roc_obj <- pROC::roc(data = test_data, predictor = test_probs, levels = c("AD", "CTL"))
         # test_results <- auc(roc_obj)

         # Print the test results
         # cat("\n", classifier_name, "with", fs_name, "feature selection (test):")
         # print(test_results)

         output_df <- data.frame(classifier = character(),
                                 feature_selection = character(),
                                 auc = numeric(),
                                 mcc = numeric())

         # Save the results to the output dataframe
         output_df <- rbind(output_df, data.frame(
           classifier = classifier_name,
           feature_selection = fs_name
           #auc = test_results$auc,
           #mcc = test_results$mcc
         ))
       }
}

# Print the final results
cat("\nFinal results:")
print(output_df)