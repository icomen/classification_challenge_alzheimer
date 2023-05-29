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

  train_file <- paste0(problem_name, "/", problem_name, "train.csv")
  test_file <- paste0(problem_name, "/", problem_name, "test.csv")
  train_data <- read.csv(train_file)
  test_data <- read.csv(test_file)

  return(list(train_data = train_data, test_data = test_data, problem_name = problem_name))
}

# Load the training and test datasets
data <- chooseProblem()
train_data <- data$train_data
test_data <- data$test_data

# Define the label and feature variables
label_var <- names(train_data)[ncol(train_data)]
train_data_tmp <- train_data[, -c(1, ncol(train_data))]
feature_vars <- names(train_data_tmp)

# Preprocess the data
preproc <- preProcess(train_data[, feature_vars], method = c("center", "scale"))
train_data[, feature_vars] <- predict(preproc, train_data[, feature_vars])
test_data[, feature_vars] <- predict(preproc, test_data[, feature_vars])


# Logistic Regression tuning grid
logi_grid <- data.frame(parameter=c(0.001, 0.01, 0.1, 1,10,100, 1000))

# Gradient Boosting Machine tuning grid
gbm_grid <- expand.grid(
  interaction.depth = c(1, 3, 5, 9),
  n.trees = c(50, 100, 200),
  shrinkage = c(0.01, 0.1, 0.3, 0.5),
  n.minobsinnode = c(10, 20, 30)
)

# Naive Bayes tune grid
nb_grid <- data.frame(
  laplace = seq(0.0, 1.0, length.out = 10),
  usekernel = c(TRUE, FALSE),
  adjust = c(TRUE, FALSE)
)

# Decision Tree tune grid
dt_grid <- data.frame(cp = seq(0.001, 0.1, length.out = 10))

# Random Forest tune grid
rf_grid <- data.frame(mtry = seq(2, floor(sqrt(ncol(train_data))), by = 1))

# Support Vector Machine tune grid
svm_grid <- data.frame(C = c(0.001, 0.01, 0.1, 1, 10, 100, 1000))

# k-Nearest Neighbour tune grid
knn_grid <- data.frame(k = 1:10)

# eXtreme Gradient Boosting Trees tune grid
xgb_grid <- expand.grid(
  nrounds = c(50, 100, 150),
  max_depth = c(3, 6, 9),
  eta = c(0.1, 0.3, 0.5),
  gamma = c(0, 0.1, 0.5),
  subsample = c(0.5, 0.75, 1),
  colsample_bytree = c(0.5, 0.75, 1),
  min_child_weight = c(1, 3, 5)
)

# Neural Network tune grid
nnet_grid <- expand.grid(
  size = c(1:3),
  decay = c(0, 0.1, 0.01, 0.001, 0.0001)
)


# Define the feature selection methods to be used
feature_selection_methods <- list(
  no_feature_selection = list(
    method = "none",
    vars = NULL
  ),
  principal_component_analysis = list(
    method = "pca",
    vars = NULL,
    thresh = 0.95
  )
  # elastic_net = list(
  #   method = "enet",
  #   vars = NULL,
  #   family = "binomial",
  #   lambda = seq(0.001, 1, by = 0.001),
  #   type.measure = "class"
  # )
)

# Define the classifiers and their hyperparameters
classifiers <- list(
  neural_network = list(
    method = "nnet",
    tune_grid = nnet_grid,
    metric = "Accuracy"
  ),
  # logistic_regression = list(
  #   method = "glm",
  #   family = "binomial",
  #   tune_grid = logi_grid,
  #   metric = "Accuracy"
  # ),
  decision_tree = list(
    method = "rpart",
    family = "binomial",
    tune_grid = dt_grid,
    metric = "Accuracy"
  ),
  # random_forest = list(
  #   method = "rf",
  #   family = "binomial",
  #   tune_grid = rf_grid,
  #   metric = "Accuracy"
  # ),
  naive_bayes = list(
    method = "naive_bayes",
    family = "binomial",
    tune_grid = nb_grid,
    metric = "Accuracy"
  ),
  knn = list(
    method = "knn",
    family = "binomial",
    tune_grid = knn_grid,
    metric = "Accuracy"
  )
  # svm = list(
  #   method = "svmLinear",
  #   family = "binomial",
  #   tune_grid = svm_grid,
  #   metric = "Accuracy"
  # ),
  # gradient_boosting_machine = list(
  #   method = "gbm",
  #   family = "binomial",
  #   tune_grid = gbm_grid,
  #   metric = "Accuracy"
  # ),
  # xgbtree = list(
  #   method = "xgbTree",
  #   tune_grid = xgb_grid,
  #   metric = "Accuracy"
  # )
)

# Define function for feature selection
featureSelection <- function(train_data, label_var, feature_selection_method) {
    if (feature_selection_method == "none") {
      selected_feature_vars <- feature_vars
      return(selected_feature_vars)
    }
    else if (feature_selection_method == "pca") {
      pca_model <- preProcess(train_data[, feature_vars],
                              method = feature_selection_methods$principal_component_analysis$method,
                              thresh = feature_selection_methods$principal_component_analysis$thresh)
      # Number of components
      # n_pcs <- pca_model$numComp
      # Get the index of the most important feature on EACH component
      most_important <- apply(abs(pca_model$rotation), 2, which.max)
      # Initial feature names
      initial_feature_names <- names(train_data[, feature_vars])
      # Get the names of the most important features
      most_important_names <- initial_feature_names[most_important]
      # Create a dictionary of PC names and their corresponding important features
      # dic <- setNames(most_important_names, paste0("PC", 1:n_pcs))
      selected_feature_vars <- most_important_names
      return(selected_feature_vars)
    }
    # else if (feature_selection_method == "enet") {
    #   x <- train_data[, feature_vars]
    #   y <- train_data[, label_var]
    #   enet_model <- cv.glmnet(as.matrix(x), y,
    #                           family = feature_selection_methods$elastic_net$family,
    #                           type.measure = feature_selection_methods$elastic_net$type.measure,
    #                           lambda = feature_selection_methods$elastic_net$lambda)
    #   # Extract the coefficients from the best lambda value
    #   coef <- coef(enet_model, s = "lambda.min")
    #   selected_feature_vars <- rownames(coef)[coef$x != 0][-1]
    #   return(selected_feature_vars)
    # }
    else {
      # Unsupported method
      stop(paste("Unsupported feature selection method:", feature_selection_method))
    }
}

# customSummary <- function(data, lev = NULL, model = NULL) {
#   # Calculate metrics
#   confusion_matrix <- table(data$obs, data$pred)
#   true_positives <- confusion_matrix[2, 2]
#   false_negatives <- confusion_matrix[2, 1]
#   true_negatives <- confusion_matrix[1, 1]
#   false_positives <- confusion_matrix[1, 2]
#
#   sensitivity <- true_positives / (true_positives + false_negatives)
#   specificity <- true_negatives / (true_negatives + false_positives)
#
#   precision <- true_positives / (true_positives + false_positives)
#   f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
#   balanced_accuracy <- (sensitivity + specificity) / 2
#
#   # Calculate AUC
#   predicted_probs <- predict(model, newdata = data, type = "prob")
#   auc <- pROC::auc(data$obs, predicted_probs)
#
#   # Calculate MCC
#   mcc <- mltools::mcc(data$obs, data$pred)
#
#   # Calculate accuracy
#   accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
#
#   # Return metrics
#   out <- c(AUC = auc, Accuracy = accuracy, Sensitivity = sensitivity,
#            Specificity = specificity, Precision = precision,
#            F1_Score = f1_score, Balanced_Accuracy = balanced_accuracy,
#            MCC = mcc)
#   names(out) <- c("AUC", "Accuracy", "Sensitivity", "Specificity",
#                   "Precision", "F1_Score", "Balanced_Accuracy", "MCC")
#   out
# }

# Split the data into training and validation sets for cross-validation
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE)
#train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = customSummary)

# Perform binary classification and feature selection for each classifier and feature selection method
for (classifier_name in names(classifiers)) {
       for (fs_name in names(feature_selection_methods)) {
         classifier_params <- classifiers[[classifier_name]]
         fs_params <- feature_selection_methods[[fs_name]]

         # Perform feature selection
         selected_features <- featureSelection(train_data[, feature_vars], train_data[, label_var], fs_params$method)
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
         # Calculate Sensitivity and Specificity
         actual_labels <- train_data[, label_var]
         predicted_classes_train <- predict(classifier_model, newdata = train_data[, selected_features_vars])
         confusion_matrix <- table(actual_labels, predicted_classes_train)
         true_positives <- confusion_matrix[2, 2]
         false_negatives <- confusion_matrix[2, 1]
         true_negatives <- confusion_matrix[1, 1]
         false_positives <- confusion_matrix[1, 2]

         sensitivity <- true_positives / (true_positives + false_negatives)
         specificity <- true_negatives / (true_negatives + false_positives)

         # Calculate Precision, F1 Score, and Balanced Accuracy (BA)
         precision <- true_positives / (true_positives + false_positives)
         f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
         balanced_accuracy <- (sensitivity + specificity) / 2

         if (data$problem_name == "MCICTL") {
           # Calculate AUC
           predicted_probs_train <- predict(classifier_model, newdata = train_data[, selected_features_vars], type = "prob")[, 2]
           auc <- pROC::auc(actual_labels, predicted_probs_train)
         }
         else {
           # Calculate AUC
           predicted_probs_train <- predict(classifier_model, newdata = train_data[, selected_features_vars], type = "prob")[, 1]
           auc <- pROC::auc(actual_labels, predicted_probs_train)
         }


         # Calculate MCC
         mcc <- mltools::mcc(as.vector(actual_labels), as.vector(predicted_classes_train))
         accuracy <- (true_positives + true_negatives) / sum(confusion_matrix)


         predicted_labels <- predict(classifier_model, newdata = test_data[, selected_features_vars])
         predicted_probs <- predict(classifier_model, newdata = test_data[, selected_features_vars], type = "prob")


         # Print the cross-validation results
         cat("\n", classifier_name, "with", fs_name, "feature selection (training):", "\n")
         # Add other metrics columns to cv_results$results
         cv_results$results$Acc <- accuracy
         cv_results$results$Sens <- sensitivity
         cv_results$results$Spec <- specificity
         cv_results$results$Prec <- precision
         cv_results$results$F1 <- f1_score
         cv_results$results$MCC <- mcc
         cv_results$results$AUC <- auc
         cv_results$results$BA <- balanced_accuracy

         print(cv_results$results)


         # Write the cross-validation results to a text file
         file_name <- "cv_results.txt"  # Specify the file name and path as needed
         # write.table(cv_results$results, file = file_name, append = TRUE, sep = "\t")


         model_name <- paste(classifier_name, fs_name, "model", sep = "_")
         # save(classifier_model, file = paste0(model_name, ".RData"))

         # Construct the output filename
         output_res <- paste0("50411_castrechini_", classifier_name, "_", fs_name, "_", data$problem_name, "res.csv")

         output_feat <- paste0("50411_castrechini_", classifier_name, "_", fs_name, "_", data$problem_name, "feat.csv")

         # Write the predictions to a CSV file
         # write.csv(cbind(test_data$ID, predicted_labels, predicted_probs), output_res, row.names = FALSE)
         # write.csv(train_data[, selected_features_vars], output_feat, row.names = FALSE)

       }
}