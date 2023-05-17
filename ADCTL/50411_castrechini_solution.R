library(caret)
library(pROC)
library(mltools)
library(magrittr)
library(randomForest)
library(MASS)

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
    thresh = 0.97
  )
  # lda_feature_selection = list(
  #   method = "lda",
  #   vars = NULL
  # )
)

# Define the classifiers and their hyperparameters
classifiers <- list(
  neural_network = list(
    method = "nnet",
    tune_grid = nnet_grid,
    metric = "Accuracy"
  ),
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
  ),
  svm = list(
    method = "svmLinear",
    family = "binomial",
    tune_grid = svm_grid,
    metric = "Accuracy"
  ),
  gradient_boosting_machine = list(
    method = "gbm",
    family = "binomial",
    tune_grid = gbm_grid,
    metric = "Accuracy"
  ),
  xgbtree = list(
    method = "xgbTree",
    tune_grid = xgb_grid,
    metric = "Accuracy"
  )
)

# Define function for feature selection
featureSelection <- function(train_data, label_var, feature_selection_method) {
  # Get the feature selection method
  #method_params <- feature_selection_methods[[feature_selection_method]]

  if (feature_selection_method == "none") {
    selected_feature_vars <- feature_vars
    return(selected_feature_vars)
  }
  else if (feature_selection_method == "pca") {
    pca_model <- preProcess(train_data[, feature_vars], method = feature_selection_methods$principal_component_analysis$method,
                            thresh = feature_selection_methods$principal_component_analysis$thresh)
    train_data_pca <- predict(pca_model, train_data[, feature_vars])
    test_data_pca <- predict(pca_model, test_data[, feature_vars])
    #selected_feature_vars <- names(train_data_pca)
    selected_feature_vars <- train_data_pca
    return(list(selected_feature_vars = selected_feature_vars, test_data_pca = test_data_pca))
  }
  else if (feature_selection_method == "lda_feature_selection") {
    # Perform LDA feature selection
    lda_model <- lda(train_data[, feature_vars], train_data[, label_var])
    train_data_lda <- as.data.frame(predict(lda_model)$x)
    test_data_lda <- as.data.frame(predict(lda_model, newdata = test_data[, feature_vars])$x)
    selected_feature_vars <- train_data_lda
    return(list(selected_feature_vars = selected_feature_vars, test_data_lda = test_data_lda))
  }
  else {
    # Unsupported method
    stop(paste("Unsupported feature selection method:", feature_selection_method))
  }
}

# Split the data into training and validation sets for cross-validation
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE)

# Perform binary classification and feature selection for each classifier and feature selection method
for (classifier_name in names(classifiers)) {
       for (fs_name in names(feature_selection_methods)) {
         classifier_params <- classifiers[[classifier_name]]
         fs_params <- feature_selection_methods[[fs_name]]

         if(fs_name == "no_feature_selection") {
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
             # Calculate AUC
             predicted_probs_train <- predict(classifier_model, newdata = train_data[, selected_features_vars], type = "prob")[, 2]
             actual_labels <- train_data[, label_var]
             auc <- pROC::roc(actual_labels, predicted_probs_train)$auc
             # Calculate MCC
             predicted_classes_train <- predict(classifier_model, newdata = train_data[, selected_features_vars])
             mcc <- mltools::mcc(as.vector(actual_labels), as.vector(predicted_classes_train))
           }
           predicted_labels <- predict(classifier_model, newdata = test_data[, selected_features_vars])
           predicted_probs <- predict(classifier_model, newdata = test_data[, selected_features_vars], type = "prob")
         }
         else if (fs_name == "principal_component_analysis") {
           # Perform feature selection
           selected_features <- featureSelection(train_data[, feature_vars], train_data[, label_var], fs_params$method)
           selected_features_vars <- selected_features$selected_feature_vars
           test_data_pca <- selected_features$test_data_pca
           if (classifier_name == "logistic") {
             classifier_model <- train(selected_features_vars, train_data[, label_var],
                                       method = classifier_params$method,
                                       family = classifier_params$family)
             # Evaluate the performance on the training data using cross-validation
             cv_results <- train(selected_features_vars, train_data[, label_var],
                                 method = classifier_params$method,
                                 family = classifier_params$family,
                                 trControl = train_control,
                                 tuneGrid = classifier_params$tune_grid,
                                 metric = classifier_params$metric)
           }
           else {
             classifier_model <- train(selected_features_vars, train_data[, label_var],
                                       method = classifier_params$method)
             # Evaluate the performance on the training data using cross-validation
             cv_results <- train(selected_features_vars, train_data[, label_var],
                                 method = classifier_params$method,
                                 trControl = train_control,
                                 tuneGrid = classifier_params$tune_grid,
                                 metric = classifier_params$metric)
             # Calculate AUC
             predicted_probs_train <- predict(classifier_model, newdata = selected_features_vars, type = "prob")[, 2]
             actual_labels <- train_data[, label_var]
             auc <- pROC::roc(actual_labels, predicted_probs_train)$auc
             # Calculate MCC
             predicted_classes_train <- predict(classifier_model, newdata = selected_features_vars)
             mcc <- mltools::mcc(as.vector(actual_labels), as.vector(predicted_classes_train))
           }
           predicted_labels <- predict(classifier_model, newdata = test_data_pca)
           predicted_probs <- predict(classifier_model, newdata = test_data_pca, type = "prob")
         }
         # else if (fs_name == "lda_feature_selection") {
         #   # Perform feature selection
         #   selected_features <- featureSelection(train_data[, feature_vars], train_data[, label_var], fs_params$method)
         #   selected_features_vars <- selected_features$selected_feature_vars
         #   test_data_lda <- selected_features$test_data_lda
         #   if (classifier_name == "logistic") {
         #     classifier_model <- train(selected_features_vars, train_data[, label_var],
         #                               method = classifier_params$method,
         #                               family = classifier_params$family)
         #     # Evaluate the performance on the training data using cross-validation
         #     cv_results <- train(selected_features_vars, train_data[, label_var],
         #                         method = classifier_params$method,
         #                         family = classifier_params$family,
         #                         trControl = train_control,
         #                         tuneGrid = classifier_params$tune_grid,
         #                         metric = classifier_params$metric)
         #   }
         #   else {
         #     classifier_model <- train(selected_features_vars, train_data[, label_var],
         #                               method = classifier_params$method)
         #     # Evaluate the performance on the training data using cross-validation
         #     cv_results <- train(selected_features_vars, train_data[, label_var],
         #                         method = classifier_params$method,
         #                         trControl = train_control,
         #                         tuneGrid = classifier_params$tune_grid,
         #                         metric = classifier_params$metric)
         #     # Calculate AUC
         #     predicted_probs_train <- predict(classifier_model, newdata = selected_features_vars, type = "prob")[, 1]
         #     actual_labels <- train_data[, label_var]
         #     auc <- pROC::roc(actual_labels, predicted_probs_train)$auc
         #     # Calculate MCC
         #     predicted_classes_train <- predict(classifier_model, newdata = selected_features_vars)
         #     mcc <- mltools::mcc(as.vector(actual_labels), as.vector(predicted_classes_train))
         #   }
         #   predicted_labels <- predict(classifier_model, newdata = test_data_lda)
         #   predicted_probs <- predict(classifier_model, newdata = test_data_lda, type = "prob")
         # }

         # Print the cross-validation results
         cat("\n", classifier_name, "with", fs_name, "feature selection (training):", "\n")
         # Add MCC and AUC columns to cv_results$results
         cv_results$results$MCC <- mcc
         cv_results$results$AUC <- auc
         print(cv_results$results)


         # Write the cross-validation results to a text file
         #file_name <- "cv_results.txt"  # Specify the file name and path as needed
         #write.table(cv_results$results, file = file_name, append = TRUE, sep = "\t")

         # Save the trained model
         model_path <- paste0("/", data$problem_name, "/", "models", "/")

         model_name <- paste(classifier_name, fs_name, "model", sep = "_")
         #save(classifier_model, file = paste0(model_name, ".RData"))


         # Define the desired file path
         output_path <- paste0("/", data$problem_name, "/", "csv", "/")

         # Construct the output filename
         output_filename <- paste0("50411_castrechini_", classifier_name, "_", fs_name, "_ADCTLres.csv")

         # Write the predictions to a CSV file
         write.csv(cbind(test_data$ID, predicted_labels, predicted_probs), output_filename, row.names = FALSE)

       }
}