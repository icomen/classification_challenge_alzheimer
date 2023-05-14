



         # Define the feature selection methods to be used
feature_selection_methods <- list(
  variance_threshold = list(method = "nearZeroVar", freqCut = 0.95),
  correlation_based = list(method = "cor", cutoff = 0.5)
)


         # Define the featureSelection function
featureSelection <- function(train_data, label_var, method_name) {
  # Get the feature selection method
  method_params <- feature_selection_methods[[method_name]]

  if (method_name == "nearZeroVar") {
    # Variance threshold method
    selected_feature_vars <- names(train_data)[!caret::nearZeroVar(train_data[, -which(names(train_data) == label_var)],
                                                                   freqCut = method_params$freqCut)]
  }
  else if (method_name == "cor") {
    # Correlation-based method
    cor_mat <- cor(train_data[, -which(names(train_data) == label_var)])
    high_cor_vars <- findCorrelation(cor_mat, cutoff = method_params$cutoff, verbose = FALSE)
    selected_feature_vars <- names(train_data)[-which(names(train_data) == label_var)][high_cor_vars]

  }
  else {
    # Unsupported method
    stop(paste("Unsupported feature selection method:", method_name))
  }

  # Return the selected feature variables
  return(selected_feature_vars)
}


                  # Perform feature selection
         selected_features <- featureSelection(train_data[, feature_vars], train_data[, label_var], method = fs_params$method)
         selected_feature_vars <- names(train_data)[selected_features]