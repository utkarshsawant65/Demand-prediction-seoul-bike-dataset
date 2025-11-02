# Optimized CUBIST Model Implementation
# Based on Sathishkumar V E & Yongyun Cho (2020)
# FULLY CORRECTED VERSION - All string operations fixed

# Install required packages if not already installed
packages <- c("Cubist", "caret", "doParallel", "tidyverse", "readr", "MLmetrics", 
              "future", "foreach", "iterators", "lattice", "ggplot2")

# Check which packages are missing
installed_packages <- packages %in% rownames(installed.packages())

if (any(installed_packages == FALSE)) {
  cat("Installing missing packages...\n")
  cat("This may take a few minutes...\n\n")
  
  # Install with dependencies
  install.packages(packages[!installed_packages], 
                   dependencies = TRUE,
                   repos = "https://cran.rstudio.com/")
  
  cat("\nPackage installation complete!\n\n")
}

# Load libraries
library(Cubist)
library(caret)
library(doParallel)
library(tidyverse)
library(readr)
library(MLmetrics)

# Set up parallel processing
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)

# Set seed for reproducibility
set.seed(42)

# Helper function for separators
sep_line <- function(n = 80, char = "=") {
  paste(rep(char, n), collapse="")
}

# Function to train CUBIST model
train_cubist_seoul_bike <- function(data_path = "data/processed/seoul_bike_processed.csv") {
  
  cat(sep_line(80), "\n")
  cat("CUBIST MODEL TRAINING FOR SEOUL BIKE DATA\n")
  cat(sep_line(80), "\n\n")
  
  # Load processed data
  cat("Loading data...\n")
  df <- read_csv(data_path, show_col_types = FALSE)
  
  cat(sprintf("Data loaded: %d rows, %d columns\n", nrow(df), ncol(df)))
  
  # Define target and features
  target_col <- "Count"
  
  # Exclude non-predictive columns
  exclude_cols <- c(target_col, "Date", "Year", "Month", "Day", "DayOfWeek")
  feature_cols <- setdiff(names(df), exclude_cols)
  
  cat(sprintf("\nFeatures used (%d): %s\n", 
              length(feature_cols), 
              paste(head(feature_cols, 10), collapse=", ")))
  
  # Prepare X and y
  X <- df %>% select(all_of(feature_cols))
  y <- df[[target_col]]
  
  # Handle categorical variables - convert to factors
  categorical_cols <- c("Holiday", "Fday", "WeekStatus", "DayName", "Season")
  for(col in categorical_cols) {
    if(col %in% names(X)) {
      # Check if column exists as dummy variables
      dummy_pattern <- paste0("^", col, "_")
      dummy_cols <- grep(dummy_pattern, names(X), value = TRUE)
      
      if(length(dummy_cols) == 0 && col %in% names(X)) {
        # Convert to factor if not already dummy encoded
        X[[col]] <- as.factor(X[[col]])
      }
    }
  }
  
  # Remove rows with NA
  complete_rows <- complete.cases(X, y)
  X <- X[complete_rows, ]
  y <- y[complete_rows]
  
  cat(sprintf("Clean data: %d rows\n", length(y)))
  
  # Time-based split (75-25) - IMPORTANT: maintain temporal order
  # --- START REPLACEMENT BLOCK ---
#
# Create the 75/25 random split as per the paper 
# We use createDataPartition from the caret package

cat("Creating 75/25 random split (as per paper)...\n")

# Combine X and y for safe splitting
full_data <- cbind(X, Count = y)

# Create random indices for the 75% training set
train_idx <- createDataPartition(
  y = full_data$Count,
  p = 0.75,  # 75% for training
  list = FALSE
)

# Create training and testing sets
train_data <- full_data[train_idx, ]
test_data  <- full_data[-train_idx, ]

# Separate X and y again
X_train <- train_data %>% select(-Count)
y_train <- train_data$Count

X_test  <- test_data %>% select(-Count)
y_test  <- test_data$Count

cat(sprintf("\nTraining set: %d samples (Paper: 6571)\n", length(y_train)))
cat(sprintf("Testing set: %d samples (Paper: 2189)\n\n", length(y_test)))

# --- END REPLACEMENT BLOCK ---
  
  cat(sprintf("\nTraining set: %d samples\n", length(y_train)))
  cat(sprintf("Testing set: %d samples\n\n", length(y_test)))
  
  # Define CUBIST parameter grid - based on paper
  cubist_grid <- expand.grid(
    committees = c(35, 38, 41, 44, 47, 50),
    neighbors = c(1, 3, 5, 7)
  )
  
  cat("Parameter grid:\n")
  print(cubist_grid)
  cat("\n")
  
  # Set up 10-fold cross-validation with 3 repeats (as per paper)
  cat("Setting up 10-fold cross-validation with 3 repeats...\n")
  ctrl <- trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 3,
    verboseIter = FALSE,
    allowParallel = TRUE,
    savePredictions = "final"
  )
  
  # Train CUBIST model
  cat("\nTraining CUBIST model with grid search...\n")
  cat("This may take several minutes...\n\n")
  
  start_time <- Sys.time()
  
  cubist_fit <- train(
    x = X_train,
    y = y_train,
    method = "cubist",
    trControl = ctrl,
    tuneGrid = cubist_grid,
    metric = "RMSE"
  )
  
  end_time <- Sys.time()
  training_time <- difftime(end_time, start_time, units = "mins")
  
  cat(sprintf("\nTraining completed in %.2f minutes\n", training_time))
  
  # Print best parameters
  cat("\n", sep_line(60), "\n")
  cat("BEST HYPERPARAMETERS\n")
  cat(sep_line(60), "\n")
  print(cubist_fit$bestTune)
  cat("\n")
  
  # Make predictions
  cat("Making predictions...\n")
  y_train_pred <- predict(cubist_fit, X_train)
  y_test_pred <- predict(cubist_fit, X_test)
  
  # Calculate comprehensive metrics
  calculate_metrics <- function(y_true, y_pred, set_name) {
    
    # R-squared
    r2 <- R2_Score(y_pred, y_true)
    
    # RMSE
    rmse <- RMSE(y_pred, y_true)
    
    # MAE
    mae <- MAE(y_pred, y_true)
    
    # Coefficient of Variation
    cv <- (rmse / mean(y_true)) * 100
    
    # Additional metrics
    mape <- MAPE(y_pred, y_true) * 100
    
    cat("\n", sep_line(60), "\n")
    cat(sprintf("%s SET RESULTS\n", toupper(set_name)))
    cat(sep_line(60), "\n")
    cat(sprintf("R²:    %.4f (%.2f%%)\n", r2, r2 * 100))
    cat(sprintf("RMSE:  %.2f\n", rmse))
    cat(sprintf("MAE:   %.2f\n", mae))
    cat(sprintf("CV:    %.2f%%\n", cv))
    cat(sprintf("MAPE:  %.2f%%\n", mape))
    cat(sep_line(60), "\n")
    
    return(list(
      R2 = r2,
      RMSE = rmse,
      MAE = mae,
      CV = cv,
      MAPE = mape
    ))
  }
  
  # Calculate metrics
  train_metrics <- calculate_metrics(y_train, y_train_pred, "Training")
  test_metrics <- calculate_metrics(y_test, y_test_pred, "Testing")
  
  # Create output directories
  dir.create("reports/figures", recursive = TRUE, showWarnings = FALSE)
  dir.create("reports/results", recursive = TRUE, showWarnings = FALSE)
  dir.create("models", recursive = TRUE, showWarnings = FALSE)
  
  # Save CV results plot
  pdf("reports/figures/cubist_cv_results.pdf", width = 10, height = 6)
  plot(cubist_fit, main = "CUBIST Cross-Validation Results")
  dev.off()
  
  # Variable importance
  cat("\n", sep_line(60), "\n")
  cat("TOP 15 MOST IMPORTANT VARIABLES\n")
  cat(sep_line(60), "\n")
  var_imp <- varImp(cubist_fit)
  top_vars <- head(var_imp$importance[order(-var_imp$importance$Overall), , drop = FALSE], 15)
  print(top_vars)
  cat("\n")
  
  # Save variable importance plot
  pdf("reports/figures/cubist_variable_importance.pdf", width = 10, height = 8)
  plot(varImp(cubist_fit), top = 15, main = "CUBIST Variable Importance (Top 15)")
  dev.off()
  
  # Create prediction plots
  pdf("reports/figures/cubist_predictions_comparison.pdf", width = 14, height = 6)
  par(mfrow = c(1, 2))
  
  # Training set
  plot(y_train, y_train_pred,
       main = "CUBIST: Training Set",
       xlab = "Actual Bike Count",
       ylab = "Predicted Bike Count",
       pch = 16,
       col = rgb(0, 0.4, 0.8, 0.4),
       cex = 0.6)
  abline(0, 1, col = "red", lwd = 2, lty = 2)
  legend("topleft",
         legend = sprintf("R² = %.4f\nRMSE = %.2f", train_metrics$R2, train_metrics$RMSE),
         bty = "n",
         cex = 1.1)
  grid()
  
  # Testing set
  plot(y_test, y_test_pred,
       main = "CUBIST: Testing Set",
       xlab = "Actual Bike Count",
       ylab = "Predicted Bike Count",
       pch = 16,
       col = rgb(0.8, 0.2, 0.2, 0.4),
       cex = 0.6)
  abline(0, 1, col = "red", lwd = 2, lty = 2)
  legend("topleft",
         legend = sprintf("R² = %.4f\nRMSE = %.2f", test_metrics$R2, test_metrics$RMSE),
         bty = "n",
         cex = 1.1)
  grid()
  
  dev.off()
  
  # Residual plots
  pdf("reports/figures/cubist_residuals.pdf", width = 14, height = 6)
  par(mfrow = c(1, 2))
  
  # Training residuals
  train_residuals <- y_train - y_train_pred
  plot(y_train_pred, train_residuals,
       main = "Residuals: Training Set",
       xlab = "Predicted Bike Count",
       ylab = "Residuals",
       pch = 16,
       col = rgb(0, 0.4, 0.8, 0.4),
       cex = 0.6)
  abline(h = 0, col = "red", lwd = 2, lty = 2)
  grid()
  
  # Testing residuals
  test_residuals <- y_test - y_test_pred
  plot(y_test_pred, test_residuals,
       main = "Residuals: Testing Set",
       xlab = "Predicted Bike Count",
       ylab = "Residuals",
       pch = 16,
       col = rgb(0.8, 0.2, 0.2, 0.4),
       cex = 0.6)
  abline(h = 0, col = "red", lwd = 2, lty = 2)
  grid()
  
  dev.off()
  
  # Save model
  cat("\nSaving model to models/cubist_seoul_bike.rds...\n")
  saveRDS(cubist_fit, "models/cubist_seoul_bike.rds")
  
  # Save predictions
  max_length <- max(length(y_train), length(y_test))
  results_df <- data.frame(
    actual_train = c(y_train, rep(NA, max_length - length(y_train))),
    predicted_train = c(y_train_pred, rep(NA, max_length - length(y_train_pred))),
    residuals_train = c(train_residuals, rep(NA, max_length - length(train_residuals))),
    actual_test = c(y_test, rep(NA, max_length - length(y_test))),
    predicted_test = c(y_test_pred, rep(NA, max_length - length(y_test_pred))),
    residuals_test = c(test_residuals, rep(NA, max_length - length(test_residuals)))
  )
  
  write_csv(results_df, "reports/results/cubist_predictions.csv")
  
  # Save metrics summary
  metrics_summary <- data.frame(
    Set = c("Training", "Testing"),
    R2 = c(train_metrics$R2, test_metrics$R2),
    RMSE = c(train_metrics$RMSE, test_metrics$RMSE),
    MAE = c(train_metrics$MAE, test_metrics$MAE),
    CV = c(train_metrics$CV, test_metrics$CV),
    MAPE = c(train_metrics$MAPE, test_metrics$MAPE)
  )
  
  write_csv(metrics_summary, "reports/results/cubist_metrics_summary.csv")
  
  
  
  # Compare with paper results
  cat("\n", sep_line(60), "\n")
  cat("COMPARISON WITH PAPER RESULTS\n")
  cat(sep_line(60), "\n")
  cat("Paper (Seoul Bike Testing Set):\n")
  cat("  R² = 0.9500 (95.00%)\n")
  cat("  RMSE = 139.64\n")
  cat("  MAE = 78.45\n")
  cat("  CV = 19.81%\n")
  cat("Your Results (Testing Set):\n")
  cat(sprintf("  R² = %.4f (%.2f%%)\n", test_metrics$R2, test_metrics$R2 * 100))
  cat(sprintf("  RMSE = %.2f\n", test_metrics$RMSE))
  cat(sprintf("  MAE = %.2f\n", test_metrics$MAE))
  cat(sprintf("  CV = %.2f%%\n", test_metrics$CV))
  cat("\n")
  
  
  return(list(
    model = cubist_fit,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    best_params = cubist_fit$bestTune
  ))
}

# Main execution
cat("\n")
cat(sep_line(60, "#"), "\n")
cat("# SEOUL BIKE CUBIST MODEL TRAINING #\n")
cat(sep_line(60, "#"), "\n\n")

# Train model
results <- train_cubist_seoul_bike()

# Stop parallel cluster
stopCluster(cl)

cat("\n")
cat(sep_line(60), "\n")
cat("EXECUTION COMPLETE\n")
cat(sep_line(60), "\n\n")