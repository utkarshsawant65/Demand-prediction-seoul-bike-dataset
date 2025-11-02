# CUBIST Model Implementation - TEMPORAL SPLIT (9 months train / 3 months test)
# Based on Sathishkumar V E & Yongyun Cho (2020)
# Real-world scenario: Train on 9 months, test on next 3 months

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

# Function to train CUBIST model with TEMPORAL split
train_cubist_temporal <- function(data_path = "data/processed/seoul_bike_processed.csv") {

  cat(sep_line(80), "\n")
  cat("CUBIST MODEL TRAINING - TEMPORAL SPLIT (REAL-WORLD SCENARIO)\n")
  cat("Training: First 9 months | Testing: Last 3 months\n")
  cat(sep_line(80), "\n\n")

  # Load processed data
  cat("Loading data...\n")
  df <- read_csv(data_path, show_col_types = FALSE)

  cat(sprintf("Data loaded: %d rows, %d columns\n", nrow(df), ncol(df)))

  # Convert Date column to Date type
  df$Date <- as.Date(df$Date)

  cat(sprintf("Date range: %s to %s\n", min(df$Date), max(df$Date)))

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
  dates <- df$Date

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
  dates <- dates[complete_rows]

  cat(sprintf("Clean data: %d rows\n", length(y)))

  # --- TEMPORAL SPLIT: 9 months train / 3 months test ---
  cat("\n", sep_line(60), "\n")
  cat("TEMPORAL SPLIT STRATEGY\n")
  cat(sep_line(60), "\n")

  # Calculate split date (first 9 months for training)
  min_date <- min(dates)
  max_date <- max(dates)

  # Add 9 months to min_date to get split point
  split_date <- min_date + months(9)

  cat(sprintf("Training period: %s to %s (9 months)\n", min_date, split_date - 1))
  cat(sprintf("Testing period:  %s to %s (3 months)\n", split_date, max_date))

  # Create temporal split
  train_idx <- which(dates < split_date)
  test_idx <- which(dates >= split_date)

  # Split data
  X_train <- X[train_idx, ]
  y_train <- y[train_idx]

  X_test <- X[test_idx, ]
  y_test <- y[test_idx]

  cat(sprintf("\nTraining set: %d samples\n", length(y_train)))
  cat(sprintf("Testing set: %d samples\n\n", length(y_test)))

  # Use SAME hyperparameters as found in random split version
  # This ensures fair comparison - we're testing the impact of temporal split,
  # not different hyperparameters

  cat(sep_line(60), "\n")
  cat("USING SAME HYPERPARAMETERS AS RANDOM SPLIT MODEL\n")
  cat("(For fair comparison - only difference is train/test split method)\n")
  cat(sep_line(60), "\n\n")

  # Define CUBIST parameter grid - same as random split version
  cubist_grid <- expand.grid(
    committees = c(35, 38, 41, 44, 47, 50),
    neighbors = c(1, 3, 5, 7)
  )

  cat("Parameter grid:\n")
  print(cubist_grid)
  cat("\n")

  # Set up 10-fold cross-validation with 3 repeats (same as random split)
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
  cat("BEST HYPERPARAMETERS (Temporal Split)\n")
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
    cat(sprintf("%s SET RESULTS (TEMPORAL SPLIT)\n", toupper(set_name)))
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

  # Create output directories (separate from random split results)
  dir.create("reports/figures/temporal", recursive = TRUE, showWarnings = FALSE)
  dir.create("reports/results/temporal", recursive = TRUE, showWarnings = FALSE)
  dir.create("models/temporal", recursive = TRUE, showWarnings = FALSE)

  # Save CV results plot
  pdf("reports/figures/temporal/cubist_cv_results_temporal.pdf", width = 10, height = 6)
  plot(cubist_fit, main = "CUBIST Cross-Validation Results (Temporal Split)")
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
  pdf("reports/figures/temporal/cubist_variable_importance_temporal.pdf", width = 10, height = 8)
  plot(varImp(cubist_fit), top = 15, main = "CUBIST Variable Importance - Temporal Split (Top 15)")
  dev.off()

  # Create prediction plots
  pdf("reports/figures/temporal/cubist_predictions_comparison_temporal.pdf", width = 14, height = 6)
  par(mfrow = c(1, 2))

  # Training set
  plot(y_train, y_train_pred,
       main = "CUBIST: Training Set (9 months)",
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
       main = "CUBIST: Testing Set (3 months)",
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
  pdf("reports/figures/temporal/cubist_residuals_temporal.pdf", width = 14, height = 6)
  par(mfrow = c(1, 2))

  # Training residuals
  train_residuals <- y_train - y_train_pred
  plot(y_train_pred, train_residuals,
       main = "Residuals: Training Set (9 months)",
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
       main = "Residuals: Testing Set (3 months)",
       xlab = "Predicted Bike Count",
       ylab = "Residuals",
       pch = 16,
       col = rgb(0.8, 0.2, 0.2, 0.4),
       cex = 0.6)
  abline(h = 0, col = "red", lwd = 2, lty = 2)
  grid()

  dev.off()

  # Save model
  cat("\nSaving model to models/temporal/cubist_temporal.rds...\n")
  saveRDS(cubist_fit, "models/temporal/cubist_temporal.rds")

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

  write_csv(results_df, "reports/results/temporal/cubist_predictions_temporal.csv")

  # Save metrics summary
  metrics_summary <- data.frame(
    Set = c("Training", "Testing"),
    Split_Method = c("Temporal (9mo/3mo)", "Temporal (9mo/3mo)"),
    R2 = c(train_metrics$R2, test_metrics$R2),
    RMSE = c(train_metrics$RMSE, test_metrics$RMSE),
    MAE = c(train_metrics$MAE, test_metrics$MAE),
    CV = c(train_metrics$CV, test_metrics$CV),
    MAPE = c(train_metrics$MAPE, test_metrics$MAPE)
  )

  write_csv(metrics_summary, "reports/results/temporal/cubist_metrics_temporal.csv")

  # Save best parameters
  best_params_df <- data.frame(
    Parameter = names(cubist_fit$bestTune),
    Value = unlist(cubist_fit$bestTune)
  )

  write_csv(best_params_df, "reports/results/temporal/cubist_best_params_temporal.csv")

  # Compare temporal split with random split
  cat("\n", sep_line(80), "\n")
  cat("TEMPORAL SPLIT VS RANDOM SPLIT COMPARISON\n")
  cat(sep_line(80), "\n")
  cat("Random Split (75/25 - Paper Replication):\n")
  cat("  [Results will be compared after both models are trained]\n\n")
  cat("Temporal Split (9mo/3mo - Real-World Scenario):\n")
  cat(sprintf("  Training R² = %.4f (%.2f%%)\n", train_metrics$R2, train_metrics$R2 * 100))
  cat(sprintf("  Testing R²  = %.4f (%.2f%%)\n", test_metrics$R2, test_metrics$R2 * 100))
  cat(sprintf("  Testing RMSE = %.2f\n", test_metrics$RMSE))
  cat(sprintf("  Testing MAE  = %.2f\n", test_metrics$MAE))
  cat(sprintf("  Testing CV   = %.2f%%\n", test_metrics$CV))
  cat("\n")
  cat("NOTE: Temporal split typically shows lower test performance\n")
  cat("      because it tests on truly unseen future data.\n")
  cat("      This is a more realistic evaluation of model performance.\n")
  cat(sep_line(80), "\n")

  return(list(
    model = cubist_fit,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    best_params = cubist_fit$bestTune,
    split_date = split_date
  ))
}

# Main execution
cat("\n")
cat(sep_line(60, "#"), "\n")
cat("# CUBIST MODEL - TEMPORAL SPLIT #\n")
cat(sep_line(60, "#"), "\n\n")

# Train model
results <- train_cubist_temporal()

# Stop parallel cluster
stopCluster(cl)

cat("\n")
cat(sep_line(60), "\n")
cat("EXECUTION COMPLETE\n")
cat(sep_line(60), "\n\n")
cat("Results saved in reports/results/temporal/\n")
cat("Figures saved in reports/figures/temporal/\n")
cat("Model saved in models/temporal/\n\n")
