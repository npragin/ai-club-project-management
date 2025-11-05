"""
Comprehensive hyperparameter tuning for Random Forest model.

After comparing multiple models in model_comparison.py, we determined that Random Forest
performs best for predicting student graduation outcomes. This script performs comprehensive
hyperparameter tuning using both GridSearchCV and RandomizedSearchCV to find the optimal
configuration for our Random Forest model.

After tuning, we save the best model so it can be loaded and used later without retraining.

Resources:
- Random Forest documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- GridSearchCV documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- RandomizedSearchCV documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
- Model persistence: https://scikit-learn.org/stable/modules/model_persistence.html
"""

# Import libraries
import os

import joblib
import pandas as pd
from exploratory_data_analysis import (
    feature_engineering,  # type: ignore[import-not-found]
)
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def transform_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[NDArray, NDArray]:
    """
    Apply feature engineering and scale the features using StandardScaler.
    """
    # Apply feature engineering to both train and test sets
    X_train_engineered = feature_engineering(X_train.copy())
    X_test_engineered = feature_engineering(X_test.copy())

    # Scale the features (including the newly engineered ones)
    # We fit the scaler on training data and transform both train and test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_engineered)
    X_test_scaled = scaler.transform(X_test_engineered)
    return X_train_scaled, X_test_scaled


def main() -> None:
    """
    Perform comprehensive hyperparameter tuning for Random Forest model.

    This function:
    1. Loads and preprocesses the student dataset
    2. Performs GridSearchCV with a focused parameter grid
    3. Performs RandomizedSearchCV with a broader parameter space
    4. Compares the results from both approaches
    5. Saves the best model for future use
    """

    # ============================================================================
    # SET UP DATA
    # ============================================================================

    # Load dataset from UCI Machine Learning Repository
    dataset = fetch_ucirepo(id=697)

    # Separate the dataset into features (X) and target column (y)
    X = dataset.data.features.copy()  # Copy to avoid modifying the original dataset
    y = dataset.data.targets["Target"]

    # Split the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Apply feature engineering and scale features using the same preprocessing as in model_comparison.py
    X_train_scaled, X_test_scaled = transform_data(X_train, X_test)

    # Create a baseline Random Forest model with default parameters
    baseline_model = RandomForestClassifier(random_state=42)

    # Use 4-fold cross-validation to evaluate baseline performance
    # Cross-validation gives us a more reliable estimate than a single train/test split.
    baseline_scores = cross_val_score(
        baseline_model, X_train_scaled, y_train, cv=4, scoring="accuracy", n_jobs=-1
    )
    baseline_mean = baseline_scores.mean()
    baseline_std = baseline_scores.std()
    print(f"Baseline accuracy: {baseline_mean:.4f} (+/- {baseline_std * 2:.4f})")

    # ============================================================================
    # GRID SEARCH CV
    # ============================================================================

    # Note: We're keeping the grid relatively small to avoid extremely long runtimes.
    # In practice, you might want to start with a broader search using RandomizedSearchCV
    # and then refine with GridSearchCV.
    # Check out the scikit-learn tutorial and model_comparison.py for more information on GridSearchCV.
    grid_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    # Create the Random Forest model with fixed random_state for reproducibility
    # random_state ensures we get the same results every time we run the code.
    # This is critical for ensuring our trials are reproducible and comparable.
    model = RandomForestClassifier(random_state=42)

    # Perform GridSearchCV
    # - cv=4: Use 4-fold cross-validation (same as baseline)
    # - scoring='accuracy': Measure performance using classification accuracy
    # - n_jobs=-1: Use all available CPU cores to speed up training
    # - verbose=1: Print progress updates during search
    grid_search = GridSearchCV(
        model,
        grid_param_grid,
        cv=4,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    # Fit the grid search on training data
    print("Starting GridSearchCV...")
    grid_search.fit(X_train_scaled, y_train)
    print("GridSearchCV completed!")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    print(f"Improvement over baseline: {grid_search.best_score_ - baseline_mean:.4f}")

    # ============================================================================
    # RANDOMIZED SEARCH CV
    # ============================================================================

    # Define a broader parameter distribution for RandomizedSearchCV
    # Unlike GridSearchCV which tries all combinations, RandomizedSearchCV randomly
    # samples from these distributions. This allows us to explore a much larger
    # parameter space efficiently.

    # We use scipy.stats distributions for continuous parameters:
    # - randint: Uniform distribution over integers in a range
    # - uniform: Continuous uniform distribution
    from scipy.stats import randint, uniform

    # Define parameter distributions for RandomizedSearchCV
    random_param_dist = {
        "n_estimators": randint(100, 500),  # Random integer between 100 and 499
        "max_depth": [5, 10, 15, 20, 25, 30, None],  # List of specific values
        "min_samples_split": randint(2, 20),  # Random integer between 2 and 19
        "min_samples_leaf": randint(1, 10),  # Random integer between 1 and 9
        "max_features": ["sqrt", "log2", None],  # List of options
        "bootstrap": [True, False],  # Whether to use bootstrap samples
        # max_samples: Only relevant when bootstrap=True
        # Controls the fraction of samples used for each tree
        "max_samples": uniform(0.7, 0.3),  # Uniform between 0.7 and 1.0 (70% to 100%)
    }

    # Create a fresh Random Forest model for RandomizedSearchCV
    # We create a new instance to ensure both search methods start from the same baseline
    model_random = RandomForestClassifier(random_state=42)

    # Perform RandomizedSearchCV
    # - n_iter=50: Try 50 random parameter combinations (much fewer than grid search)
    #   This is a good balance between exploration and computation time.
    random_search = RandomizedSearchCV(
        model_random,
        random_param_dist,
        n_iter=50,  # Number of random parameter combinations to try
        cv=4,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    # Fit the randomized search on training data
    print("Starting RandomizedSearchCV...")
    random_search.fit(X_train_scaled, y_train)
    print("RandomizedSearchCV completed!")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")
    print(f"Improvement over baseline: {random_search.best_score_ - baseline_mean:.4f}")

    # ============================================================================
    # SAVE THE BEST MODEL
    # ============================================================================
    # Compare the results from both search methods
    print(f"Baseline accuracy:           {baseline_mean:.4f}")
    print(f"GridSearchCV best accuracy:  {grid_search.best_score_:.4f}")
    print(f"RandomizedSearchCV accuracy: {random_search.best_score_:.4f}")

    # Select the best model based on cross-validation score
    if grid_search.best_score_ >= random_search.best_score_:
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        search_method = "GridSearchCV"
    else:
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        search_method = "RandomizedSearchCV"

    print(f"\nBest model found using: {search_method}")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation accuracy: {best_score:.4f}")

    # ============================================================================
    # EVALUATE ON TEST SET
    # ============================================================================
    # Now that we've selected the best model using cross-validation on the training data,
    # we evaluate it once on the test set to get an unbiased estimate of performance.
    # This is the ONLY time we use the test set, after all hyperparameter tuning is complete.
    #
    # The test set accuracy gives us confidence that our model will generalize well to new data.
    # If there's a large gap between cross-validation accuracy and test accuracy, it might
    # indicate overfitting to the training data. If you're concerned about overfitting, check out
    # the week 6 slides.
    test_accuracy = best_model.score(X_test_scaled, y_test)
    print(f"\nTest set accuracy: {test_accuracy:.4f}")
    print(f"Cross-validation accuracy: {best_score:.4f}")
    print(f"Difference: {abs(test_accuracy - best_score):.4f}")

    # ============================================================================
    # SAVE THE BEST MODEL
    # ============================================================================
    # Create a directory to save models if it doesn't exist
    # We save models in a 'models' directory to keep the project organized.
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Save the best model using joblib (recommended by scikit-learn)
    # joblib is more efficient than pickle for sklearn models with large numpy arrays.
    # The model file can be loaded later using: joblib.load('model_filename.joblib')
    model_filename = os.path.join(model_dir, "best_random_forest_model.joblib")
    joblib.dump(best_model, model_filename)
    print(f"Model saved to: {model_filename}")


if __name__ == "__main__":
    main()
