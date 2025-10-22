"""
Compare multiple machine learning models to predict student graduation outcomes.

We explore the UCI student dropout dataset and train several predictive models.
Each model is tuned using Grid Search and evaluated with Cross Validation to 
compare accuracy and identify the best performing approach.
"""

# Import libraries
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def main():

    # Load dataset from UCIrepo
    dataset = fetch_ucirepo(id=697)

    # Seperate the dataset into features (X) and target colum n(y)
    X = dataset.data.features.copy()
    y = dataset.data.targets["Target"]

    # Split the data into training and test sets (80% train, 20% test)
    # This allows us to evaluate performance on data the model hasn’t seen before.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features up front so every model receives the same transformed data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define a dictionary of models to compare
    # Each key (like "Decision Tree" or "Random Forest") represents a model name
    # Each model has two parts:
    #   1. "model": the actual classifier object from scikit-learn
    #   2. "params": a grid of hyperparameters we want to test during tuning
    # GridSearchCV will automatically try all combinations of these parameters
    # to find the set that gives the best accuracy. For example:
    #   - Random Forest tests different numbers of trees and depths
    #   - Decision Tree tests different values of max_depth and min_samples_split
    #   - MLP Classifier tests different neural network architectures or learning rates
    # For more info on GridSearchCV and hyperparameter tuning, see our scikit-learn tutorial on github.

    models = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
            },
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {
                "C": [0.1, 1.0, 10.0],
            },
        },
        "MLP Classifier": {
            "model": MLPClassifier(max_iter=3000, random_state=42),
            "params": {
                "hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "alpha": [0.0001, 0.001], 
                "learning_rate_init": [0.001, 0.01],
            },
        },
    }

    # Train and tune each model using 4-Fold Cross Validation
    # For each model, we'll start by training it with the default parameters
    # to see how it performs without tuning.
    # Then, we use 4-Fold Cross Validation to evaluate its accuracy:
    # the training data is split into four parts, and the model is trained
    # and validated on different parts each time. This helps us get a more
    # reliable estimate of performance than a single train/test split.
    # Finally, we average the accuracy across all four folds.    
    for name, cfg in models.items():
       
        print(f"\n--- {name} ---")

        model = cfg["model"]
        param_grid = cfg["params"]

        # Check how the model performs with its default settings
        # Cross-validation helps us estimate model performance on unseen data
        # Here we use 4 folds, meaning the dataset is split into 4 equal parts.
        # Each fold is used once for validation while the others are used for training.
        base_scores = cross_val_score(model, X_train_scaled, y_train, cv=4, scoring="accuracy")
        print(f"Baseline mean accuracy: {base_scores.mean():.3f}")

        # Use GridSearchCV to tune the hyperparameters
        # This will try different parameter combinations and pick the best
        grid = GridSearchCV(model, param_grid, cv=4, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train_scaled, y_train)

        print("Best parameters:", grid.best_params_)
        print(f"Tuned mean accuracy: {grid.best_score_:.3f}")
    
if __name__ == "__main__":
    main()
