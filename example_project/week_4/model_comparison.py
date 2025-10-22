"""
Compare multiple machine learning models to predict student graduation outcomes.

We explore the UCI student dropout dataset and train several predictive models.
Each model is tuned using Grid Search and evaluated with Cross Validation to
compare accuracy and identify the best performing approach.
"""

# Import libraries
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


# Helper function to perform all of our data preprocessing in one place.
# This will be used throughout the project to ensure consistency in our data preprocessing.
# We want to ensure we use the same preprocessing steps when training and deploying the model.
def transform_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[NDArray, NDArray]:
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def main() -> None:

    # Load dataset from UCIrepo
    # Depending on how your dataset is available, you may load using pandas.read_csv
    dataset = fetch_ucirepo(id=697)

    # Seperate the dataset into features (X) and target column (y)
    X = dataset.data.features.copy()  # Copy to avoid modifying the original dataset during preprocessing
    y = dataset.data.targets["Target"]

    # Split the data into training and test sets (80% train, 20% test)
    # This allows us to evaluate performance on data the model hasn't seen before.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features up front so every model receives the same transformed data.
    # This is important for models that are sensitive to feature scaling, such as neural networks.
    # Whenever you transform your data, you must transform your test data the same way.
    # We use a function to scale the data so we can reuse it in the future. This will be critical
    # when we deploy the model and need to apply the same transformation to new data.
    X_train_scaled, X_test_scaled = transform_data(X_train, X_test)

    # Define a dictionary of models to compare
    # Each key (like "Decision Tree" or "Random Forest") represents a model name
    # Each model has two parts:
    #   1. "model": the actual classifier object from scikit-learn
    #   2. "params": a grid of hyperparameters we want to test during tuning
    # GridSearchCV will automatically try all combinations of these parameters
    # to find the set that gives the best accuracy. For example:
    #   - Random Forest tests different numbers of trees and depths
    #   - Logistic Regression tests different values of C
    #   - MLP Classifier tests different hidden layer sizes, alpha, and learning rate initialization
    # For more info on GridSearchCV and hyperparameter tuning,
    # see the Hyperparameters section of the scikit-learn tutorial.
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
    # Then, we use 4-Fold Cross Validation to evaluate its accuracy.
    # Recall, in 4-Fold Cross Validation, the training data is split into four parts,
    # and the model is trained and validated on different parts each time.
    # This helps us get a more reliable estimate of performance than a single train/test split.
    # Finally, we average the accuracy across all four folds to get a single estimate of performance.
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

    # That's it! We don't touch test data until we finalize the model.


if __name__ == "__main__":
    main()
