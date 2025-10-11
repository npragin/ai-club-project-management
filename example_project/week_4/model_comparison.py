"""
Compare multiple machine learning models to predict student graduation outcomes.

We explore the UCI student dropout dataset and train several predictive models.
Each model is tuned using Grid Search and evaluated with Cross Validation to 
compare accuracy and identify the best performing approach.
"""

# Import libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Access dataset
from ucimlrepo import fetch_ucirepo

# Load dataset from UCIrepo
dataset = fetch_ucirepo(id=697)

# Seperate the dataset into features (X) and target colum n(y)
X = dataset.data.features.copy()
y = dataset.data.targets["Target"]

# Split the data into training and test sets (80% train, 20% test)
# This allows us to evaluate performance on data the model hasn’t seen before.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create 3 models to test: Decison Tree, Random Forest, Logistic Regression
# Each contians its own parameter grid for tuning 
models = {
    "Decision Tree":{
        "model": DecisionTreeClassifier(random_state=42),
        "params":{
            "max_depth": [None, 5, 10, 20],
            "min_samples_split":[2,5,10]
        }
    },
    "Random Forest":{
        "model":RandomForestClassifier(random_state=42),
        "params":{
            "n_estimators": [100,200],
            "max_depth":[None, 10, 20]
        }
    },
    "Logistic Regression":{
        "model":Pipeline([
            ('scalar', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000))
        ]),
        "params":{
            "clf__C": [0.1,1.0,10.0]
        }
    }
}

def main():

    # Create array an empty array to store each model's results
    # We'll use this later to compare the results
    results = []

    # Train and tune each model in a for loop
    for name, cfg in models.items():
        print(f"\n--- {name} ---")

        model=cfg["model"]
        param_grid=cfg["params"]

        # Check how the model performs with its default settings
        base_scores=cross_val_score(model, X_train, y_train, cv=4, scoring="accuracy")
        print(f"Baseline mean accuracy: {base_scores.mean():.3f}")

        # Use GridSearchCV to tune the hyperparameters
        # This will try different parameter combinations and pick the best
        grid = GridSearchCV(model, param_grid, cv=4, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)

        print("Best parameters:", grid.best_params_)
        print(f"Tuned mean accuracy: {grid.best_score_:.3f}")
    
        # Test the best model on the held-out test data
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        print(f"Test accuracy: {test_acc:.3f}")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
