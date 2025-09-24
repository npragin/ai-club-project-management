"""
Explore student dropout dataset and train a baseline predictive model.

We use ydata-profiling to perform exploratory data analysis. A Decision Tree Classifier 
is then trained to predict a student's graduation status. To improve performance, 
we create new features and evaluate the model using K-Fold Cross Validation. 
"""

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo
from ydata_profiling import ProfileReport


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """Add new derived features to the student dataset."""

    # Combine multiple finanical indicators into one feature
    X["Financial standing"] = X["Tuition fees up to date"] + X["Scholarship holder"] - X["Debtor"]

    # Calculate ratio of approved (passed and received credit) to enrolled credits 
    X["Approval rate 2nd sem"] = X["Curricular units 2nd sem (approved)"] / X["Curricular units 2nd sem (enrolled)"]

    # Calculate grade average improvement by students from the first to second semester
    X["Grade improvement"] = X["Curricular units 2nd sem (grade)"] - X["Curricular units 1st sem (grade)"]

    # Encode whether parents hold higher education degrees
    higher_education = [2, 3, 4, 5, 40, 41, 42, 43, 44]
    X["Mother higher education"] = X["Mother's qualification"].isin(higher_education).astype(int)
    X["Father higher education"] = X["Father's qualification"].isin(higher_education).astype(int)
    X["Parent higher education"] = (
        X["Mother's qualification"].isin(higher_education) | 
        X["Father's qualification"].isin(higher_education)
    ).astype(int)

    return X


def main():
    """Run the student graduation status prediction workflow."""

    # Fetch students dataset from ucirepo
    dataset = fetch_ucirepo(id=697) 
    
    # Store the dataset in two separate pandas dataframes
    # X = all columns (features) describing students (i.e., age, financial status)
    # y = graduation status of each student: Graduate, Dropout, or Enrolled
    X = dataset.data.features 
    y = dataset.data.targets 

    # Create profile report to visualize and explore the student dataset (EDA)
    df = pd.concat([X, y], axis=1) # combine features and target into one dataframe
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file("example_project/week_2/student_data_report.html")

    # Split data into train and test subsets (80% train - 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Decision Tree Classifier model
    model = DecisionTreeClassifier(random_state=42)

    # Perform 4-Fold Cross-Validation to estimate model performance
    # In each iteration, the model is trained on 3/4 of the training data and
    # validated on the remaining 1/4 (validation set). This process repeats 4 times
    kf = KFold(n_splits=4)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")

    print("Cross-Validation Scores:", cv_scores)
    print(f"Mean Validation Score: {np.mean(cv_scores)}")

    # Update dataset with new features
    X = feature_engineering(X)

    # Split updated dataset into train and test subsets (80% train - 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use 4-Fold Cross-Validation to re-evaluate model performance after feature engineering
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")

    print("\nScores with Feature Engineering: ")
    print("Cross Validation scores:", cv_scores)
    print(f"Mean Validation Score: {np.mean(cv_scores)}\n")

    # Train model on entire training set
    model.fit(X_train, y_train)

    # Evaluate model on test set
    predictions = model.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    print(f"Accuracy Score on Test Data: {accuracy}")


if __name__ == "__main__":
    main()
