"""
Explore student dropout dataset from the UCI Machine Learning Repository and train initial model.

We use ydata-profiling to perform an initial exploratory data analysis. We then train a 
Decision Tree Classifier model to predict a students graduation status. To improve
performance, we create new features, and evaluate the model using K-Fold Cross Validation. 
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score


# Load data
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 

# Create report for EDA
df = pd.concat([X, y], axis=1) # combine features and target into one dataframe
profile = ProfileReport(df, title="Profiling Report")
profile.to_file("example_project/week_2/student_data_report.html")

# Split data into train and test subsets (80%-20% split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier model
model = DecisionTreeClassifier(random_state=42)

# Train model and make predictions
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(predictions, y_test)
print(f"Accuracy Score: {accuracy}")

# Use K-Fold Cross-Validation to evaluate model performance
kf = KFold(n_splits=4)
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
print("Cross Validation Scores:", cv_scores)
print(f"Mean Validation Score: {np.mean(cv_scores)}")

# Feature Engineering
def feature_engineering(X):
    X["Financial standing"] = X["Tuition fees up to date"] + X["Scholarship holder"] - X["Debtor"]

    # X["Approval rate 1st sem"] = X["Curricular units 1st sem (approved)"] / X["Curricular units 1st sem (enrolled)"]
    X["Approval rate 2nd sem"] = X["Curricular units 2nd sem (approved)"] / X["Curricular units 2nd sem (enrolled)"]

    # X["Economic status"] = X["Unemployment rate"] + X["Inflation rate"]

    X["Grade improvement"] = X["Curricular units 2nd sem (grade)"] - X["Curricular units 1st sem (grade)"]

    # X["Standard age at enrollment"] = ((X["Age at enrollment"] >=18) & (X["Age at enrollment"] <= 20)).astype(int)

    # X["Single status"] = (X["Marital Status"] == 1).astype(int)

    # X["Admission vs previous grade"] = X["Admission grade"] - X["Previous qualification (grade)"]

    higher_education = [2, 3, 4, 5, 40, 41, 42, 43, 44]
    X["Mother higher education"] = X["Mother's qualification"].isin(higher_education).astype(int)
    X["Father higher education"] = X["Father's qualification"].isin(higher_education).astype(int)
    X["Parent higher education"] = (
        X["Mother's qualification"].isin(higher_education) | 
        X["Father's qualification"].isin(higher_education)
    ).astype(int)
    # X["Both parent higher education"] = (
    #     X["Mother's qualification"].isin(higher_education) & 
    #     X["Father's qualification"].isin(higher_education)
    # ).astype(int)

    # X["Higher education"] = X["Previous qualification"].isin(higher_education).astype(int)

    return X

# Update data with new features and create train and test subsets
X = feature_engineering(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use K-Fold Cross-Validation to evaluate model performance
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
print("\nScores with feature engineering: ")
print("Cross Validation scores:", cv_scores)
print(f"Mean Validation Score: {np.mean(cv_scores)}")