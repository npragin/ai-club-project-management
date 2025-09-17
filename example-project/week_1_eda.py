# In order to run this file make sure to run the command "pip install ucimlrepo"
# link to dataset: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import math

# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 

# metadata 
print(f"\nDataset metadata: \n {predict_students_dropout_and_academic_success.metadata}") 

# Feature Data Overview
print(f"\nNumber of rows: {X.shape[0]}\nNumber of columns: {X.shape[1]}")
print(f"\nColumn information: \n {predict_students_dropout_and_academic_success.variables}")

# Target Data Distribution
y.value_counts().plot(kind="bar", title="Graduation Status Distribution")
plt.xticks(rotation=0, ha="center")
plt.xlabel("")
plt.ylabel("Count")
plt.show()

# Plot features
int_columns = [col for col in X.columns if X[col].dtype == "int64"]
float_columns = [col for col in X.columns if X[col].dtype == "float"]

def plot_in_batches(features, plot_type="bar"):
    """Plot columns in multiple 2x3 grids."""
    n = len(features)
    batches = math.ceil(n / 6)  # how many 2x3 pages needed

    for b in range(batches):
        # Get the current batch of up to 6 feature distributions
        batch_features = features[b*6 : (b+1)*6]

        fig, axes = plt.subplots(2, 3, figsize=(15, 12))
        axes = axes.flatten()

        for i, col in enumerate(batch_features):
            if plot_type == "bar":
                X[col].value_counts().head(10).plot(
                    kind="bar", ax=axes[i], title=f"{col} Distribution"
                )
                axes[i].set_xlabel("")
                axes[i].set_ylabel("Count")
            elif plot_type == "hist":
                X[col].plot(
                    kind="hist", bins=20, ax=axes[i], title=f"{col} Histogram"
                )
                axes[i].set_xlabel("")
                axes[i].set_ylabel("Count")

        # Hide any unused subplots
        for j in range(len(batch_features), 6):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=3.0)
        plt.show()

# plot the columns
if int_columns:
    plot_in_batches(int_columns, plot_type="bar")

if float_columns:
    plot_in_batches(float_columns, plot_type="hist")

# Get Null Value Counts
null_counts = X.isnull().sum()
print(f"\nNull value count for each feature column:\n{null_counts}")

target_null_counts = y.isnull().sum()
print(f"\nNull value count for target column:\n{target_null_counts}")







