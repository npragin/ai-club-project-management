"""
Load student dropout dataset from the UCI Machine Learning Repository.

Utilizing this dataset we will be classifying students into one of three
categories: graduate, dropout, or enrolled. The dataset includes features
about students such as their age, financial status, and past
school performance.

The dataset can be found on the UCI Machine Learning Repository here:
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
"""

import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Set plotting style for better readability
plt.style.use("ggplot")


def main() -> None:
    """Load the student dataset, print summary information, and plot graduation status distribution."""

    # Fetch students dataset from ucirepo
    dataset = fetch_ucirepo(id=697)

    # Store the dataset in two separate pandas dataframes
    # X = all columns (features) describing students (i.e., age, financial status)
    # y = graduation status of each student: Graduate, Dropout, or Enrolled
    X = dataset.data.features
    y = dataset.data.targets

    # Display dataset general information
    print(f"\nDataset metadata: \n {dataset.metadata}")
    print(f"\nNumber of rows: {X.shape[0]}\nNumber of columns: {X.shape[1]}")
    print(f"\nColumn information: \n {dataset.variables}")

    # Plot the graduation status (Target) distribution as a bar chart
    target_counts = y["Target"].value_counts()

    ax = target_counts.plot(kind="bar", title="Student Graduation Status", xlabel="", ylabel="Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    plt.show()


if __name__ == "__main__":
    main()
