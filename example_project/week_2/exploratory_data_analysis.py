"""
Explore student dropout dataset from the UCI Machine Learning Repository.

TODO: Put further explanation here

"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


# load data
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 

# metadata 
print(f"\nDataset metadata: \n {predict_students_dropout_and_academic_success.metadata}") 

# Feature Data Overview
print(f"\nNumber of rows: {X.shape[0]}\nNumber of columns: {X.shape[1]}")
print(f"\nColumn information: \n {predict_students_dropout_and_academic_success.variables}")

# Target Data Distribution
target_counts = y["Target"].value_counts()

ax = target_counts.plot(
    kind="bar",
    title="Student Graduation Status",
    xlabel="", 
    ylabel="Count"
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
plt.show()

# things to include
# distribution charts for each feature (update the x-values if possible)
# value counts for each
# null counts
# 