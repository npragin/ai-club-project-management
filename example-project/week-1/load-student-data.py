"""
Load student dropout dataset from the UCI Machine Learning Repository.

Utilizing this dataset we will be classifying students into one of three
categories: graduate, dropout, or enrolled. The dataset includes features 
about students such as their age, financial status, and past
school performance. 

The dataset can be found on the UCI Machine Learning Repository here: 
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


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
target_counts = y["Target"].value_counts()

ax = target_counts.plot(
    kind="bar",
    title="Student Graduation Status",
    xlabel="", 
    ylabel="Count"
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
plt.show()







