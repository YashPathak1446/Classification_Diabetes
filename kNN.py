#kNN classification model 
#target values to classify: <30, >30, NO

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.model_selection import GridSearchCV, cross_val_score
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import ssl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

seed = 1234
np.random.seed(seed) 

ssl._create_default_https_context = ssl._create_unverified_context

# fetch dataset 
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 
  
# data (as pandas dataframes) 
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets 
y = np.array(y)

unique_values = np.unique(y)
print("Unique values for y:", unique_values) #<30, >30, NO

# Convert categorical variables to numerical using one-hot encoding
X_pandas = pd.get_dummies(X)
X = X_pandas

# Remove constant features
selector = VarianceThreshold()
X = selector.fit_transform(X)

# Perform feature selection
k_best = SelectKBest(score_func=f_classif, k=10)
X_selected = k_best.fit_transform(X, np.ravel(y))

# Get the indices of selected features
selected_features_indices = k_best.get_support(indices=True)

# Print out the selected feature indices
print("Selected feature indices:", selected_features_indices)

# Print out the names of the selected features
# If feature names are not available, you need to specify them manually
X = np.array(X)
y = np.ravel(y)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Split the rest into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=seed)

#to understand the data better 

# Define the range of hyperparameters to search
param_grid = {'n_neighbors': [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45]}  # Adjust the range of k as needed

# Initialize the kNN classifier
knn = KNeighborsClassifier()

# Initialize GridSearchCV with the kNN classifier and parameter grid
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')  # cv is the number of folds for cross-validation

# Perform cross-validation
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

#based on best model
# Predict on the test set
y_pred = best_model.predict(X_test)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", accuracy)

y_val_pred = best_model.predict(X_val)

# Calculate accuracy on validation set
validation_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Set Accuracy:", validation_accuracy)

# Calculate training accuracy
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Set Accuracy:", train_accuracy)

# Evaluate the best model using cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

#graph accuracies 
# Define the range of hyperparameters (number of neighbors)
neighbors = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45]

# Initialize empty lists to store accuracies and corresponding k values
accuracies = []
k_values = []

# Loop through each neighbor value
for k in neighbors:
    # Initialize KNN classifier with current k value
    knn = KNeighborsClassifier(n_neighbors=k)
    # Perform cross-validation
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
    # Calculate mean accuracy
    mean_accuracy = cv_scores.mean()
    # Append mean accuracy and corresponding k value to lists
    accuracies.append(mean_accuracy)
    k_values.append(k)

# Plot the graph
plt.plot(k_values, accuracies, marker='o')
plt.title('Accuracy vs Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# # Initialize KNN classifier with best hyperparameters
# knn = KNeighborsClassifier(**best_params)

# # # Define a range of training set sizes
# # train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# # # Initialize empty lists to store accuracies
# # accuracies = []

# # # Loop through each training set size
# # for train_size in train_sizes:
# #     # Split the dataset into training and validation sets
# #     X_train_partial, _, y_train_partial, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=seed)
    
# #     # Fit the model on the partial training set
# #     knn.fit(X_train_partial, y_train_partial)
    
# #     # Predict on the validation set
# #     y_val_pred = knn.predict(X_val)
    
# #     # Calculate accuracy on validation set
# #     accuracy = accuracy_score(y_val, y_val_pred)
    
# #     # Append accuracy to list
# #     accuracies.append(accuracy)

# # # Plot the graph
# # plt.plot(train_sizes, accuracies, marker='o')
# # plt.title('Effect of Training Set Size on Validation Accuracy')
# # plt.xlabel('Training Set Size')
# # plt.ylabel('Validation Accuracy')
# # plt.grid(True)
# # plt.show()





