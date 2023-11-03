# OptiKNN-Custom-RandomSearchCV-for-K-Nearest-Neighbors-Hyperparameter-Optimization

## Custom RandomSearchCV Implementation

This project involves the implementation of a custom Randomized Search Cross-Validation (RandomSearchCV) method to optimize hyperparameters for a machine learning model, specifically the K-Nearest Neighbors (KNN) classifier. The primary objective is to find the best hyperparameter value that maximizes model performance.

### Function Description

- `RandomSearchCV(x_train, y_train, classifier, param_range, folds)`: This function performs a randomized search cross-validation to find the optimal hyperparameter for the KNN classifier.
  - `x_train`: A NumPy array of shape (n, d) representing the training data.
  - `y_train`: A NumPy array of shape (n,) or (n, 1) representing the training labels.
  - `classifier`: Typically, a KNeighborsClassifier.
  - `param_range`: A tuple (a, b) where a < b, representing the range of hyperparameter values to explore.
  - `folds`: An integer representing the number of folds for cross-validation.

### Steps in the Project

1. Generate 10 unique hyperparameter values, uniformly distributed within the specified `param_range`.
2. Divide the data indices into groups based on the number of `folds`.
3. Perform cross-validation for each hyperparameter by splitting the data into training and test sets according to the defined groups.
4. Calculate and store both the mean training and test accuracies for each hyperparameter.
5. Return the lists of training and test scores.
6. Plot a hyperparameter versus accuracy plot to visualize the performance.
7. Determine and choose the best hyperparameter.

### How to Use

1. Create an instance of the KNeighborsClassifier.
2. Generate 10 random hyperparameter values within the desired range and sort them.
3. Define the number of folds for cross-validation.
4. Call the `RandomSearchCV` function with your data and settings.
5. Plot the hyperparameter versus accuracy curve to select the optimal hyperparameter value.
6. Plot the decision boundaries for the model initialized with the best hyperparameter.

Feel free to use and adapt this custom RandomSearchCV implementation for your specific machine learning projects.
