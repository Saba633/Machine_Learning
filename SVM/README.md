# Spam Detection Using SVM

This project implements a spam detection system using Support Vector Machine (SVM) for classifying text messages as spam or ham (not spam).

## Challenges Faced
- **CountVectorization**: Remember that this is used to extract unique individual words (tokenization) and count the occurrences of each word.
- **Using `fit_transform`**: In `X_train`, we use `fit_transform`, which is only applied to the training data to transform it into a numerical format.
- **Using `transform`**: In `X_test`, we use `transform`, which ensures that the test data is in the same numerical format as the training data.
- **Plotting with `.value_counts()`**: This method counts only the unique values in the specified column, which is helpful for visualizing distributions in a pie chart.
