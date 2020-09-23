# Credit Loans Analysis with Machine Learning (RandomOverSampler, SMOTE, Cluster Centroids, SMOTEENN, BalancedRandomForestClassifier, EasyEnsembleClassifier)

## Background
Credit risk is an inherently unbalanced classification problem, as the number of good loans easily outnumber the number of risky loans. Therefore, I employed different techniques to train and evaluate models with unbalanced classes. I used imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. I also evaluated the performance of these models and made a recommendation on whether they should be used to predict credit risk.

## Objectives
* Extract credit data
* Implement machine learning models. Train the model and generate predictions.
* Use resampling to attempt to address class imbalance.
* Oversample the data using the RandomOverSampler and SMOTE algorithms.
* Undersample the data using the cluster centroids algorithm.
* Use a combination approach with the SMOTEENN algorithm.
* Evaluate the performance of machine learning models.
* Calculate the balanced accuracy score.
* Generate a confusion matrix.
* Train a logistic regression classifier (from Scikit-learn) using the resampled data.
* Calculate the balanced accuracy score using balanced_accuracy_score from sklearn.metrics.
* Generate a confusion_matrix.
* Print the classification report (classification_report_imbalanced from imblearn.metrics).
* For the BalancedRandomForestClassifier, print the feature importance, sorted in descending order (from most to least important feature), along with the feature score.
* Write an analysis of the models’ performance. Describe the precision and recall scores, as well as the balanced accuracy score.

## Technologies Used
* Python (Jupyter Notebook)
* Pandas
* sklearn
* imblearn

## Analysis

##  Recommendation on ML model
