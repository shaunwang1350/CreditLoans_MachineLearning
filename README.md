# Credit Loans Analysis with Machine Learning (RandomOverSampler, SMOTE, Cluster Centroids, SMOTEENN, BalancedRandomForestClassifier, EasyEnsembleClassifier)

## Background
Credit risk is an inherently unbalanced classification problem, as the number of good loans easily outnumber the number of risky loans. Therefore, I employed different techniques to train and evaluate models with unbalanced classes. I used imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. I also evaluated the performance of these models and made a recommendation on whether they should be used to predict credit risk.

## Objectives
* Extract credit data
* Encode Obj Data Types into Int/Float Data Types through  LabelEncoder from sklearn.preprocessing
* Implement machine learning models. Train the model and generate predictions.
* Use resampling to attempt to address class imbalance.
* Oversample the data using the RandomOverSampler and SMOTE algorithms.
* Undersample the data using the cluster centroids algorithm.
* Use a combination approach with the SMOTEENN algorithm.
* Train a logistic regression classifier (from Scikit-learn) using the resampled data.
* Evaluate the performance of machine learning models.
* Calculate the balanced accuracy score using balanced_accuracy_score from sklearn.metrics.
* Generate a confusion matrix.
* Print the classification report (classification_report_imbalanced from imblearn.metrics).
* Write an analysis of the models’ performance. Describe the precision and recall scores, as well as the balanced accuracy score.

## Technologies Used
* Python (Jupyter Notebook)
* Pandas
* sklearn
* imblearn

## Analysis

#### Oversampling with RandomOverSampler and SMOTE algorithms: 

**imbalanced classification report**
|             | pre  | rec  | spe  | f1   | geo  | iba  | sup   |
|-------------|------|------|------|------|------|------|-------|
| high_risk   | 0.01 | 0.60 | 0.67 | 0.02 | 0.64 | 0.40 | 101   |
| low_risk    | 1.00 | 0.67 | 0.60 | 0.80 | 0.64 | 0.41 | 17104 |
| avg / total | 0.99 | 0.67 | 0.60 | 0.80 | 0.64 | 0.41 | 17205 |
Balanced Accuracy Score: 0.608

#### Undersampling with Cluster Centroids algorithms: 

**imbalanced classification report**
|             | pre  | rec  | spe  | f1   | geo  | iba  | sup   |
|-------------|------|------|------|------|------|------|-------|
| high_risk   | 0.01 | 0.65 | 0.42 | 0.01 | 0.52 | 0.28 | 101   |
| low_risk    | 1.00 | 0.42 | 0.65 | 0.59 | 0.52 | 0.27 | 17104 |
| avg / total | 0.99 | 0.42 | 0.65 | 0.58 | 0.52 | 0.27 | 17205 |
Balanced Accuracy Score: 0.534

#### Combination approach with SMOTEEN algorithms: 

**imbalanced classification report**
|             | pre  | rec  | spe  | f1   | geo  | iba  | sup   |
|-------------|------|------|------|------|------|------|-------|
| high_risk   | 0.01 | 0.77 | 0.53 | 0.02 | 0.64 | 0.42 | 101   |
| low_risk    | 1.00 | 0.53 | 0.77 | 0.69 | 0.64 | 0.40 | 17104 |
| avg / total | 0.99 | 0.57 | 0.77 | 0.72 | 0.64 | 0.40 | 17205 |
Balanced Accuracy Score: 0.650

##  Recommendation on ML model
