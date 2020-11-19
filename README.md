# Credit Loans Analysis with Machine Learning 
![Developement](https://img.shields.io/badge/progress-complete-green)
![reposize](https://img.shields.io/github/repo-size/shaunwang1350/CreditLoans_MachineLearning)
![githubfollows](https://img.shields.io/github/followers/shaunwang1350?style=social)
<br >

(RandomOverSampler, SMOTE, Cluster Centroids, SMOTEENN, BalancedRandomForestClassifier, EasyEnsembleClassifier)


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

## Results

### Oversampling with RandomOverSampler and SMOTE algorithms: 

**Confusion Matrix**
|             | Predicted True  | Predicted False | 
|-------------|------|------|
| Actually True | 71 | 30 |
| Actually False | 6451 | 10653 |

**imbalanced classification report**
|             | pre  | rec  | spe  | f1   | geo  | iba  | sup   |
|-------------|------|------|------|------|------|------|-------|
| high_risk   | 0.01 | 0.60 | 0.67 | 0.02 | 0.64 | 0.40 | 101   |
| low_risk    | 1.00 | 0.67 | 0.60 | 0.80 | 0.64 | 0.41 | 17104 |
| avg / total | 0.99 | 0.67 | 0.60 | 0.80 | 0.64 | 0.41 | 17205 |

Balanced Accuracy Score: 0.608
Precision TP/(TP+FP): 0.10
Recall / Sensitivity TP/(TP+FN): 0.77

### Undersampling with Cluster Centroids algorithms: 

**Confusion Matrix**
|             | Predicted True  | Predicted False | 
|-------------|------|------|
| Actually True | 66 | 35 |
| Actually False | 9989 | 7115 |

**imbalanced classification report**
|             | pre  | rec  | spe  | f1   | geo  | iba  | sup   |
|-------------|------|------|------|------|------|------|-------|
| high_risk   | 0.01 | 0.65 | 0.42 | 0.01 | 0.52 | 0.28 | 101   |
| low_risk    | 1.00 | 0.42 | 0.65 | 0.59 | 0.52 | 0.27 | 17104 |
| avg / total | 0.99 | 0.42 | 0.65 | 0.58 | 0.52 | 0.27 | 17205 |

Balanced Accuracy Score: 0.534
Precision TP/(TP+FP): 0.006
Recall / Sensitivity TP/(TP+FN): 0.77

### Combination approach with SMOTEEN algorithms:

**Confusion Matrix**
|             | Predicted True  | Predicted False | 
|-------------|------|------|
| Actually True | 78 | 23 |
| Actually False | 8073 | 9031 |

**imbalanced classification report**
|             | pre  | rec  | spe  | f1   | geo  | iba  | sup   |
|-------------|------|------|------|------|------|------|-------|
| high_risk   | 0.01 | 0.77 | 0.53 | 0.02 | 0.64 | 0.42 | 101   |
| low_risk    | 1.00 | 0.53 | 0.77 | 0.69 | 0.64 | 0.40 | 17104 |
| avg / total | 0.99 | 0.57 | 0.77 | 0.72 | 0.64 | 0.40 | 17205 |

Balanced Accuracy Score: 0.650
Precision TP/(TP+FP): 0.009
Recall / Sensitivity TP/(TP+FN): 0.77

## Analysis

While comparing the balanced accuracy scores, it is clear that the Combination approach with SMOTEEN algorithms has the highest accuracy. This is also true with the recall/sensitivity, which takes into consideration how the ML model is capable of correctly predicting the credit risk of a specific individual. That said, Oversampling with RandomOverSampler and SMOTE algorithms has a 0.001 higher likely hood of predicting the credit risk of specifical individual when measured against false negative cases. That said, with recall and accuracy much higher than other models, Combination approach with SMOTEEN algorithms seems to be the best approach.

##  Recommendation on ML model

Even though we employed random over, under, and combined sampling appoarches to offset the small number of actual high-risk credit loans, there is no by-passing that high-risk credit loans is a minority data set within the population of the dataset. This has a big effect on the performance on our model even when we have employed techniques to offset this distribution. A recommendation would be to locate more high-risk credit loan data. The reason for this is so that the ML model can be better trained in identifying the loans through the input features. 
