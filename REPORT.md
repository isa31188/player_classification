# Paying player classification problem

## Problem statement, modeling approach and results.

The problem at hand is about building a model for classifying as paying/non-paying based on a dataset of players.
The dataset was initially cleaned for duplicated rows and missing values. 

Based on the distributions of different fields, a simple predictive model based on player_level was built. This model reached nearly 80% precision in the 10th decile of players (test set) and was used as baseline.

The dataset was afterwards used for training 2 logistic regression classifiers (standard scaling vs min-max scaling) and 1 random forest classifier. These algorithms were selected for their relative simplicity. There was little difference between the 2 logreg classifiers. On total, there were three modeling iterations:
* training the models based on the original dataset
* training the models based on the dataset with additional features (ratios between numeric variables)
* training the models based on the dataset undersampled for balanced target classes.

In all cases, the random forest models tipically showed better performance on the training set than on the test set, while logistic regression classifiers generally showed nearly the same performance in both sets. 

In any case, the random forest classifiers showed better performance in all cases even when evaluated in the test set.

Adding new features or training based on an undersampled dataset didn't improve much the models' performance.

The results of these modeling stages are summarized below in terms of lift and precision on the 10th decile evaluated on the training and on the test sets. For simplicity only random forest models are represented.

Table 1. 10th decile performance metrics for alternative predictive models - training set
| Model                                               | n    | tp   | lift     |precision | max_lift |
|---|---|---|---|---|---|
| Rule-based                                          |	3150 | 2430 | 3.01 | 77.1% | 3.91 |
| Random forest, trained  using the original  dataset | 3150 | 3106	| 3.85 | 98.6% | 3.91 |
| Random forest, trained with additional features     | 3150 | 3103 | 3.85 | 98.5% | 3.91 |
| Random forest, trained with an undersampled dataset |	1613 | 1612 | 2.00 | 99.9% | 2.00 |

Table 2. 10th decile performance metrics for alternative predictive models - test set
| Model                                               | n   | tp  |	lift     |precision | max_lift |
|---|---|---|---|---|---|
| Rule-based                                          | 350 | 279 |	3.11 | 79.7% | 3.91 |
| Random forest, trained  using the original  dataset | 350	| 313 | 3.49 | 89.4%	| 3.91 |
| Random forest, trained with additional features     | 350 | 311 |	3.47 | 88.9% | 3.91 |
| Random forest, trained with an undersampled dataset |	350 | 314 |	3.50 | 89.7% | 3.91 |

Based on these tables, the winning model would be the random forest trained on an undersampled dataset, as this slightly higher precision and lift on the test set. This has the additional benefit of training with a smaller dataset, and thus requiring less cpu and memory resources. The simple rule-based model, while substantially less performant, shows nevertheless a reasonable precision in the test dataset.


## To-Do and other remarks.

* Outliers were not handled during the cleaning process. This should be revisited and its impact tested.
* Distributions of new features created were not plotted. These could eventually lead to better baseline models.
* The random forest model were clearly overfitting with a performance on the training set substantially higher than that on the test set. 
