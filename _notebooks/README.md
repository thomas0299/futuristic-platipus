# Table of content for code and notebooks

First, you will need to use the `environment.yml` environment with all relevant dependencies.

1. All relevant packages imported, functions and variables defined. Werun this script in every subsequent notebook

2. Accessing the WPDx API for water points in Uganda

3. Accessing demographic data from the Uganda Bureau of Statistics, dataset was manually downloaded then loaded in

4. Accessing the ACLED API for conflicts in Uganda

5. Merging water, demographic and conflict data into one dataset. We end up with information for each water point

6. EDA of all features, visualising information and feature engineering as we gain a better understanding of the data

7. Introudcing the problem, aim and running a Logistic Regression

8. K Nearest Neighbours

9. Decision Tree

10. Random Forest

11. Gaussian Naïve Bayes

12. Support Vector Machine

13. AdaBoost

14. XGBoost

15. Neural Network

16. Loading model results from the WPDx ML model as a benchmark and for future comparison

17. Model comparison to choose best model and identify next steps

ARCHIVED EDA. Initial EDA made to define my question and outcome variable. This notebook is very large and messy. This was purely for exploration's sake.

**NOTE that certain comments and interpretations do not always match perfectly certain outputs. This is because the model may have been re-ran, with some hyperparameters having slightly changed when running a random search cross validation. Since these small differences don't actually impact the performance of the models, we did not change our text every time.**