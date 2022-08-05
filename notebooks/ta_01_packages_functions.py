################################
#A. importing all relevant packages
################################

#for data wrangling
import numpy as np
import pandas as pd
from scipy import stats
from imblearn.over_sampling import SMOTE

#for api requests
import requests
import os
from sodapy import Socrata

#for visualisations
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


#for modelling
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve, auc, precision_recall_curve, precision_recall_fscore_support,\
    classification_report, plot_confusion_matrix, confusion_matrix

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shap

#various other
from tempfile import mkdtemp
import joblib
from datetime import datetime, date
import time
import glob
import pickle

#For the sake of output, we disable warnings. All warnings related to the version of libraries
import warnings
warnings.filterwarnings('ignore')

################################
# B. some constant variables
################################

# defining working directory
data_filepath = '/Users/thomasadler/Desktop/futuristic-platipus/data/'
model_filepath = '/Users/thomasadler/Desktop/futuristic-platipus/models/'
dictionary_filepath='/Users/thomasadler/Desktop/futuristic-platipus/data_dictionary/'
images_filepath = '/Users/thomasadler/Desktop/futuristic-platipus/images/'

#set random seed for consistent results
rand_seed=1
    
#setting color palette for bar chart
palette ={"positive": "royalblue", "neutral": "silver", "negative": "firebrick"}

################################
# C. creating functions
################################

#1. convert an object column to a datetime
def date_converter(df, col):
    '''
    param df: dataframe
    param col: 'object' column in dataframe that needs to be converted
                to datetime in datafrane

    return: changes col to a datetime column
    '''
    df[col]=pd.to_datetime(df[col])

#2. convert an object column to a float
def float_converter(df, col):
    '''
    param df: dataframe
    param col: 'object' column in dataframe that needs to be converted
                to float in datafrane

    return: changes col to a float column
    '''
    df[[col]]=df[[col]].astype('float32')

#3. fill missing values with regional median
def missing_to_regional_median(df, col, region):
    '''
    param df: dataframe
    param col: column with missing values that needs to be filled
    param region: regional level at which we want to choose the median from

    return: fills missing value in col with median of its region
    '''
    df[col] = df[col].fillna(df.groupby(region)[col].transform('median'))


#4. calculating the correlation between X (independent variables) and y (dependent variable)
def correl_loop(X, y):
    '''
    param X: dataframe or series of independent variables
    param y: series of dependent variables

    return: correlation coefficient and p-value between all independent variables
            with dependent variable. Including the direction of the correlation and
            the statistical significance.
    '''
    #Dataset with only numerical columns
    new_X=X.select_dtypes(exclude=['object','datetime64'])
    
    #Loop over every numerical column
    for column in new_X.columns:

        #Calculate correl coeff and p-value
        pears=stats.pearsonr(y,X[column])
        print(f"The correlation coefficient is {round(pears[0],4)} with a p-value of {pears[1]}." )
        
        #Print the direction of the correlation
        if pears[0]>0.5:
            print (f'\033[1m{y.name}\033[0m is  \033[1mstrongly positively\033[0m correlated with \033[1m{column}\033[0m.')
        
        elif pears[0]>0:
            print (f'\033[1m{y.name}\033[0m is  \033[1mweakly positively\033[0m correlated with \033[1m{column}\033[0m.')
        
        elif pears[0]>-0.5:
            print (f'\033[1m{y.name}\033[0m is  \033[1mweakly negatively\033[0m correlated with \033[1m{column}\033[0m.')
            
        else:
            print (f'\033[1m{y.name}\033[0m is  \033[1mstrongly negatively\033[0m correlated with \033[1m{column}\033[0m.')       
        
        
        #Print the statistical significance
        if pears[1]<0.05:
            print ('\033[1mStatistically significant\033[0m.')
        else: 
            print ('\033[1mNot significant\033[0m.')
        print(' -------------- ')


#5. Check LINEARITY with function plotting each independent with the dependent variable
def plot_loop(X,y):
    '''
    param X: dataframe or series of independent variables
    param y: series of dependent variables

    return: subplots with a scatter for each independent variable with dependent variable
    '''  
    #Creating subplot for readability
    plt.subplots(12,3, figsize=(20,100))
    
    #Loop through all independent variables and create scatter plot with dependent variable
    for i, col in enumerate(X.columns, 1):
        plt.subplot(12,3,i)
        sns.scatterplot(X[col], y)
        plt.ylabel(y.name)
        plt.xlabel(col)
    
    plt.tight_layout()
    plt.show() 


#6. boxplots to look at correlation between all independent variables and dependent variable
def box_loop(X,y):
    '''
    param X: dataframe or series of independent variables
    param y: series of dependent variables

    return: subplots with a boxplot for each numerical independent variable with categorical dependent variable
    '''  
    #Creating subplot for readability
    plt.subplots(20,2, figsize=(20,100))
    
    #Loop through all independent variables and create scatter plot with dependent variable
    for i, col in enumerate(X.columns, 1):
        plt.subplot(20,2,i)
        sns.boxplot(y=X[col], x=y)
        plt.ylabel(X[col].name)
        plt.xlabel(y.name)
    
    plt.tight_layout()
    plt.show()

#7. run a grid search cross validation on defined set of estimators and parameters
def pipeline_cross_val_grid(estimator, param, X_training, y_training, X_testing, y_testing):
    '''
    param estimator: dictionary of transformers/models
    param param: dictionary of hyperparameters
    param X_training: training independent variables
    param y_training: training dependent variable
    param X_testing: testing independent variables
    param y_testing: testing dependent variable

    return: prints model best parameters found from gridsearch crosss validation and prints score on test set
    '''

    # make temporary folder to store results
    cachedir = mkdtemp()

    # setting up which transformers/scalers we want to grid search
    estimators = estimator

    # setting up the pipeline
    pipeline = Pipeline(estimators, memory=cachedir)

    # defining parameters we want to compare
    params = param

    # instantiating grid search
    grid_search = GridSearchCV(pipeline, param_grid=params)

    # cross validating with training set
    fitted_search = grid_search.fit(X_training, y_training)

    # getting results
    print(
        f"The model with the best CV score has the following parameters: {fitted_search.best_params_}.")
    print(
        f"The best model has an accuracy score of {fitted_search.score(X_testing, y_testing)} on the test set")


#8. run a randomised search cross validation on defined set of estimators and parameters
def pipeline_cross_val_random(estimator, param, X_training, y_training, X_testing, y_testing):
    '''
    param estimator: dictionary of transformers/models
    param param: dictionary of hyperparameters
    param X_training: training independent variables
    param y_training: training dependent variable
    param X_testing: testing independent variables
    param y_testing: testing dependent variable

    return: prints model best parameters found from random search crosss validation and prints score on test set
    '''

    # make temporary folder to store results
    cachedir = mkdtemp()

    # setting up which transformers/scalers we want to grid search
    estimators = estimator

    # setting up the pipeline
    pipeline = Pipeline(estimators, memory=cachedir)

    # defining parameters we want to compare
    params = param

    # instantiating grid search
    rand_search = RandomizedSearchCV(pipeline, param_distributions=params)

    # cross validating with training set
    fitted_search = rand_search.fit(X_training, y_training)

    # getting results
    print(
        f"The model with the best CV score has the following parameters: {fitted_search.best_params_}.")
    print(
        f"The best model has an accuracy score of {fitted_search.score(X_testing, y_testing)} on the test set")

#9. PCA transform independent variables
def run_PCA(components, X_train, X_test):
    '''
    param components: number of components (float between 0 and 1 or whole number) for PCA to choose from
    param X_train: training independent variables
    param X_test: testing independent variables

    return: PCA transformed X_train and X_test
    '''
    # instantiate PCA
    my_PCA = PCA(n_components=components, random_state=1)

    # fit with training
    X_train_PCA = my_PCA.fit_transform(X_train)

    # transform test
    X_test_PCA = my_PCA.transform(X_test)

    # check dimensions
    print(f"train shape: {X_train_PCA.shape}")
    print(f"test shape: {X_test_PCA.shape}")

    return X_train_PCA, X_test_PCA

#10. plot confusion matrix -- modified from https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/
def conf_matrix_plot(model, X, y):
    '''
    param model: model instantiated and fitted to training set
    param X: dependent variable
    param y: dependent variable

    return: prints confusion matrix
    '''

    y_proba = model.predict(X)

    cf_matrix=confusion_matrix(y, y_proba)

    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Greens')

    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ax.xaxis.set_ticklabels(['Not functioning','Functioning'])
    ax.yaxis.set_ticklabels(['Not functioning','Functioning'])

    plt.show()

#11. print classification report and calculate false/true positive rates for ROC visualisation
def print_report(model, X, y):
    '''
    param model: model instantiated and fitted to training set
    param X: dependent variable
    param y: dependent variable

    return: prints classification report
    return: prints confusion matrix
    return: false positive rate (fpr), true positive rate (tpr) and area under curve (auc)
    '''
    # calculate various accuracy scores

    #time process
    start=time.time()

    # prediction of our model on test set
    y_proba = model.predict_proba(X)[:, 1]

    end=time.time()

    time_predict=end-start

    # getting false positive rate and false negative rate
    fpr, tpr, thresholds_roc = roc_curve(y, y_proba)

    #getting precision/recall scores
    precision, recall, thresholds_pr = precision_recall_curve(y, y_proba)

    # storing values
    roc_auc = auc(fpr, tpr)
    pr_auc=auc(recall, precision)

    # seeing model results
    print(f'ROC AUC: {roc_auc}')
    print(f'PR AUC: {pr_auc}')
    print(classification_report(y, model.predict(X)))

    #getting confusion matrix
    conf_matrix_plot(model, X, y)

    return fpr, tpr, roc_auc, precision, recall, pr_auc, time_predict

#12. get accuracy scores
def get_scores(model, X, y):
    '''
    param model: model instantiated and fitted to training set
    param X: dependent variable
    param y: dependent variable

    return: accuracy, precision, recall and f1 score
    '''

    # store accuracy scores
    accuracy = model.score(X, y)

    # prediction of our model on test set
    y_proba = model.predict(X)

    # getting accuracy scores
    precision, recall, f1, support = precision_recall_fscore_support(y, y_proba, average='weighted')

    return accuracy, precision, recall, f1

#13. scale our data
def scaling(scaler, X_train, X_test):
    '''
    param scaler: type of scaler to use
    param X_train: training independent variables
    param X_test: testing independent variables

    return: scaled X_train and X_test
    '''
    # instantiate PCA
    my_scaler = scaler

    # fit with training
    X_train_scaled = my_scaler.fit_transform(X_train)

    # transform test
    X_test_scaled = my_scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

#14. compare ROCs on train and test
def plot_curve_roc(model_name, fpr_train_base, tpr_train_base, roc_auc_train_base, fpr_train_opt, tpr_train_opt, roc_auc_train_opt, fpr_test_base,
 tpr_test_base, roc_auc_test_base,  fpr_test_opt, tpr_test_opt, roc_auc_test_opt):

    plt.subplots(1,2, figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot([0,1], [0,1], color='black', linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve - {model_name} Train')
    plt.plot(fpr_train_base, tpr_train_base, color='green', lw=2,
        label='Baseline AUC = %0.2f' % roc_auc_train_base)
    plt.plot(fpr_train_opt, tpr_train_opt, color='red', lw=2,
        label='Optimised AUC = %0.2f' % roc_auc_train_opt)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot([0,1], [0,1], color='black', linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve - {model_name} Test')
    plt.plot(fpr_test_base, tpr_test_base, color='green', lw=2,
        label='Baseline AUC = %0.2f' % roc_auc_test_base)
    plt.plot(fpr_test_opt, tpr_test_opt, color='red', lw=2,
        label='Optimised AUC = %0.2f' % roc_auc_test_opt)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid()

    plt.show()

#15. compare Precision/Recall Curve on train and test
def plot_curve_prec_recall(model_name, recall_train_base_plot, precision_train_base_plot, pr_auc_train_base, recall_train_opt_plot, precision_train_opt_plot, pr_auc_train_opt,
 recall_test_base_plot, precision_test_base_plot,  pr_auc_test_base, recall_test_opt_plot, precision_test_opt_plot, pr_auc_test_opt):

    plt.subplots(1,2, figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot([1,0], [0,1], color='black', linestyle='--')
    plt.title('Precision/Recall Curve - {model_name} Train')
    plt.plot(recall_train_base_plot, precision_train_base_plot, color='green', lw=2,
        label='Baseline AUC = %0.2f' % pr_auc_train_base)
    plt.plot(recall_train_opt_plot, precision_train_opt_plot, color='red', lw=2,
        label='Optimised AUC = %0.2f' % pr_auc_train_opt)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot([1,0], [0,1], color='black', linestyle='--')
    plt.title('Precision/Recall Curve - {model_name} Test')
    plt.plot(recall_test_base_plot, precision_test_base_plot, color='green', lw=2,
        label='Baseline AUC = %0.2f' % pr_auc_test_base)
    plt.plot(recall_test_opt_plot, precision_test_opt_plot, color='red', lw=2,
        label='Baseline AUC = %0.2f' % pr_auc_test_opt)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid()

    plt.show()

#16. bar chart of feature/coefficient importance
def coeff_bar_chart(features, X_columns, t=True):
    if t==True:

        #what are the most important coefficients for the best model?
        coefficients=pd.DataFrame(data=features, columns=X_columns).T.reset_index()

    else:
        coefficients=pd.DataFrame(data=features, index=X_columns).reset_index()
    
    #rename columns
    coefficients.set_axis(['feature', 'coefficient'], axis=1, inplace=True)

    #sorting values
    coefficients=coefficients.sort_values('coefficient', ascending=False)

    #get our hypotheses
    hypotheses_df=pd.read_csv(data_filepath + 'ta_hypotheses.csv', index_col=False)

    #merge coefficients and hypotheses
    coefficients_df=pd.merge(
        coefficients,
        hypotheses_df,
        how="left",
        on='feature',
        copy=True,
    )
    # visualise coefficient importance
    plt.figure(figsize=(10,10))

    sns.barplot(data=coefficients_df, x='coefficient', y='feature', hue='hypotheses', palette=palette, orient='horizontal', dodge=False)

    plt.xlabel('Coefficient')
    plt.ylabel('Feature')

    plt.title('Feature importance')
    
    plt.xticks()
    plt.yticks()
    plt.legend()
    plt.show()