#importing relevant packages
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

#function to convert an object column to a datetime
def date_converter(df, col):
    '''
    param df: dataframe
    param col: 'object' column in dataframe that needs to be converted
                to datetime in datafrane

    return: changes col to a datetime column
    '''
    df[col]=pd.to_datetime(df[col])

#function to convert an object column to a float
def float_converter(df, col):
    '''
    param df: dataframe
    param col: 'object' column in dataframe that needs to be converted
                to float in datafrane

    return: changes col to a float column
    '''
    df[[col]]=df[[col]].astype('float32')

#function to fill missing values with regional median
def missing_to_regional_median(df, col, region):
    '''
    param df: dataframe
    param col: column with missing values that needs to be filled
    param region: regional level at which we want to choose the median from

    return: fills missing value in col with median of its region
    '''
    df[col] = df[col].fillna(df.groupby(region)[col].transform('median'))


#Function calculating the correlation between X (independent variables) and y (dependent variable)

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


# Check LINEARITY with function plotting each independent with the dependent variable

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


#Function showing boxplots to look at correlation between all independent variables and dependent variable
def box_loop(X,y):
    '''
    param X: dataframe or series of independent variables
    param y: series of dependent variables

    return: subplots with a boxplot for each numerical independent variable with categorical dependent variable
    '''  
    #Creating subplot for readability
    plt.subplots(19,2, figsize=(20,100))
    
    #Loop through all independent variables and create scatter plot with dependent variable
    for i, col in enumerate(X.columns, 1):
        plt.subplot(19,2,i)
        sns.boxplot(y=X[col], x=y)
        plt.ylabel(X[col].name)
        plt.xlabel(y.name)
    
    plt.tight_layout()
    plt.show()