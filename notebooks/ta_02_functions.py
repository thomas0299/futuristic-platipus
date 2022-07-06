#importing relevant packages
import pandas as pd

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
