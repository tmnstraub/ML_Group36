#Importing Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Create new features based on the binned groups of the original features
def newFeature_binnedGroups(X_train, X_val, X_test, columns, bins=4):
    '''
    Create new features based on the binned groups of the original features

    Parameters:
    X_train: DataFrame
    X_val: DataFrame
    X_test: DataFrame
    columns: list
    bins: int
    '''

    for col in columns:
        # Define bins based on training data
        train_bins = pd.qcut(X_train[col], q=bins, retbins=True)[1]  # Get bin edges

        # Apply the bins to all datasets
        X_train[f'{col} Group'] = pd.cut(X_train[col], bins=train_bins, labels=False, include_lowest=True)
        X_val[f'{col} Group'] = pd.cut(X_val[col], bins=train_bins, labels=False, include_lowest=True)
        X_test[f'{col} Group'] = pd.cut(X_test[col], bins=train_bins, labels=False, include_lowest=True)

        X_train[f'{col} Group'] = X_train[f'{col} Group'].astype(str)
        X_val[f'{col} Group'] = X_val[f'{col} Group'].astype(str)
        X_test[f'{col} Group'] = X_test[f'{col} Group'].astype(str)

    return X_train, X_val, X_test

# Create new feature month based on the date feature
def newFeature_month(X_train, X_val, X_test, columns):
    '''
    Create new feature month based on the date feature. 
    Need to be applied to columns with datetime format.

    Parameters:
    X_train: DataFrame
    X_val: DataFrame
    X_test: DataFrame
    date_col: str
    '''

    for col in columns:
        X_train[f'{col} Month'] = X_train[col].dt.month
        X_val[f'{col} Month'] = X_val[col].dt.month
        X_test[f'{col} Month'] = X_test[col].dt.month

    return X_train, X_val

# Create new feature days since the last event
def newFeature_daysBetween(X_train, X_val, X_test, firstDate, secondDate):
    '''
    Create new feature days since the last event. 
    Need to be applied to columns with datetime format.

    Parameters:
    X_train: DataFrame
    X_val: DataFrame
    X_test: DataFrame
    date_col: str
    '''

    X_train[f'Days Between {firstDate} and {secondDate}'] = (X_train[secondDate].max() - X_train[firstDate]).dt.days
    X_val[f'Days Between {firstDate} and {secondDate}'] = (X_val[secondDate].max() - X_val[firstDate]).dt.days
    X_test[f'Days Between {firstDate} and {secondDate}'] = (X_test[secondDate].max() - X_test[firstDate]).dt.days

    return X_train, X_val, X_test


# Push outliers to upper and lower bounds
def outliers_iqr(X_train, X_val, columns):
    '''
    Push outliers to upper and lower bounds using the IQR method
    '''
    for col in columns:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X_train[col] = np.where(X_train[col] < lower_bound, lower_bound, X_train[col])
        X_train[col] = np.where(X_train[col] > upper_bound, upper_bound, X_train[col])
        X_val[col] = np.where(X_val[col] < lower_bound, lower_bound, X_val[col])
        X_val[col] = np.where(X_val[col] > upper_bound, upper_bound, X_val[col])
    return X_train, X_val

# Push outliers to upper bound and lower bound but give the possibilty to choos a specific value for both bounds. If the value is None, the bound will be calculated using the IQR method lower and upper bound is not filled use the IQR method
def outliers_specific(X_train, X_val, columns, lower_bound=None, upper_bound=None):
    '''
    Push outliers to upper bound and lower bound but give the possibilty to choose a specific value for both bounds. 
    If the any boundary value is None, the bound will be calculated using the IQR method
    '''
    for col in columns:
        if lower_bound is None or upper_bound is None:
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        X_train[col] = np.where(X_train[col] < lower_bound, lower_bound, X_train[col])
        X_train[col] = np.where(X_train[col] > upper_bound, upper_bound, X_train[col])
        X_val[col] = np.where(X_val[col] < lower_bound, lower_bound, X_val[col])
        X_val[col] = np.where(X_val[col] > upper_bound, upper_bound, X_val[col])
    return X_train, X_val


# MinMax Scaler
def scaling_minmax(X_train, X_val, columns):
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=columns)
    return X_train_scaled, X_val_scaled

# Standard Scaler
def scaling_standard(X_train, X_val, columns):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=columns)
    return X_train_scaled, X_val_scaled

# Label Encoder for target variable
def encoding_label(y_train, y_val):
    '''
    Label Encoder for target variable
    '''
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    return y_train_encoded, y_val_encoded

# OneHot Encoder for categorical variables with low cardinality
def encoding_onehot(X_train, X_val, columns):
    '''
    OneHot Encoder for categorical variables with low cardinality
    '''
    # Changing datatype to string for encoding
    X_train[columns] = X_train[columns].astype(str)
    X_val[columns] = X_val[columns].astype(str)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_encoded = pd.DataFrame(ohe.fit_transform(X_train[columns]))
    X_val_encoded = pd.DataFrame(ohe.transform(X_val[columns]))
    return X_train_encoded, X_val_encoded

# Frequency Encoding for categorical variables with high cardinality -> not filling unseen values
def encoding_frequency1(X_train, X_val, columns):
    '''
    Frequency Encoding for categorical variables with high cardinality -> not filling unseen values
    '''
    for col in columns:
        # Changing datatype to string for encoding
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)

        fe = X_train.groupby(col).size() / len(X_train)
        X_train[col] = X_train[col].apply(lambda x : fe[x])
        X_val[col] = X_val[col].apply(lambda x : fe[x])
    return X_train, X_val

# Frequency Encoding for categorical variables with high cardinality -> filling unseen values
def encoding_frequency2(X_train, X_val, columns):
    '''
    Frequency Encoding for categorical variables with high cardinality -> filling unseen values
    '''
    
    for col in columns:
        # Changing datatype to string for encoding
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)

        frequency_map = X_train[col].value_counts(normalize=True)
        X_train[col] = X_train[col].map(frequency_map)
        X_val[col] = X_val[col].map(frequency_map).fillna(0)
    return X_train, X_val

# Ordinal Encoding for categorical variables with ordinality
def encoding_ordinal(X_train, X_val, columns):
    '''
    Ordinal Encoding for categorical variables with ordinality
    '''
    oe = OrdinalEncoder()
    X_train_encoded = pd.DataFrame(oe.fit_transform(X_train[columns]), columns=columns)
    X_val_encoded = pd.DataFrame(oe.transform(X_val[columns]), columns=columns)
    return X_train_encoded, X_val_encoded




