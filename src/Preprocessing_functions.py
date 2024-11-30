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
from sklearn.feature_selection import RFE, RFECV

CODE_COLUMNS = ['Industry Code', 'WCIO Cause of Injury Code',
       'WCIO Nature of Injury Code', 'WCIO Part Of Body Code']

DESCRIPTION_COLUMNS = ['WCIO Cause of Injury Description', 
                       'WCIO Nature of Injury Description','WCIO Part Of Body Description',
                       'Industry Code Description']

BOOLEAN_COLUMNS = ['Alternative Dispute Resolution', 'Attorney/Representative','COVID-19 Indicator']

# convert all date columns to datetime format
def convert_to_datetime(X_train, X_val, columns):
    '''
    Convert all columns to datetime format
    '''
    for col in columns:
        X_train[col] = pd.to_datetime(X_train[col], errors='coerce')
        X_val[col] = pd.to_datetime(X_val[col], errors='coerce')
    return X_train, X_val

# Convert all columns to timestemp format
def convert_to_timestamp(X_train, X_val, columns):
    '''
    Convert all columns to timestamp format
    '''
    for col in columns:
        X_train[col] = pd.to_datetime(X_train[col], errors='coerce')
        X_val[col] = pd.to_datetime(X_val[col], errors='coerce')    

        X_train[col].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
        X_val[col].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    return X_train, X_val

#function to transform Y and N into boolean while preserving the NaNs
def convert_to_bool(X_train, X_val, col_names=BOOLEAN_COLUMNS):
    '''
    Convert 'Y' and 'N' to True and False respectively while preserving NaNs

    Parameters:
    X_train: DataFrame
    X_val: DataFrame
    col_names: list default: BOOLEAN_COLUMNS
    '''
    for col_name in col_names:
        X_train[col_name] = X_train[col_name].map({'Y': True, 'N': False, np.nan: np.nan})
        X_val[col_name] = X_val[col_name].map({'Y': True, 'N': False, np.nan: np.nan})
    return X_train, X_val

# Create new features based on the binned groups of the original features
def newFeature_binnedGroups(X_train, X_val, X_test, columns, bins=4):
    '''
    Create new features based on the binned groups of the original features

    Parameters:
    X_train: DataFrame
    X_val: DataFrame
    X_test: DataFrame
    columns: list
    bins: int default: 4
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

def remove_outliers_iqr(X_train, X_val, columns):
    """
    Remove rows with outliers using the IQR method.
    
    """
    for col in columns:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove outliers from X_train
        X_train = X_train[(X_train[col] >= lower_bound) & (X_train[col] <= upper_bound)]
        
        # Remove outliers from X_val (based on X_train bounds)
        X_val = X_val[(X_val[col] >= lower_bound) & (X_val[col] <= upper_bound)]

    return X_train, X_val

def convert_dates_to_timestamps(df):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    for col in df.columns:
        if 'Date' in col:
            # Convert the column to datetime, coerce invalid parsing to NaT
            df[col] = pd.to_datetime(df[col], errors='coerce')

        #Convert to timestamp only for valid datetime values
            df[col] = df[col].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)

    return df


def impute_mean_numerical(X_train, X_val, columns):
    """
    Impute missing values for continuous (numerical) variables with the mean of the training data.
    
    """

    for col in columns:
        # Calculate the mean of the column in the training data
        mean_value = X_train[col].mean()
        
        # Impute missing values in both datasets
        X_train[col].fillna(mean_value, inplace=True)
        X_val[col].fillna(mean_value, inplace=True)

    return X_train, X_val

def impute_mode_categorical(X_train, X_val, columns):
    """
    Impute missing values for categorical variables with the mode of the training data.
    
    """

    for col in columns:
        # Calculate the mode of the column in the training data
        mode_value = X_train[col].mode()[0]  # Mode can return multiple values; take the first
        
        # Impute missing values in both datasets
        X_train[col].fillna(mode_value, inplace=True)
        X_val[col].fillna(mode_value, inplace=True)

    return X_train, X_val

def compute_most_frequent_code_per_description(df, code_columns):
    '''
    This is an auxiliary function to help fill the
    missing values for Code columns
    '''

    # List to store results as tuples of (description, most_frequent_code) per column
    results = {}

    for code_col in code_columns:
        if code_col != 'Industry Code':
            
            description_col = code_col.replace('Code', 'Description')
        else:
            description_col = 'Industry Code Description'
        
        # Calculate most frequent code for each unique description
        most_frequent_code_series = (
            df.groupby(description_col)[code_col]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        )
        
        # Convert to numpy array
        results[code_col] = np.array(
            list(zip(most_frequent_code_series.index, most_frequent_code_series.values)),
            dtype=[('description', 'O'), ('most_frequent_code', 'O')]
        )

    return results

def fill_missing_codes_description_based(X_train, X_val):
   '''
   This function fills the missing code values in the dataset
   using the most frequent code for each description with the help
   of the compute_most_frequent_code_per_description function.
   '''

   X_train_most_frequent_codes = compute_most_frequent_code_per_description(X_train, CODE_COLUMNS)
   X_val_most_frequent_codes = compute_most_frequent_code_per_description(X_val, CODE_COLUMNS)

   for code_col in CODE_COLUMNS:
       
       # Extract the numpy array for the current code column
       X_train_most_frequent_array = X_train_most_frequent_codes[code_col]
       
       # Map description values to their most frequent codes using numpy indexing
       X_train_description_to_code = {desc: code for desc, code in X_train_most_frequent_array}

       # Fill missing values in the DataFrame (X_train) using the mapping derived from X_train.
       X_train[code_col] = X_train.apply(
          lambda row: (
              # Look up the most frequent code for the description in the mapping (X_train_description_to_code).
              # If a matching description is found in the mapping, use its associated code.
              X_train_description_to_code.get(row[code_col.replace('Code', 'Description')], row[code_col])
              if (
                  # Only fill the value if the current code is missing (NaN)...
                  pd.isna(row[code_col])
                  # ...and the corresponding description is NOT missing (NaN).
                  and not pd.isna(row[code_col.replace('Code', 'Description')])
              )
              # If the conditions aren't met, keep the original value of the code column.
              else row[code_col]
          ),
          # Apply the function row-wise across the DataFrame.
          axis=1
       )

       # Fill missing values in the DataFrame (X_val) using the mapping derived from X_train.
       X_val[code_col] = X_val.apply(
           lambda row: (
               # Look up the most frequent code for the description in the mapping (X_train_description_to_code).
               # This ensures that X_val is filled based on the same mapping as X_train, avoiding data leakage.
               X_train_description_to_code.get(row[code_col.replace('Code', 'Description')], row[code_col])
               if (
                   # Only fill the value if the current code is missing (NaN)...
                   pd.isna(row[code_col])
                   # ...and the corresponding description is NOT missing (NaN).
                   # Missing codes with missing descriptions will be filled with mode later.
                   and not pd.isna(row[code_col.replace('Code', 'Description')])
               )
               # If the conditions aren't met, keep the original value of the code column.
               else row[code_col]
           ),
           # Apply the function row-wise across the DataFrame.
           axis=1
       )

       X_train = X_train.infer_objects(copy = False) 
       X_val = X_val.infer_objects(copy = False)

       return X_train, X_val
   
def fillna_zip_code(X_train, X_val):
    """
    Fills missing 'Zip Code' values in train and validation datasets based on modes.
    first if their is a conty of injury and district name match then fill with mode zip code
    if not fill with mode zip code of district name
    """
    # Save original indices
    original_index_train = X_train.index
    original_index_val = X_val.index

    # Calculate mode Zip Codes for each group in X_train
    mode_zip_codes_train = (
        X_train.groupby(['County of Injury', 'District Name'])['Zip Code']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
    )
    mode_zip_district_train = (
        X_train.groupby('District Name')['Zip Code']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
    )
    mode_zip_district_train.columns = ['District Name', 'Zip Code_mode_district']

    # Define a helper function for filling missing Zip Codes
    def fill_zip_codes(dataset):
        # Merge with mode_zip_codes based on County of Injury and District Name
        dataset = dataset.merge(
            mode_zip_codes_train,
            on=['County of Injury', 'District Name'],
            how='left',
            suffixes=('', '_mode')
        )

        # Fill missing Zip Code values with the mode from X_train
        dataset['Zip Code'] = dataset['Zip Code'].combine_first(dataset['Zip Code_mode'])

        # Merge with mode_zip_district for District Name fallback
        dataset = dataset.merge(
            mode_zip_district_train,
            on='District Name',
            how='left'
        )

        # Fill remaining missing Zip Code values with District Name mode
        dataset['Zip Code'] = dataset['Zip Code'].combine_first(dataset['Zip Code_mode_district'])

        # Drop the added columns
        dataset.drop(columns=['Zip Code_mode', 'Zip Code_mode_district'], inplace=True)

        return dataset

    # Apply the helper function to each dataset
    X_val = fill_zip_codes(X_val)
    X_train = fill_zip_codes(X_train)

    # Restore original indices
    X_train.index = original_index_train
    X_val.index = original_index_val

    return X_train, X_val

def fillnan_accident_date(X_train,X_val):
    """
    this function fills the missing values in the 'Accident Date' column with the mean difference between 'C-2 Date' and 'Accident Date'
    """
    X_train['time_diff C2 accident'] = X_train['Accident Date']-X_train['C-2 Date']

    mean_difference_c2_accident = X_train['time_diff C2 accident'].mean()
    
    X_train['Accident Date'].fillna(X_train['C-2 Date'] - mean_difference_c2_accident, inplace=True)
    X_val['Accident Date'].fillna(X_val['C-2 Date'] - mean_difference_c2_accident, inplace=True)
    return X_train, X_val

def fillnan_birth_year(X_train, X_val):
    def process_birth_year_columns(df):
        """
        this function processes the 'Birth Year' column by filling NaN values with calculated birth years in seconds
        """
        accident_col = df['Accident Date'].copy()
        # Replace 0.0 values with NaN in 'Birth Year'
        df['Birth Year'] = df['Birth Year'].replace(0.0, np.nan)

        # Ensure 'Accident Date' is in datetime format
        accident_col = pd.to_datetime(accident_col, errors='coerce')

        # Fill NaN values with calculated birth years in seconds
        if df['Birth Year'].isna().any():
            # Calculate the birth year by subtracting 'Age at Injury' (converted to years * 365.25 days) from 'Accident Date'
            calculated_birth_years = accident_col - pd.to_timedelta(df['Age at Injury'] * 365.25, unit='D')

            # Extract the year from the resulting date
            df['Birth Year'].fillna(calculated_birth_years.dt.year, inplace=True)

        # Ensure 'Birth Year' is numeric (in case it was converted to datetime)
        df['Birth Year'] = pd.to_numeric(df['Birth Year'], errors='coerce')

        # Convert the 'Birth Year' to datetime using the first day of the year (01-01)
        df['Birth Year'] = pd.to_datetime(df['Birth Year'], format='%Y', errors='coerce')

        # Convert to Unix timestamp using timestamp() method
        df['Birth Year'] = df['Birth Year'].apply(lambda x: x.timestamp() if pd.notnull(x) else None)

        return df
    """
    this apply the function just above to the train and validation data
    """
    X_train = process_birth_year_columns(X_train)
    X_val = process_birth_year_columns(X_val)
    return X_train, X_val
    
def fill_missing_with_mode(X_train, X_val):
    '''
    Fill missing values with the mode of the column
    '''
    for col in CODE_COLUMNS:
        # Get the mode (most frequent value)
        mode = X_train[col].mode().iloc[0]

        X_train.loc[:, col] = X_train[col].fillna(mode)
        X_val.loc[:,col] = X_val[col].fillna(mode)
    
    X_train = X_train.infer_objects(copy=False)
    X_val = X_val.infer_objects(copy=False)

    return X_train, X_val

def drop_description_columns(X_train, X_val):
    '''
    Drop the description columns because they hold no additional
    information compared to the codes
    '''
    X_train = X_train.drop(DESCRIPTION_COLUMNS, axis=1)
    X_val = X_val.drop(DESCRIPTION_COLUMNS, axis=1)

    return X_train, X_val

def feature_creation_has_Cdate (X_train, X_val):
    X_train['Has C-3 Date'] = X_train['C-3 Date'].apply(lambda x: 0 if pd.isna(x) else 1)
    X_val['Has C-3 Date'] = X_val['C-3 Date'].apply(lambda x: 0 if pd.isna(x) else 1)
    X_train['Has C-2 Date'] = X_train['C-2 Date'].apply(lambda x: 0 if pd.isna(x) else 1)
    X_val['Has C-2 Date'] = X_val['C-2 Date'].apply(lambda x: 0 if pd.isna(x) else 1)
    return X_train, X_val

def feature_selection_rfe(X_train, y_train, n_features, model):
    """
    Applies Recursive Feature Elimination (RFE) for feature selection.

    Parameters:
    - model: The base model for RFE (e.g., RandomForestClassifier, LogisticRegression).
    - X_train: Training features.
    - y_train: Training labels.
    - n_features: Number of features to select.

    Returns:
    - X_train_selected: Transformed training features with selected features.
    - selected_features: List of selected feature names.
    - feature_ranking: Pandas DataFrame with feature names and their rankings.
    """
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    X_train_selected = rfe.fit_transform(X_train, y_train)

    # Extract feature rankings
    feature_ranking = pd.DataFrame({
        'Feature': X_train.columns,
        'Ranking': rfe.ranking_
    }).sort_values(by='Ranking')

    # Get selected feature names
    selected_features = feature_ranking[feature_ranking['Ranking'] == 1]['Feature'].tolist()

    return X_train_selected, selected_features, feature_ranking


def feature_selection_rfecv(X_train, y_train, model, cv_folds=5, scoring='accuracy'):
    """
    Applies Recursive Feature Elimination with Cross-Validation (RFECV) for feature selection.

    Parameters:
    - model: The base model for RFECV (e.g., RandomForestClassifier, LogisticRegression).
    - X_train: Training features as a Pandas DataFrame.
    - y_train: Training labels.
    - cv_folds: Number of cross-validation folds (default: 5).
    - scoring: Scoring metric for cross-validation (default: 'accuracy').

    Returns:
    - X_train_selected: Transformed training features with selected features.
    - selected_features: List of selected feature names.
    - feature_ranking: Pandas DataFrame with feature names and their rankings.
    - optimal_num_features: Optimal number of features determined by RFECV.
    """
    # Initialize RFECV
    rfecv = RFECV(estimator=model, step=1, cv=cv_folds, scoring=scoring)
    rfecv.fit(X_train, y_train)

    # Extract feature rankings
    feature_ranking = pd.DataFrame({
        'Feature': X_train.columns,
        'Ranking': rfecv.ranking_,
        'Support': rfecv.support_
    }).sort_values(by='Ranking')

    # Get selected feature names
    selected_features = feature_ranking[feature_ranking['Support'] == True]['Feature'].tolist()

    # Transform X_train to include only selected features
    X_train_selected = X_train[selected_features]

    # Get optimal number of features
    optimal_num_features = rfecv.n_features_

    return X_train_selected, selected_features, feature_ranking, optimal_num_features