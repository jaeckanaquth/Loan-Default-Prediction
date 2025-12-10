import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('data/Dataset.csv', low_memory=False)

# Choose columns relevant to finding loan default based on Data_Dictionary.csv
chosen_col = [
    'ID',                       # identifier
    'Client_Income',            # $
    'Car_Owned',                # 0/1
    'Bike_Owned',               # 0/1
    'Active_Loan',              # 0/1
    'House_Own',                # 0/1
    'Child_Count',              # Count (can be safely filled with zero)
    'Credit_Amount',            # $
    'Loan_Contract_Type',       # string/categorical
    'Loan_Annuity',             # $
    'Client_Marital_Status',    # string/categorical
    'Client_Housing_Type',      # string/categorical
    'Employed_Days',            # integer
    'Client_Occupation',        # string/categorical
    'Client_Family_Members',    # integer (fill 0 if missing)
    'Type_Organization',        # string/categorical
    'Score_Source_1',           # float score
    'Score_Source_2',           # float score
    'Score_Source_3',           # float score
    'Credit_Bureau',            # integer (enquiries count, fill 0 if missing)
    'Default'                   # target variable
]

# Columns to fill NA with zero (mainly binary indicators, counts, and IDs)
zero_col = [
    'ID', 
    'Car_Owned', 
    'Bike_Owned', 
    'Active_Loan', 
    'House_Own',
    'Child_Count', 
    'Employed_Days', 
    'Client_Family_Members', 
    'Credit_Bureau',
    'Default'
]

# Columns with $ values or float scores; fill NA with median
median_col = [
    'Client_Income', 
    'Credit_Amount', 
    'Loan_Annuity', 
    'Score_Source_1', 
    'Score_Source_2', 
    'Score_Source_3'
]


# Columns to fill NA with mode (categorical/string columns)
# These will be one-hot encoded later
mod_col = [
    'Loan_Contract_Type',
    'Client_Marital_Status', 
    'Client_Housing_Type', 
    'Client_Occupation', 
    'Type_Organization'
]

# handle the missing values
for col in zero_col:
    # Replace non-numeric and invalid entries with NaN, then fillna with 0, then cast to int
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
for col in median_col:
    df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace('&', '', regex=False)
    df[col] = df[col].replace(['#VALUE!', '', 'nan', 'None'], np.nan)
    df[col] = df[col].astype(float)
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
    df[col] = df[col].fillna(df[col].median())
for col in mod_col:
    # Fill NA with mode
    df[col] = df[col].fillna(df[col].mode()[0])
    data = df[col].to_list()
    data_reshaped = np.array(data).reshape(-1, 1)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 
    one_hot_encoded_data = encoder.fit_transform(data_reshaped)
    df[col] = one_hot_encoded_data

df_selected = df[chosen_col]
if not os.path.exists('data/processed'):
    os.makedirs('data/processed')
df_selected.to_csv('data/processed/loan.csv', index=False)


