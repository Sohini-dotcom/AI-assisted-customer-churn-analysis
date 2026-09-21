import pandas as pd

df = pd.read_csv('data/raw/Telco-Customer-Churn.csv')   # load the raw file

print(df.shape)                                          # rows, columns
print(df.isna().sum())                                   # missing values per column
print(df.duplicated().sum())                             # fully duplicated rows
print(df.customerID.is_unique)                           # does any ID repeat?
print((df['TotalCharges'].str.strip() == '').sum())      # blank-space values