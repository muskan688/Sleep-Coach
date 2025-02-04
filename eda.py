import pandas as pd

#Load the data
data = pd.read_csv('cmu-sleep.csv')

#Data Overview
print(data.head(2))

#Data Information
print(data.info())

#Statistical summary
print(data.describe())

#Missing Value
print(data.isnull().sum())









