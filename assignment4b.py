import researchpy as rp
import pandas as pd

#load in data
df = pd.read_csv('data/data.csv')
df.columns

#Prints out descriptive information for DataFrame.
rp.codebook(df)

#getting descriptive of single/continuous variables
rp.summary_cont(df[['Age', 'HR', 'sBP']])

#getting descriptive of categorical variables
rp.summary_cat(df[['Group', 'Smoke']])


