##############################################
# Import Libraries #
# #############################################
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import shapiro
from scipy.stats import normaltest

terrorism_df = pd.read_csv('globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')

def missing_data(data):

    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    df_missing = pd.concat([total, percent], axis=1, keys=['Total NaN Values', 'Percentage of NaN Values'])

    data_clean = data.copy()

    missing_col = ['nkill','nwound']
    for i in missing_col:
        data_clean.loc[data_clean.loc[:, i].isnull(), i] = math.ceil(data_clean.loc[:, i].mean())

    data_clean.dropna(axis=1, inplace=True)

    data_clean.drop(['eventid','dbsource','INT_LOG','INT_IDEO','INT_MISC','INT_ANY'], axis=1, inplace=True)

    description_df = data_clean.describe()

    return  data_clean, description_df

def normality_shapiro(data):
    stat, p = shapiro(data)
    print('Shapiro-Wilk Test')
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

def normality_pearsons(data):

    print('DAgostino and Pearsons Test')
    stat, p = normaltest(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

data_clean, description_df = missing_data(terrorism_df)
kills = data_clean['nkill']
wounded = data_clean['nwound']

normality_shapiro(kills)
normality_pearsons(kills)

normality_shapiro(wounded)
normality_pearsons(wounded)
