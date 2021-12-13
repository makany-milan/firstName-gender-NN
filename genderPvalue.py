# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:50:33 2021

@author: Milan Makany
"""

import pandas as pd
import os

SSADAtaLoc = r'C:/Users/Milan/OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/names'
UNKData = r'C:/Users/Milan/OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/ukProcessed.csv'


df1r = pd.DataFrame()
for file in os.listdir(SSADAtaLoc):
    if file.endswith(".txt"):
        dfx = pd.read_csv(SSADAtaLoc + '/' + file, delimiter=',', index_col=False, header=None, names=['name', 'gender', 'count'])
        df1r = df1r.append(dfx, ignore_index=True)
        
df1 = df1r.groupby(['name', 'gender'], as_index=False).sum()
df1Total = df1.groupby(['name']).sum()
df1Total = df1Total.rename(columns={'count': 'total'})

df1 = pd.merge(df1, df1Total, how='left', on=['name'])

df1['prob'] = df1['count']/df1['total']
df1.gender.replace({'F': 1, 'M':0}, inplace=True)
df1 = df1.dropna(subset=['prob'])

        

df2r = pd.read_csv(UNKData, error_bad_lines=False)
df2 = df2r[['Name', 'count.male', 'count.female']]
df2 = df2.rename(columns={'Name': 'name'})
df2['total'] = df2['count.male'] + df2['count.female']

df2.loc[df2['count.male'] > df2['count.female'], 'gender'] = 0
df2.loc[df2['count.male'] < df2['count.female'], 'gender'] = 1

df2.loc[df2['gender'] == 0, 'prob'] = df2['count.male'] / df2['total']
df2.loc[df2['gender'] == 1, 'prob'] = df2['count.female'] / df2['total']

df2 = df2.dropna(subset=['prob'])

df2.gender.replace({'Female': 1, 'Male': 0}, inplace=True)

df = pd.merge(df1, df2, how='outer', on=['name'])
df.loc[df['gender_x'] == df['gender_y'], 'prob'] = (df['prob_x'] + df['prob_y']) / 2