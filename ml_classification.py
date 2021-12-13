import pandas as pd
import numpy as np
import pickle
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


wgndDataLoc = r'C:\Users\Milan\OneDrive\Desktop\Said\Art\ArtistData\genderClassification\rawdata\wgnd_noctry.csv'
SSADAta = r'C:/Users/Milan/OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/names'
chinaData = r'C:/Users/Milan/OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/name_gender_china.csv'
UNKData = r'C:/Users/Milan/OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/ukProcessed.csv'
hunDataMale = r'C:/Users/Milan/OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/hun_male.txt'
hunDataFemale = r'C:/Users/Milan/OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/hun_female.txt'


# Import gender databases to Dataframes.
# Keeping only the relevant data: name and gender.
df1r = pd.read_csv(wgndDataLoc)
df1 = df1r[['name', 'gender']]
df1 = df1[df1.gender != '?']
df1 = df1[df1['name'].map(len) > 3]
del df1r
# 1: Female, 0: Male
df1.gender.replace({'F': 1, 'M': 0}, inplace=True)


import pinyin
df2 = pd.read_csv(chinaData)
translated = []
for row in df2.iterrows():
    try:
        name = pinyin.get(row[1]['name'], format='strip')
        translated.append(name)
    except:
        translated.append('')
df2['nameT'] = translated
# 1: Female, 0: Male
df2.gender.replace({'男': 0, '女': 1}, inplace=True)


df3r = pd.read_csv(UNKData, error_bad_lines=False)
df3 = df3r[['Name', 'prob.gender']]
df3 = df3.rename(columns={'prob.gender': 'gender', 'Name': 'name'})
df3 = df3[df3.gender != 'Unknown']
del df3r
df3.gender.replace({'Female': 1, 'Male': 0}, inplace=True)
df3 = df3[df3['name'].map(len) > 3]


df4r = pd.DataFrame()
for file in os.listdir(SSADAta):
    if file.endswith(".txt"):
        dfx = pd.read_csv(SSADAta + '/' + file, delimiter=',', index_col=False, header=None, names=['name', 'gender'])
        df4r = df4r.append(dfx, ignore_index=True)
df4 = df4r.drop_duplicates()
df4.gender.replace({'F': 1, 'M':0}, inplace=True)
df4 = df4[df4['name'].map(len) > 3]


femaleNames = []
with open(hunDataFemale, 'r') as r:
    for line in r:
        femaleNames.append(line.strip())
dfFH = pd.DataFrame(columns=['name', 'gender'])
dfFH['name'] = femaleNames
dfFH['gender'] = 1
maleNames = []
with open(hunDataMale, 'r') as r:
    for line in r:
        maleNames.append(line.strip())
dfMH = pd.DataFrame(columns=['name', 'gender'])
dfMH['name'] = maleNames
dfMH['gender'] = 0

df5 = pd.concat([dfFH, dfMH])
del dfFH, dfMH



df = pd.concat([df1, df3, df4, df5])
df = df.drop_duplicates()


# Extracts the unique features of the name.
# Can be improved upon in the future.
# Research on most effective features: 
# https://www.aclweb.org/anthology/U14-1021.pdf
def extractFeatures(string):
    name = string.lower().lstrip().strip()
    length = len(name)

    firstLetter = name[0]
    secondLetter = name[1]
    firstTwoLetters = name[0:2]
    firstThreeLetters = name[0:3]

    lastLetter = name[-1]
    oneBeforeLastLetter = name[-2]
    lastTwoLetters = name[-2:]
    lastThreeLetters = name[-3:]

    letterVowels = ''
    vowelLetters = ''
    consonantLetters = ''
    
    vowels = ['a', 'e', 'i', 'o', 'u']
    consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q',
                  'r', 's', 't', 'v', 'w', 'x', 'y', 'z'
                  ]

    for letter in string.lower():
        if letter in vowels:
            letterVowels = letterVowels + '1'
            vowelLetters = vowelLetters + str(letter)
        elif letter in consonants:
            letterVowels = letterVowels + '0'
            consonantLetters = consonantLetters + str(letter)
   
    # According to research n-gram = 3 yields the best results. 
    # The model additionally includes the first and last letters of the name.
    # Limitations: names shorter than 3 letters cannot be processed by the model.
    retDict = {
        'firstLetter': firstLetter,
        'lastLetter': lastLetter,
        # 'secondLetter': secondLetter,
        # 'oneBeforeLastLetter': oneBeforeLastLetter,
        'firstTwoLetters': firstTwoLetters,
        'lastTwoLetters': lastTwoLetters,
        'firstThreeLetters': firstThreeLetters,
        'lastThreeLetters': lastThreeLetters,
        
        # These parameters might make the model less accurate (based on tests).
        'vowelLetters': vowelLetters,
        # 'letterVowels': letterVowels,
        # 'consonantLetters': consonantLetters
        }

    return retDict


featureExtractor = np.vectorize(extractFeatures)


dT = DecisionTreeClassifier()
dV = DictVectorizer()

Xfeatures = featureExtractor(df['name'])
Ylabels = df['gender']

xTrain, xTest, yTrain, yTest = train_test_split(Xfeatures, Ylabels, test_size=0.1)

xFeatures = dV.fit_transform(xTrain)

dT.fit(xFeatures, yTrain)


with open('dTM.pkl', 'wb') as w:
    pickle.dump(dT, w)


def genderPredictor(name):
    transform_dv = dV.transform(extractFeatures(name))
    vector = transform_dv.toarray()
    if dT.predict(vector) == 1:
        return 1
    else:
        return 0
