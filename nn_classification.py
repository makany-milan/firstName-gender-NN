#%%
from operator import index
import numpy as np
from numpy.core.numeric import full
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle


from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


LAPTOP = False

if LAPTOP == False:
    CD = r'C:/Users/Milan/'
elif LAPTOP == True:
    CD = r'C:/Users/makan/'


SPECIAL_CHARS = {
    'á': 'a',
    'ä': 'a',
    'é': 'e',
    'ü': 'u',
    'ű': 'u',
    'ö': 'o',
    'ő': 'o'
}


def removeSpecialChars(string):
    retstr = ''
    for letter in string:
        if letter in SPECIAL_CHARS.keys():
            retstr = retstr + SPECIAL_CHARS[letter]
        else:
            retstr = retstr + letter
    
    return retstr


def importDataMultipleLoc():
    wgndDataLoc = rf'{CD}OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/wgnd_noctry.csv'
    SSADAta = rf'{CD}OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/names'
    # chinaData = r'{CD}OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/name_gender_china.csv'
    UNKData = rf'{CD}OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/ukProcessed.csv'
    hunDataMale = rf'{CD}OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/hun_male.txt'
    hunDataFemale = rf'{CD}OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/hun_female.txt'
    bordex = rf'{CD}/OneDrive/Desktop/Said/Art/ArtistData/genderClassification/rawdata/bordex.csv'
    
    # Import gender databases to Dataframes.
    # Keeping only the relevant data: name and gender.
    df1r = pd.read_csv(wgndDataLoc)
    df1 = df1r[['name', 'gender']]
    df1 = df1[df1.gender != '?']
    # df1 = df1[df1['name'].map(len) > 3]
    del df1r
    # 1: Female, 0: Male
    df1.gender.replace({'F': 1, 'M': 0}, inplace=True)

    # Unfortunately names cannot be identified based on pinyin.
    # We ommit the Chinese name-gender data due to this issue.
    # If we want to identify Chinese names we have to do a 1:1 search.
    '''
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
    '''

    df3r = pd.read_csv(UNKData, error_bad_lines=False)
    df3 = df3r[['Name', 'prob.gender']]
    df3 = df3.rename(columns={'prob.gender': 'gender', 'Name': 'name'})
    df3 = df3[df3.gender != 'Unknown']
    del df3r
    df3.gender.replace({'Female': 1, 'Male': 0}, inplace=True)
    # df3 = df3[df3['name'].map(len) > 3]

    df4 = pd.DataFrame()
    for file in os.listdir(SSADAta):
        if file.endswith(".txt"):
            dfx = pd.read_csv(SSADAta + '/' + file, delimiter=',', index_col=False, header=None, names=['name', 'gender'])
            df4 = df4.append(dfx, ignore_index=True)
    df4.gender.replace({'F': 1, 'M':0}, inplace=True)
    # df4 = df4[df4['name'].map(len) > 3]

    '''
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
    '''
    
    df6 = pd.read_csv(bordex, error_bad_lines=False)
    df6 = df6.rename(columns={'OrganisationCountry': 'country', 'IndividualName': 'fullname', 'IndividualGender': 'gender'})
    df6 = df6.drop(['IndividualID'], axis=1)
    df6 = df6[df6.country != "China"]
    df6 = df6[df6.country != "Hong Kong"]
    df6['name'] = df6.fullname.str.split(' ')
    df6['name'] = df6['name'].str[0]
    df6 = df6[['name', 'gender']]
    df6.gender.replace({'F': 1, 'M':0}, inplace=True)
    
    
    df = pd.concat([df1, df3, df4, df6])
    df['name'] = df['name'].str.lower()
    df['name'] = df['name'].str.strip()
    # Remove dupl icates where gender and name are the same
    df = df.drop_duplicates(['name', 'gender'])
    # Find names to which both genders are assigned.
    df['duplicated'] = df.duplicated(subset=['name'], keep=False)
    df = df[df['duplicated'] == False]
    return df


import string
def letter2vec(letter, letters=string.ascii_lowercase):
    vector = [0 if char != letter else 1 for char in letters]
    return vector


def extractFeatures(nameRaw):
    try:
        name = removeSpecialChars(nameRaw.lower().strip().replace(' ', ''))
    except:
        name = ''
    nameL = len(name)
    retVect = np.zeros((15,26))
    if nameL <= 15 & nameL != 0:
        for inx, letter in enumerate(name):
            try:
                onehot = letter2vec(letter)
                retVect[inx] = onehot
            except:
                pass
    return retVect



def revertFeatures(features):
    pass


if __name__ == '__main__':
    trainingDataCompiled = True
    if not trainingDataCompiled:
        df = importDataMultipleLoc()
        df.to_csv(r'C:\Users\Milan\OneDrive\Desktop\Said\GenderClassification\trainingData\training_data.csv', index=False)
    else:
        df = pd.read_csv(r'C:\Users\Milan\OneDrive\Desktop\Said\GenderClassification\trainingData\training_data.csv')
    #%%
    # Shuffle the data
    df = df.sample(frac=1, random_state=1)
    
    #for row in df.iterrows():
     #   name = row[1]['name']
      #  f = extractFeatures(name)
       # row[1]['features'] = f
    
    
    # NERUAL NETWORK CLASSIFICATION
    Xi = []
    yi = []
    for row in df.iterrows():
        name = row[1]['name']
        vec = extractFeatures(name)
        num_of_non_zeros = np.count_nonzero(vec)
        if num_of_non_zeros == 0:
            pass
        else:
            Xi.append(vec)
            yi.append(row[1]['gender'])
    # yic = [y if y == 1 else -1 for y in yi]
    
    xTR, xTE, yTR, yTE = train_test_split(Xi, yi, test_size=0.2, random_state=1)
    
    xTR = np.array(xTR)
    yTR = np.array(yTR)
    
    xTE = np.array(xTE)
    yTE = np.array(yTE)
    
    
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Input(shape=(xTR.shape[1],)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu, kernel_initializer='he_normal'))
    model.add(tf.keras.layers.Dense(60, activation=tf.nn.relu, kernel_initializer='he_normal'))
    model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu, kernel_initializer='he_normal'))
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu, kernel_initializer='he_normal'))
    model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu, kernel_initializer='he_normal'))
    model.add(tf.keras.layers.Dense(60, activation=tf.nn.relu, kernel_initializer='he_normal'))
    model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu, kernel_initializer='he_normal'))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    h = model.fit(xTR, yTR, epochs=5, batch_size=16, verbose=2, validation_data=(xTE, yTE), shuffle=True)
    
    # plt.plot(h.history['loss'])
    # plt.plot(h.history['val_loss'])
    
    plt.plot(h.history['binary_accuracy'])
    plt.plot(h.history['val_binary_accuracy'])
    
    #_, accuracy = model.evaluate(xTE, yTE, verbose=1)
    #print('Accuracy: %.2f' % (accuracy))
    
    
    # Predict results for Renee's databases
    fileloc = r'C:\Users\Milan\Downloads\gender'
    export_excel_loc = r'C:\Users\Milan\Downloads\gender\results'
    
    from tqdm import tqdm
    
    data = []
    #%%
    for file in os.listdir(fileloc):
        if file.endswith('.csv'):
            dfx = pd.read_csv(os.path.join(fileloc, file))

            if 'forename' in dfx.columns:
                for row in tqdm(dfx.iterrows(), total=dfx.shape[0]):
                    name = row[1]['forename']
                    # MATCH 1:1 IN DATABASE
                    try:
                        match = df[df['name'].isin([name.lower()])]
                        if not match.empty:
                            dfx.loc[row[0], 'exactMatch'] = 1
                            matchValue = match.values[0][1]
                            if matchValue == 1:
                                dfx.loc[row[0], 'exactMatchGender'] = 'Female'
                            elif matchValue == 0:
                                dfx.loc[row[0], 'exactMatchGender'] = 'Male'
                        else:
                            dfx.loc[row[0], 'exactMatch'] = 0
                            dfx.loc[row[0], 'exactMatchGender'] = None
                    except:
                        dfx.loc[row[0], 'exactMatch'] = 0
                        dfx.loc[row[0], 'exactMatchGender'] = None
                        

                    # NEURAL NETWORK PREDICTION
                    try:
                        vec = extractFeatures(name)
                    except:
                        vec = np.zeros((15,26))

                    num_of_non_zeros = np.count_nonzero(vec)
                    
                    if num_of_non_zeros == 0:
                        dfx.loc[row[0], 'NNvalidInput'] = 0
                        dfx.loc[row[0], 'NNvalue'] = None
                        dfx.loc[row[0], 'NNgender'] = None
                    else:
                        arrs = np.array([vec])
                        try:
                            prediction = float(model.predict(arrs))
                        except:
                            prediction = None
                    
                        dfx.loc[row[0], 'NNvalidInput'] = 1
                        dfx.loc[row[0], 'NNvalue'] = prediction
                        if prediction > 0.5:
                            dfx.loc[row[0], 'NNgender'] = 'Female'
                        else:
                            dfx.loc[row[0], 'NNgender'] = 'Male'
                            
            
            elif 'artist' in dfx.columns:
                for row in tqdm(dfx.iterrows(), total=dfx.shape[0]):
                    fullname = row[1]['artist']
                    try:
                        parts = fullname.split(' ')
                        name = parts[0]
                        
                         # MATCH 1:1 IN DATABASE
                        try:
                            match = df[df['name'].isin([name.lower()])]
                            if not match.empty:
                                dfx.loc[row[0], 'exactMatch'] = 1
                                matchValue = match.values[0][1]
                                if matchValue == 1:
                                    dfx.loc[row[0], 'exactMatchGender'] = 'Female'
                                elif matchValue == 0:
                                    dfx.loc[row[0], 'exactMatchGender'] = 'Male'
                            else:
                                dfx.loc[row[0], 'exactMatch'] = 0
                                dfx.loc[row[0], 'exactMatchGender'] = None
                        except:
                            dfx.loc[row[0], 'exactMatch'] = 0
                            dfx.loc[row[0], 'exactMatchGender'] = None
                    except:
                        dfx.loc[row[0], 'exactMatch'] = 0
                        dfx.loc[row[0], 'exactMatchGender'] = None
                            
                        
                    # NEURAL NETWORK PREDICTION
                    try:
                        vec = extractFeatures(name)
                    except:
                        vec = np.zeros((15,26))
                    num_of_non_zeros = np.count_nonzero(vec)
                    
                    if num_of_non_zeros == 0:
                        dfx.loc[row[0], 'NNvalidInput'] = 0
                        dfx.loc[row[0], 'NNvalue'] = None
                        dfx.loc[row[0], 'NNgender'] = None
                    else:
                        arrs = np.array([vec])
                        try:
                            prediction = float(model.predict(arrs))
                        except:
                            prediction = None
                            
                        dfx.loc[row[0], 'NNvalidInput'] = 1
                        dfx.loc[row[0], 'NNvalue'] = prediction
                        try:
                            female = prediction > 0.5
                            if female:
                                dfx.loc[row[0], 'NNgender'] = 'Female'
                            else:
                                dfx.loc[row[0], 'NNgender'] = 'Male'
                        except:
                            dfx.loc[row[0], 'NNgender'] = None
            
            elif 'firstname' in dfx.columns:
                for row in tqdm(dfx.iterrows(), total=dfx.shape[0]):
                    fullname = row[1]['firstname']
                    try:
                        fullname = fullname.encode('ascii', 'ignore')
                        fullname = fullname.decode()
                    
                        try:
                            parts = fullname.split(' ')
                            name = parts[1].lower()

                            
                            # MATCH 1:1 IN DATABASE
                            try:
                                match = df[df['name'].isin([name.lower()])]
                                if not match.empty:
                                    matchValue = match.values[0][1]
                                    if matchValue == 1:
                                        dfx.loc[row[0], 'Match Gender'] = 'F'
                                    elif matchValue == 0:
                                        dfx.loc[row[0], 'Match Gender'] = 'M'
                                else:
                                    dfx.loc[row[0], 'Match Gender'] = None
                            except:
                                dfx.loc[row[0], 'Match Gender'] = None
                        except:
                            dfx.loc[row[0], 'Match Gender'] = None
                                
                            
                        # NEURAL NETWORK PREDICTION
                        try:
                            vec = extractFeatures(name)
                        except:
                            vec = np.zeros((15,26))
                        num_of_non_zeros = np.count_nonzero(vec)
                        
                        if num_of_non_zeros == 0:
                            dfx.loc[row[0], 'NN Gender'] = None
                        else:
                            arrs = np.array([vec])
                            try:
                                prediction = float(model.predict(arrs))
                            except:
                                prediction = None
                            try:
                                female = prediction > 0.5
                                confidence = abs(prediction - 0.5) > 0.25
                                if confidence:
                                    if female:
                                        dfx.loc[row[0], 'NN Gender'] = 'F'
                                    else:
                                        dfx.loc[row[0], 'NN Gender'] = 'M'
                                else:
                                    dfx.loc[row[0], 'NN Gender'] = None
                            except:
                                dfx.loc[row[0], 'NN Gender'] = None
                    except:
                        dfx.loc[row[0], 'Match Gender'] = None
                        dfx.loc[row[0], 'NN Gender'] = None

                
        export_filename = file.split('.')[0] + '-done.xlsx'
        export_loc = os.path.join(export_excel_loc, export_filename)

        with pd.ExcelWriter(export_loc, mode='w') as ew:
            dfx.to_excel(ew, index=False, sheet_name='PhD data')
    