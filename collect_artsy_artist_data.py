import requests
from bs4 import BeautifulSoup as bs

import pandas as pd
from multiprocessing.pool import ThreadPool

from time import sleep


EXPORTLOC = r'C:\Users\Milan\OneDrive\Desktop\SBS\Gender Classification\data\ArtsyArtists.xlsx'
EXPORTLOC2 = r'C:\Users\Milan\OneDrive\Desktop\SBS\Gender Classification\data\ArtsyArtistsNew.xlsx'
MAX_RETRIES = 5
URL_COLLECTED = True


def importData():
    df = pd.read_excel(EXPORTLOC)
    return df


def assembleURLS():
    beginning = 'https://www.artsy.net/artists/artists-starting-with-'
    urls = []
    from string import ascii_lowercase
    for letter in ascii_lowercase:
        urls.append(beginning + letter)

    return urls


def exportData(df: pd.DataFrame, exp):
    with pd.ExcelWriter(exp) as w:
        df.to_excel(w, sheet_name='Data', index=False)


def getArtistData(url: str):
    data = []
    headers = {
        'User-Agent': r'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36',
        'From': 'david.zhuang@outlook.com'}
    pageNumber = 1
    retryCount = 0
    while True:
        searchURL = url + '?page=' + str(pageNumber)
        print(searchURL)
        try:
            r = requests.get(searchURL, timeout=20, headers=headers)
            if r.status_code == 200:
                soup = bs(r.text, 'html.parser')
                elements = soup.findAll('a', {'class': 'highlight-link'})
                if len(elements) > 0:
                    for element in elements:
                        try:
                            data.append([element.text, 'https://www.artsy.net' + element.get('href')])
                        except:
                            print('issue')
                if len(elements) == 0:
                    print(url + ' ------- ' + str(pageNumber))
                    break
            else:
                break

        except ConnectionResetError as e:
            if retryCount < MAX_RETRIES:
                sleep(2)
                getArtistData(url)
            else:
                print(e)
                print(url + ' ------- ' + str(pageNumber))
                break
        except Exception as e:
            print(e)
            print(url + ' ------- ' + str(pageNumber))
            break

        pageNumber += 1

    return data


def countPronouns(text):
    m = 0
    f = 0
    male_pronouns = ['he', 'his', 'him', 'himself']
    female_pronouns = ['she', 'her', 'hers', 'herself']

    words = text.split(' ')
    for w in words:
        wClean = w.replace(',', '').replace('.', '').lower()
        if wClean in male_pronouns:
            m += 1
        if wClean in female_pronouns:
            f += 1

    return f, m


def determineConfidence(f: int, m: int):
    t = f + m
    # if the prediction is male p<0, if the prediction is female p>0
    # if the f=m, p=0 as the model cannot make any prediction on the gender
    if f > m:
        sign = 1
        p = sign * f / t
    elif f < m:
        sign = -1
        p = sign * m / t
    elif f == m:
        p = 0

    return p

retryCount = 0
def getArtistGender(dfrow):
    inx, item = dfrow
    url = dfrow[1][1]
    headers = {
        'User-Agent': r'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36',
        'From': 'david.zhuang@outlook.com'
        }
    try:
        r = requests.get(url, timeout=20, headers=headers)
        if r.status_code == 200:
            soup = bs(r.text, 'html.parser')
            body = soup.find('span', {'class': 'bjlDrS'})
            if body is not None:
                try:
                    text = body.text
                    item['artsyPredictionDummy'] = 1
                    item['Description'] = text
                    f, m = countPronouns(text)
                    p = determineConfidence(f, m)
                    item['artsyFemaleCounter'] = f
                    item['artsyMaleCounter'] = m
                    if p > 0:
                        item['artsyGenderPrediction'] = 'Female'
                    elif p < 0:
                        item['artsyGenderPrediction'] = 'Male'
                    elif p == 0:
                        item['artsyGenderPrediction'] = ''
                    item['artsyPrediction'] = p
                    print(p)
                except Exception as e:
                    print(e)
    except ConnectionResetError as e:
        sleep(2)
        getArtistGender(dfrow)
    except TimeoutError as e:
        sleep(2)
        getArtistGender(dfrow)
    except Exception as e:
        print(e)

    return item


if __name__ == "__main__":
    if URL_COLLECTED is False:
        urls = assembleURLS()
        pool = ThreadPool(26)
        artistResults = pool.map(getArtistData, urls)
        pool.close()

        df = pd.DataFrame(artistResults[0], columns=['Name', 'URL'])
        exportData(df, EXPORTLOC)
    else:
        df = importData()

    pool = ThreadPool(40)
    genderResults = pool.map(getArtistGender, df.iterrows())
    artsyResults = pd.DataFrame(genderResults)
    exportData(artsyResults, EXPORTLOC2)
