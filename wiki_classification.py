import pandas as pd
import requests
import wikipedia
from multiprocessing.pool import ThreadPool

DATALOC = r'C:\Users\Milán\OneDrive\Desktop\SBS\Artworks\Artists\sampleClassifiedArtistData.xlsx'
EXPORTLOC = r'C:\Users\Milán\OneDrive\Desktop\SBS\Artworks\Artists\sampleWikiClassificationData.xlsx'


def determineGender(p:float):
    if p > 0:
        gender = 'Female'
    if p < 0:
        gender = 'Male'
    if p == 0:
        gender = ''
    return gender


def determineConfidence(f:int, m:int):
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


def wikiSearch(name: str):
    try:
        page = wikipedia.page(name)
        status = True
    # Handle error if the search query returns multiple results
    except wikipedia.DisambiguationError:
        try:
            # Adding (artist) to the search might help narrow the results.
            guideName = name + '(artist)'
            page = wikipedia.page(guideName)
            status = True
        except:
            status = False
    except:
        status = False

    if status:
        url = page.url
        content = page.content
        f, m = countPronouns(content)
        p = determineConfidence(f, m)
        gender = determineGender(p)
        return url, f, m, p, gender

    if not status:
        return False


def getWikiData(row):
    inx, item = row
    result = wikiSearch(item['artist_Name'])
    if result:
        url, female, male, prediction, gender = result
        item['wikiURL'] = url
        item['wikiFemaleCounter'] = female
        item['wikiMaleCounter'] = male
        item['wikiPrediction'] = prediction
        item['wikiGenderPrediction'] = gender
        print(inx)

    return item


def importData():
    df = pd.read_excel(DATALOC, sheet_name='Sheet1', header=0, index_col=0)
    return df


def exportData(df:pd.DataFrame):
    with pd.ExcelWriter(EXPORTLOC) as w:
        df.to_excel(w, sheet_name='Data2', index=False)



if __name__ == '__main__':
    df = importData()

    pool = ThreadPool(40)
    wikiresults = pool.map(getWikiData, df.iterrows())
    wikiexport = pd.DataFrame(wikiresults)
    exportData(wikiexport)
    pool.close()
