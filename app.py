from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from numpy import nan as Nan
import random

desc = input("LAY ME DOWN SOME WORDS")
abv = input("How Strong?")

df = pd.read_csv("cleanedfinalbeerdata.csv")

df2 = pd.DataFrame([[33050, "UserBeer", Nan, desc, abv, Nan, Nan, Nan, Nan, Nan, Nan, Nan, Nan, Nan, Nan, Nan, Nan, Nan, Nan, abv]],
                   columns=list(df.columns.values))
df3 = pd.concat([df, df2])

df3.loc[df3['id'] == 'UserBeer']

finaldf1 = df3[(df3.description.notnull())]

documents = np.ndarray.tolist(finaldf1['description'].values)

my_stop_words = text.ENGLISH_STOP_WORDS.union(["beer"])
vectorizer = TfidfVectorizer(stop_words=my_stop_words, decode_error='replace', encoding='utf-8')

X = vectorizer.fit_transform(documents)

kmeans = KMeans(n_clusters=15)
srmdata = np.array(list(zip(finaldf1['newabv'].values)), X)
kmeans.fit(srmdata)
predicted_clusters = kmeans.predict(srmdata)

finaldf1['predicted group'] = predicted_clusters

finaldf1[finaldf1['id'].str.contains('UserBeer')]

chosenbeergroup = finaldf1[finaldf1['id'].str.contains('UserBeer')]['predicted group'].values[0]


filterdf = finaldf1[finaldf1['predicted group']==chosenbeergroup]
filterdf.head()

selectarr = []
numberingroup = filterdf['id'].count()

for x in range(0, 6):
    selectarr.append (random.randint(0, int(numberingroup)))

finaluserdf = filterdf.iloc[selectarr]

print(finaluserdf["description"])
