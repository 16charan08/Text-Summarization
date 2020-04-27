import string
import unicodedata

import networkx
import numpy as np  # linear algebra
import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import gc
import random
import re
import nltk
import numpy as np
stop_words = nltk.corpus.stopwords.words('english')
from nltk.stem.wordnet import WordNetLemmatizer
from random import Random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer
import sklearn

# seed = 6000
# rand = Random(seed)
alljsonfiles = glob.glob('CORD-19-research-challenge/**/*.json', recursive=True)


def randomfiles(num, alljsonfiles):
    sampling = random.choices(alljsonfiles, k=int((num * len(alljsonfiles) / 100)))
    return sampling


def extraxtext(path):
    pap = []
    bodytext = []
    with open(path) as i:
        Read_data = json.load(i)
        pap.append(Read_data["paper_id"])
    for j in Read_data["body_text"]:
        bodytext.append(j["text"])
    bodytext = '\n'.join(bodytext)

    return pap, bodytext


def datadf():
    paper = []
    text = []
    files = randomfiles(10, alljsonfiles)
    # print(len(files))
    res_list = []
    for i in range(0, len(alljsonfiles)):
        if alljsonfiles[i] in files:
            res_list.append(i)
    # print(len(res_list))
    for f in files:
        p = extraxtext(f)
        paper.append(p[0])
        text.append(p[1])
    df_covid = pd.DataFrame()
    df_covid['paper_id'] = paper
    df_covid['body_text'] = text
    df_covid.head()
    df_covid["body_text"].replace('', np.nan, inplace=True)
    df_covid = df_covid.dropna(subset=['body_text'])
    result = [df_covid, res_list]
    return result


data = datadf()[0]  # original data frame creation
indices = datadf()[0]  # Data selected indices from alljson files


# datadf().to_csv("out.csv")
# from lecture


def normalize_document(txt):
    txt = re.sub(r'^https?:\/\/.*[\r\n]*', '', txt, flags=re.MULTILINE)
    txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt, re.I | re.A)
    txt = re.sub(r'[^\w\s]', '', txt)
    txt = re.sub(" \d+", " ", txt)
    txt = txt.lower()
    txt = txt.strip()
    tokens = nltk.word_tokenize(txt)
    custom_stop_words = ['figure', 'fig', 'fig.', 'www', 'https', 'et', 'al', 'medrxiv', 'biorxiv', 'mol', 'cl', 'moi']
    stop_words.append(custom_stop_words)
    clean_tokens = [t for t in tokens if t not in stop_words]
    wordnet_lem = [WordNetLemmatizer().lemmatize(w) for w in clean_tokens]
    return ' '.join(wordnet_lem)


def noramizedf():
    normalizedlist = []
    for t in data['body_text']:
        # print(t)
        normalizedlist.append(normalize_document(t))
    df_normalized = pd.DataFrame()
    df_normalized["paper_id"] = data['paper_id']
    df_normalized["body_text"] = normalizedlist
    return df_normalized


# print(df_normalized.head(5))
# print(noramizedf()['body_text'][0])


def clusterfind():
    datadf = noramizedf()['body_text']
    cv = CountVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.8)
    cv_matrix = cv.fit_transform(datadf)
    print(cv_matrix.shape)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(3, 4), max_df=0.9, min_df=0.005, sublinear_tf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(datadf)
    km = KMeans(max_iter=10000, n_init=50, random_state=42, n_jobs= -1)
    visualizer = KElbowVisualizer(km, k=(2, 15))
    visualizer.fit(tfidf_matrix)        # Fit the data to the visualizer
    result = [tfidf_matrix,visualizer.elbow_value_]
    return result


matrix = clusterfind()[0]
k = clusterfind()[1]
print("cluster formed will be", k)


def summarize(k):
    km1 = KMeans(n_clusters=k, max_iter=10000, n_init=50, random_state=42)
    ck = km1.fit(matrix)
    data["labels"] = km1.labels_
    df = pd.DataFrame()
    df = data
    df = df.sort_values(by='labels')
    df["body_text"] = data.groupby('labels')['body_text'].transform("".join)
    dfset1 = df.body_text.unique()
    for et in range(0, len(dfset1)):
        count = 0
        sentnorm = []
        # sentences = []
        sentences = nltk.sent_tokenize(dfset1[et])
        # print(len(sentences))
        for every in sentences:
            sentnorm.append(normalize_document(every))
        # print(sentnorm)
        cvs = CountVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.8)
        cv_matrixs = cvs.fit_transform(sentnorm)
        similarity_matrix = (cv_matrixs * cv_matrixs.T)
        similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
        scores = networkx.pagerank(similarity_graph)
        ranked_sentences = sorted(((score, index)
                                   for index, score
                                   in scores.items()), reverse=True)
        top_sentence_indices = [ranked_sentences[index][1] for index in range(3)]
        top_sentence_indices.sort()
        print(top_sentence_indices)

        summarizedata = []
        for index in top_sentence_indices:
            if os.path.exists("cluster" + str(et)):
                append_write = 'a'  # append if already exists
            else:
                append_write = 'w'
            filessum = open("cluster" + str(et), append_write)
            filessum.write(sentences[index] + '\n')
            filessum.close()

    return summarizedata  # print(sentences[index])


summarize(k)

