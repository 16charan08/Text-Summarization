# CS5293Spring2020Project2
In this project COVID-19 research dataset is chosen from kaggle which is prepared by different research groups for research purposes, randoly 10% of all files will be selected randomly to summarize them with respect to clusters formed.
### Author :- Ram charan Reddy Kankanala
### Email :- Ramcharankankanala@gmail.com

### Packages installed/used 
pip install pipenv \
pipenv install numpy &nbsp; \
pipenv install nltk &nbsp; \
pipenv install pandas \
pipenv install yellowbrick \
pipenv install sklearn \
pipenv install networkx\
import networkx \
import numpy as np  \
import os \
import pandas as pd  \
import glob
import json\
import gc\
import random \
import re \
import nltk \
import numpy as np \
nltk.download('punkt') \
nltk.download('stopwords') \
nltk.download('wordnet') \
stop_words = nltk.corpus.stopwords.words('english') \
from nltk.stem.wordnet import WordNetLemmatizer \
from random import Random \
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer \
from yellowbrick.cluster import KElbowVisualizer \
from sklearn.cluster import KMeans \
from yellowbrick.cluster.elbow import kelbow_visualizer \
import sklearn


#### Python indentation is not followed in below code samples.


## project2.py--randomfiles(num, alljsonfiles)
This function takes 2 arguments(number, path list) to select required number of files randomly from all json files. This function returns a list of links. 
>  alljsonfiles(path list) = glob.glob('CORD-19-research-challenge/**/*.json', recursive=True) \
   def randomfiles(num, alljsonfiles): \
     sampling = random.choices(alljsonfiles, k=int((num * len(alljsonfiles) / 100))) \
     return sampling
     
## project2.py-- extraxtext(path)
##### Assumptions made in this step:
1) From every paper only paperID and text from body_text is selected.\
This function take path as argument to extract required data(as specified in assumptions). Json data is loaded and paperID , body_text(text) will be read into a list to pass to further process.This function returns two lists containg paper_id and body_text.
>   Read_data = json.load(i)\
    pap.append(Read_data["paper_id"])

    
## project2.py-- datadf()
This function takes no arguments but calles above two functions to extract required data randomly to store it in dataframe. In this function empty values of body_text will be identified to remove. Finally This function returns a dataframe with paper_id and body_text as features.

*step1:-* Getting random file links.
> files = randomfiles(1, alljsonfiles).

*step2:-* Extracting required data.
> p = extraxtext(f) \
   paper.append(p[0]) \
   text.append(p[1])
   
*step3:-* Stroing in a dataframe.
> df_covid = pd.DataFrame() \
    df_covid['paper_id'] = paper \
    df_covid['body_text'] = text \
    df_covid.head() \
    df_covid["body_text"].replace('', np.nan, inplace=True) \
    df_covid = df_covid.dropna(subset=['body_text'])
    

## project2.py-- normalize_document(txt)
This function takes text/string as a input and will be normalized(tokenized)

*step1:-* All http links are removed from text.
> txt = re.sub(r'^https?:\/\/.*[\r\n]*', '', txt, flags=re.MULTILINE)

*step1:-* All numbers, punctuations are removed.
>  txt = re.sub(r'[^\w\s]', '', txt)
    txt = re.sub(" \d+", " ", txt)
    
*step3:-* Text is converted into lower cases.
> txt = txt.lower()

*step4:-* Words are tokenized. 
> tokens = nltk.word_tokenize(txt)

*step5:-* Along with english stop words some custom stop words are selected from observing some outputs. 
 > custom_stop_words = ['figure', 'fig', 'fig.', 'www', 'https', 'et', 'al', 'medrxiv', 'biorxiv', 'mol', 'cl', 'moi']
    stop_words.append(custom_stop_words)
    
*step6:-* Lemmatizing of words. 
>  wordnet_lem = [WordNetLemmatizer().lemmatize(w) for w in clean_tokens]
 
*step7:-* Every step is done one after other finally will be joining while returning.
 > return ' '.join(wordnet_lem)
 
 
## project2.py-- noramizedf()
This function simply calls above function to normalize every body_text row from our original dataframe and returns noramized dataframe.

Normalizing each row.
>    for t in data['body_text']: \
     normalizedlist.append(normalize_document(t)) \
    df_normalized = pd.DataFrame() \
    df_normalized["paper_id"] = data['paper_id'] \
    df_normalized["body_text"] = normalizedlist

## project2.py-- clusterfind()
##### Assumptions made in this step:
1) From yellowbrick package we can visualize keblow curve using (KElbowVisualizer) which as an attribute "elbow_value_" this value is returned and passed onto next summarization process.

*step1:-* This function finds a best k - value for the choosen data and retruns corresponded vectorized matrix and k -value.
Vectorization.
>tfidf_vectorizer = TfidfVectorizer(ngram_range=(3, 4), max_df=0.9, min_df=0.005, sublinear_tf=True)\
    tfidf_matrix = tfidf_vectorizer.fit_transform(datadf)
    
*step2:-* K-means best value.
>km = KMeans(max_iter=10000, n_init=50, random_state=42, n_jobs= -1) \
    visualizer = KElbowVisualizer(km, k=(2, 15)) \
    visualizer.fit(tfidf_matrix) \
    result = [tfidf_matrix,visualizer.elbow_value_]
    
 Thus instead of seeing elbow graph and deciding best value of k, this attribute from yellowbrick makes things easy to automate the whole process.

## project2.py-- summarize(k)
##### Assumptions made in this step:
1) Summarization is done clustering wise .i.e for each cluster top 8 sentences are selected to write in a document. So finally we will be having k * documents number of documents, where k is number of clusters obtanied from previous function.

This function takes only one argument of best k value obtained from previous function.

*step1:-* Clustering based on k value to find labels and append to original dataframe.
>km1 = KMeans(n_clusters=k, max_iter=10000, n_init=50, random_state=42) \
    ck = km1.fit(matrix) \ # matrix from previous function
    data["labels"] = km1.labels_\
    df = pd.DataFrame() \
    df = data \
    df = df.sort_values(by='labels')\
    df["body_text"] = data.groupby('labels')['body_text'].transform("".join) \
    dfset1 = df.body_text.unique()
 This step finally gives k number of documents which are joined together based on lables from k-means clustering in order.


*step2:-* Sentense tokenization and each sentense normalization.
> sentnorm = [] \
    sentences = nltk.sent_tokenize(dfset1[et]) \
        for every in sentences: \
            sentnorm.append(normalize_document(every)) 
            
  In this step, first, sentences are broken and on each sentense our normalized function is applied to store them in sentenwise in another list. So finally after this step we will be having normalized sentenses in a list.
  
*step3:-* Vectorization on sentenses to get similarity matrix.
> cvs = CountVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.8)\
        cv_matrixs = cvs.fit_transform(sentnorm)\
        similarity_matrix = (cv_matrixs * cv_matrixs.T)
This step vectorizes normalized sentenses list

*step4:-*  Next is getting similarity matrix and using Text Ranking getting top sentense indices.
> similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix) \
        scores = networkx.pagerank(similarity_graph)\ 
        ranked_sentences = sorted(((score, index)\
                                   for index, score \
                                   in scores.items()), reverse=True) \
        top_sentence_indices = [ranked_sentences[index][1] for index in range(8)] \
        top_sentence_indices.sort() \
 
 This step gets us top ranked indices based on the score.
 
 *step5:-* Writing top sentences of each cluster into each file.
 > summarizedata = [] \
        for index in top_sentence_indices: \
            if os.path.exists("cluster" + str(et)): \
                append_write = 'a'  # append if already exists \
            else:\
                append_write = 'w' \
            filessum = open("cluster" + str(et), append_write) \
            filessum.write(sentences[index] + '\n') \
            filessum.close()

This step takes indices from previous step and finds original sentences from list of sentences that we tokenized. Thus finally we will be having top sentences per cluster.


## Execution

- To get project folder:- 
> Run install pipenv

> git clone 

- For results:- 
run below command in command line.
pipenv run python project2.py


