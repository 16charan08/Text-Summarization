# CS5293Spring2020Project2
In this project COVID-19 research dataset is chosen from kaggle which is prepared by different research groups for research purposes, randoly 10% of all files will be selected randomly to summarize them with respect to clusters formed.
### Author :- Ram charan Reddy Kankanala
### Email :- Ramcharankankanala@gmail.com


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

Getting random file links.
> files = randomfiles(1, alljsonfiles).

Extracting required data.
> p = extraxtext(f) \
   paper.append(p[0]) \
   text.append(p[1])
   
Stroing in a dataframe.
> df_covid = pd.DataFrame() \
    df_covid['paper_id'] = paper \
    df_covid['body_text'] = text \
    df_covid.head() \
    df_covid["body_text"].replace('', np.nan, inplace=True) \
    df_covid = df_covid.dropna(subset=['body_text'])
    

## project2.py-- normalize_document(txt)
This function takes text/string as a input and will be normalized(tokenized)
##### steps: - 
All http links are removed from text.
> txt = re.sub(r'^https?:\/\/.*[\r\n]*', '', txt, flags=re.MULTILINE)

All numbers, punctuations are removed.
>  txt = re.sub(r'[^\w\s]', '', txt)
    txt = re.sub(" \d+", " ", txt)
    
Text is converted into lower cases.
> txt = txt.lower()

Words are tokenized. 
> tokens = nltk.word_tokenize(txt)

Along with english stop words some custom stop words are selected from observing some outputs. 
 > custom_stop_words = ['figure', 'fig', 'fig.', 'www', 'https', 'et', 'al', 'medrxiv', 'biorxiv', 'mol', 'cl', 'moi']
    stop_words.append(custom_stop_words)
    
Lemmatizing of words. 
>  wordnet_lem = [WordNetLemmatizer().lemmatize(w) for w in clean_tokens]
 
 Every step is done one after other finally will be joining while returning.
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

This function finds a best k - value for the choosen data and retruns corresponded vectorized matrix and k -value.
Vectorization.
>tfidf_vectorizer = TfidfVectorizer(ngram_range=(3, 4), max_df=0.9, min_df=0.005, sublinear_tf=True)\
    tfidf_matrix = tfidf_vectorizer.fit_transform(datadf)
    
K-means best value.

>km = KMeans(max_iter=10000, n_init=50, random_state=42, n_jobs= -1) \
    visualizer = KElbowVisualizer(km, k=(2, 15)) \
    visualizer.fit(tfidf_matrix) \
    result = [tfidf_matrix,visualizer.elbow_value_]
    
 Thus instead of seeing elbow graph and deciding best value of k, this attribute from yellowbrick makes things easy to automate the whole process.
