# CS5293Spring2020Project2
In this project COVID-19 research dataset is chosen from kaggle which is prepared by different research groups for research purposes, randoly 10% of all files will be selected randomly to summarize them with respect to clusters formed.
### Author :- Ram charan Reddy Kankanala
### Email :- Ramcharankankanala@gmail.com


#### Python indentation is not followed in below code samples.
### Assumptions made:
1) From every paper only paperID and text from body_text is selected.
2) 


### project2--randomfiles.py
This function takes 2 arguments(number, path list) to select required number of files randomly from all json files. This function returns a list of links. 
>  alljsonfiles(path list) = glob.glob('CORD-19-research-challenge/**/*.json', recursive=True) \
   def randomfiles(num, alljsonfiles): \
     sampling = random.choices(alljsonfiles, k=int((num * len(alljsonfiles) / 100))) \
     return sampling
     
### project2-- extraxtext.py
This function take path as argument to extract required data(as specified in assumptions). Json data is loaded and paperID , body_text(text) will be read into a list to pass to further process.This function returns two lists containg paper_id and body_text.
>   Read_data = json.load(i)\
    pap.append(Read_data["paper_id"])


    
### project2-- datadf.py
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
    
 
