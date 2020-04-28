## Summarized Data

### This markdown contains summarized sentences for each cluster.

#### Steps in summarizing data.
**step1:-** Best "k" value will be obtained from knee-elbow curve. \
**step2:-** This "k" value will be passed to summarize data function. This function initially forms cluster based on "k" value given and labels for each document are appended to original dataframe. By grouping labels, "data_text" column is joined together. So finally we will be having only "k" number of rows in dataframe with each row representing each cluster in order. \
**step3:-** Dataframe containing clusters will be parsed to do sentense tokenization and then sentense normalization for each sentense separately.This done by looping through dataframe, so it is done for each cluster at a time. \
**step4:-** After getting normalized sentences data vectorized(countvectorization) and matrix is returned.\
**step5:-** Matrix returned from vectorization will be used along with its "Transpose" to get similarity matrix(matrix x matrix.T).\
**step6:-** This similarity matrix is used to get scores of each sentense using "Text Ranking" algorithm which is from "networkx" package.\
**step7:-** From this scores are sorted and top 8 scores are selected to get indices from our original sentences list.\
**step8:-** Finally, using above indices top 8 sentences of each cluster are written into separate file. Thus at the end of the execution we will be having summarized text of each cluster in each file. \

#### Assumption: - Sentences are tokenized using sent_tokenize from nltk, so sentences in each cluster are according to this tokenization which in some clusters may not be appropriate.
