import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import gensim.corpora as corpora
from gensim.models import CoherenceModel

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.corpora import Dictionary
from gensim.models import ldamodel
import numpy
from sklearn.utils import resample

#nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk



def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in my_stopwords and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

if __name__ == '__main__':


    '''RUN THIS CODE FOUR TIMES: downsample train, downsample test upsample train, upsample test'''

    # df = pd.read_csv('data/df_suicide_downsampled_Train.csv', low_memory=False)
    # df = pd.read_csv('data/df_suicide_downsampled_Test.csv', low_memory=False)

    # df = pd.read_csv('data/df_suicide_upsampled_Train.csv', low_memory=False)
    df = pd.read_csv('data/df_suicide_upsampled_Test.csv', low_memory=False)

    df.reset_index(inplace=True)

    df['motive'].replace(to_replace=['Unknown', 'The specific motive for the attack is unknown.'],value=np.NaN, inplace=True)
    data = df[['motive']]
    data.dropna(inplace = True)
    data_text = data[['motive']]
    documents = data_text
    my_stopwords = {'motive', 'attack', 'Unknown', 'unknown', 'however', 'sources', 'specific', 'stated', 'statement', 'States', 'state', 'target', 'speculate', 'incident', 'targeted', 'targeting', 'speculated', 'suicide', 'bomb', 'bombing', 'bomber', 'responsibility', 'claim', 'claimed', 'noted', 'State', 'carried', 'majority', 'minority'}

    stemmer = SnowballStemmer('english')
    processed_docs = documents['motive'].map(preprocess)
    texts = [[''.join(item) for item in document] for document in processed_docs]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    numpy.random.seed(1) # setting random seed to get the same results each time.
    model = ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3)
    topics = model.show_topics()


    doc_top_prob = [model.get_document_topics(corpus[i]) for i in range(len(corpus))]

    '''THERE ARE SOMETIMES MISSING DATA POINTs IN PART OF A TUPLE IN THE UPSAMPLE. THIS REMOVES THOSE CASES'''

    three_scores = []
    for i in doc_top_prob:
        if len(i) == 3:
            three_scores.append(i)
        else:
            pass

    topic1_tup = []
    topic2_tup = []
    topic3_tup = []
    for i, j, k in three_scores:
        topic1_tup.append(i)
        topic2_tup.append(j)
        topic3_tup.append(k)

    topic1 = []
    topic2 = []
    topic3 = []
    for i in topic1_tup:
        topic1.append(i[1])
    for i in topic2_tup:
        topic2.append(i[1])
    for i in topic3_tup:
        topic3.append(i[1])

    df_topic1 = pd.DataFrame(i for i in topic1)
    df_topic2 = pd.DataFrame(i for i in topic2)
    df_topic3 = pd.DataFrame(i for i in topic3)

    df_processed_docs = pd.DataFrame(data=processed_docs)
    df_processed_docs.reset_index(inplace=True)
    final = pd.concat([df_processed_docs,df_topic1,df_topic2,df_topic3], axis=1)
    final.columns = ['index', 'motive2', 'topic1', 'topic2', 'topic3']
    LDA_merged = df.merge(final, how='right', on='index')

    '''make ready for Decision Tree'''
    LDA_merged = LDA_merged.drop(['motive', 'motive2'], axis=1)
    # LDA_merged.to_csv('data/downsample_LDA_Train_forDT.csv', index=False)
    # LDA_merged.to_csv('data/downsample_LDA_Test_forDT.csv', index=False)
    # LDA_merged.to_csv('data/upsample_LDA_Train_forDT.csv', index=False)
    LDA_merged.to_csv('data/upsample_LDA_Test_forDT.csv', index=False)
