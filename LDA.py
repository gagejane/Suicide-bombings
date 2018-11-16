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
    # data_text['index'] = data_text.index
    documents = data_text
    # print(len(documents))
    # print(documents[:100])
    my_stopwords = {'motive', 'attack', 'Unknown', 'unknown', 'however', 'sources', 'specific', 'stated', 'statement', 'States', 'state', 'target', 'speculate', 'incident', 'targeted', 'targeting', 'speculated', 'suicide', 'bomb', 'bombing', 'bomber', 'responsibility', 'claim', 'claimed', 'noted', 'State', 'carried', 'majority', 'minority'}
#consider making a single set

    stemmer = SnowballStemmer('english')
    processed_docs = documents['motive'].map(preprocess)
    texts = [[''.join(item) for item in document] for document in processed_docs]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    numpy.random.seed(1) # setting random seed to get the same results each time.
    model = ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3)
    topics = model.show_topics()


    doc_top_prob = [model.get_document_topics(corpus[i]) for i in range(len(corpus))]
    three_scores = []
    for i in doc_top_prob:
        if len(i) == 3:
            three_scores.append(i)
        else:
            pass

    '''THERE ARE SOMETIMES MISSING DATA POINTs IN PART OF A TUPLE IN THE UPSAMPLE. FIGURE THIS OUT'''
    # doc_top_prob[21393] = [(0, 0.99014634),(1,0.0)]
    # for i in doc_top_prob:
    #     if len(i) <
    #
    #
    topic1_tup = []
    topic2_tup = []
    topic3_tup = []
    # topic4_tup = []
    # topic5_tup = []
    # topic6_tup = []
    for i, j, k in three_scores:
        topic1_tup.append(i)
        topic2_tup.append(j)
        topic3_tup.append(k)
        # topic4_tup.append(l)
        # topic5_tup.append(m)
        # topic6_tup.append(n)

    topic1 = []
    topic2 = []
    topic3 = []
    # topic4 = []
    # topic5 = []
    # topic6 = []
    for i in topic1_tup:
        topic1.append(i[1])
    for i in topic2_tup:
        topic2.append(i[1])
    for i in topic3_tup:
        topic3.append(i[1])
    # for i in topic4_tup:
    #     topic4.append(i[1])
    # for i in topic5_tup:
    #     topic5.append(i[1])
    # for i in topic6_tup:
    #     topic6.append(i[1])

    # processed_docs = pd.Series.values
    # df_topic1 = pd.DataFrame(np.array(topic1).reshape(1813,1), columns = list('topic1'))
    # df_topic2 = pd.DataFrame(np.array(topic2).reshape(1813,1), columns = list('topic2'))
    df_topic1 = pd.DataFrame(i for i in topic1)
    df_topic2 = pd.DataFrame(i for i in topic2)
    df_topic3 = pd.DataFrame(i for i in topic3)
    # df_topic4 = pd.DataFrame(i for i in topic4)
    # df_topic5 = pd.DataFrame(i for i in topic5)
    # df_topic6 = pd.DataFrame(i for i in topic6)

    df_processed_docs = pd.DataFrame(data=processed_docs)
    # df_processed_docs['merging_index'] = df_processed_docs.index
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



#
#     # final_LDA = final_LDA.append(df_topic2)
    # final_LDA=processed_docs.copy()
    # final_LDA['topic1'] = df_topic1
    # final_LDA['topic2'] = df_topic2


        #
#     # print('get_document_topics', model.get_document_topics(corpus[0]))
#
#     # common_corpus = [dictionary.doc2bow(text) for text in texts]
#     # bow = model.id2word.doc2bow(texts) # convert to bag of words format first
#     # doc_topics, word_topics, phi_values = model.get_document_topics(corpus, per_word_topics=True)
#
#     # print(word_topics)
#
#
#     # df = pd.read_csv('data/globalterrorismdb_0718dist.csv', low_memory=False)
#     # # # #
#     # # # # '''full text dataset with only summary variable'''
#     # # # # # df_summary=df[['summary']]
#     # df_summary_vars=df[['summary', 'motive', 'region', 'gname', 'country']]
#     # df_summary_vars.to_csv('data/df_summary_vars.csv', index=False)
# #     df = pd.read_csv('data/df_summary_vars.csv', low_memory=False)
# #     # # #
# #     # # # '''summary dataset with only n = 100'''
# #     # # # # df_100 = df.iloc[:100]
# #     # # # # df_100.to_csv('data/df_summary_100.csv', index=False)
# #     # # # # df = pd.read_csv('data/df_summary_100.csv', low_memory=False)
# #     # # #
# #     # # # '''summary + other vars with n = 50000'''
# #     # df_summary_vars=df[['summary', 'motive', 'region', 'gname', 'country']]
# #     # df_summary_vars_50000 = df_summary_vars.iloc[:50000]
# #     # df_summary_vars_50000.to_csv('data/df_summary_vars_50000.csv', index=False)
# #     # df = pd.read_csv('data/df_summary_vars_50000.csv', low_memory=False)
# #     # df=df.values
# #
# #
# #     df.dropna(inplace = True)
# #     pd.set_option('display.max_colwidth', -1)
# #
# #     '''name of target entity: corp1
# #     motive: motive'''
# #
# #     documents = df['motive'].values
# #     # num_features = 1000
# #     # tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
# #     # tf = tf_vectorizer.fit_transform(documents)
# #     # tf_feature_names = tf_vectorizer.get_feature_names()
# #     #
# #     # num_topics = 10
# #     # lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online',random_state=0)
# #     # lda.fit(tf)
# #     # # print(lda.components_)
# #     #
# #     # def display_topics(model, feature_names, num_top_words):
# #     #     for topic_idx, topic in enumerate(model.components_):
# #     #         print("Topic %d:" % (topic_idx))
# #     #         print(" ".join([feature_names[i]
# #     #                         for i in topic.argsort()[:-num_top_words - 1:-1]]))
# #     #
# #     # num_top_words = 10
# #     # display_topics(lda, tf_feature_names, num_top_words)
# #     #
# #     # print("Model perplexity: {}".format(lda.perplexity(tf)))
# #     # print("Model log likelihood: {}".format(lda.score(tf)))
# #
# # import pandas as pd
# # import numpy as np
# #
# # #gensim
# # #pip install --upgrade gensim
# # import gensim
# # from gensim.utils import simple_preprocess
# # from gensim.parsing.preprocessing import STOPWORDS
# # import gensim.corpora as corpora
# # from gensim.models import CoherenceModel
# #
# # #nltk
# # from nltk.stem import WordNetLemmatizer, SnowballStemmer
# # from nltk.stem.porter import *
# # import nltk
# # nltk.download('wordnet')
# #
# # # Plotting tools
# # #pip install pyldavis
# # import pyLDAvis
# # import pyLDAvis.gensim  # don't skip this
# # import matplotlib.pyplot as plt
# # # %matplotlib inline
# #
# # import warnings
# # warnings.filterwarnings("ignore",category=DeprecationWarning)
# #
# #
# #     #
# #     # dict_region = {1:'North America', 2:'Central American and Caribbean', 3:'South America', 4:'East Asian', 5:'Southeast Asia', 6:'South Asia', 7:'Central Asia', 8:'Western Europe', 9:'Eastern Europe', 10:'Middle East and North Africa', 11:'Sub-Saharan Africa', 12:'Australasia & Oceania'}
# #     # df['region_names'] = df['region'].replace(dict_region)
# #     # df_region_names = df['region_names']
# #     # #
# #     # # dict_suicide = {1: 'suicide bombing', 0: 'not suicide bombing'}
# #     # # df['suicide_text'] = df['suicide'].replace(dict_suicide)
# #     #
# #     #
# #     # # assert len(df) == 100
# #     # # assert len(df['gname'].unique()) == len(df['gname'].unique())
# #     #
# #     # vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
# #     # # vectorizer = TfidfVectorizer(stop_words='english')
# #     # X = vectorizer.fit_transform(df['motive'])
# #     # features = vectorizer.get_feature_names()
# #     #
#     # distxy = squareform(pdist(X.todense(), metric='cosine'))
#     # # 4. Pass this matrix into scipy's linkage function to compute our
#     # # hierarchical clusters.
#     # link = linkage(distxy, method='complete')
#     #
#     # # 5. Using scipy's dendrogram function plot the linkages as
#     # # a hierachical tree.
#     # # dendro = dendrogram(link, color_threshold=1.5, leaf_font_size=9)
#     # # plt.show()
