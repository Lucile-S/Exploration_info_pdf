
# topic_modeling.py 
import numpy as np
import pandas as pd
import os 
import re
import glob
from collections import defaultdict
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


#NLP packages
import string 
import stop_words

## NLTK
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords 

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from nltk.stem import PorterStemmer

## Spacy
import spacy
import scispacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS  as stop_words_spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex

# Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import joblib # to save model

# word cloud
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Gensim
import gensim, spacy, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import TfidfModel
from gensim import corpora, models, similarities 
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim


#pip install scispacy
# pip install <Model URL> :  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_md-0.2.5.tar.gz
# For the parser, we will use en_core_sci_lg. This is a model for processing biomedical, scientific or clinical text.
import en_core_sci_sm
nlp = en_core_sci_sm.load(disable=["tagger", "ner"])


# Initialize the stopwords
stop_words_ =stop_words.get_stop_words('fr')
stop_words_nltk = stopwords.words('english')
custom_stop_words = []
stopwords= list( 
    set( 
        list(stop_words_nltk)  + stop_words_ + list(stop_words_spacy) + custom_stop_words
    )
)

# Initialize ponctuation
punctuations = string.punctuation +  "«" + "»"+ "’" + '—'

def find_AAV_terms(text):
    """
    returns a dictionnay 
    """
    #regex_aav = r"AAV(\d*\/|\d*-)*\w*(\([\w, -]*\))*(-\w+)*"
    AAV_terms = defaultdict(int)
    regex_AAV =r'''(AAV(\d*\/|\d*-)*\w*(\([\w, -]*\))*(-\w+)*)'''
    #ref_reg = r"^\d+( *\**\.)* ([A-Z]\S* ([A-Z]\S* )*[A-Z]\w*(, )*)+"
    for match in re.findall(regex_AAV, text):
        if len(match[0]) < 45:
            AAV_terms[match[0]] += 1
    #AAV_terms.append({k: v for k,v in d.items()})
    return AAV_terms 


def tokenize_anf_lemmatize(text,  allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    # regex specific to AAV variant 
    text =re.sub('-|\(|\)',' ', text)
    # tokens + lower case
    tokens = [token.text.lower().strip() for token in nlp(text)]
    ## remove number
    # tokens = [word for word in tokens if word.isalpha()]
    # remove stop words and punctuations and keep words >= 1 letter
    tokens = [ token for token in tokens if token not in stopwords and token not in punctuations and len(token) > 1]
    #print(tokens)
    # lemmatization
    tokens=[nlp(token)[0].lemma_  if nlp(token)[0].lemma_ != "-PRON-" else token for token in tokens]
    #print(tokens)
    # join the tokens
    tokenized_text = " ".join([token for token in tokens])
    return tokenized_text 



def K_means_optimal_K(X):
    K = range(2, 20)
    summary = pd.DataFrame(columns=['k','silhouette_score','distortion'])
    Sum_of_squared_distances =[]
    Silhouette_scores =[]
    K_list=[]
    for k in K:
        k_means = KMeans(n_clusters=k, random_state=0).fit(X)
        k_means.fit(X)
        Sum_of_squared_distance = k_means.inertia_
        labels= k_means.predict(X)
        Silhouette_score = silhouette_score(X, labels, metric='euclidean')
        print("For n_clusters = {}, silhouette score is {}".format(k,Silhouette_score))
        print("For n_clusters = {}, sum of squared distances is {}".format(k,Sum_of_squared_distance))
        K_list.append(k)
        Sum_of_squared_distances.append(Sum_of_squared_distance)
        Silhouette_scores.append(Silhouette_score)
    summary['k'] = K_list
    summary['silhouette_score']= Silhouette_scores
    summary['distortion']= Sum_of_squared_distances
    optimal_k = summary.iloc[summary['silhouette_score'].idxmax()]['k']
    print(f'Optimal k is {optimal_k}')
    plt.subplot(1,2,1)
    plt.plot(K, summary['distortion'], 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.savefig(model_dir + "Elbow_Method_For_Optimal_k.png")
    plt.show()
    plt.subplot(1,2,2)
    plt.plot(K, summary['silhouette_score'], 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette Method For Optimal k')
    plt.savefig(model_dir + "Silhouette_Method_For_Optimal_k.png")
    plt.show()
    return int(optimal_k)
    
def get_top_keywords(tfidf_matrix, clusters, vocabulary, n_terms):
  tfidf_df = pd.DataFrame(X.todense())
  tfidf_df['Cluster']=clusters
  grouped = tfidf_df.groupby('Cluster').mean()
  for index,row in grouped.iterrows():
    print(f'\nCluster {index}')
    print(', '.join([vocabulary[t] for t in np.argsort(row)[-n_terms:]]))

    wordcloud = WordCloud(max_words=50).generate(' '.join([vocabulary[t] for t in np.argsort(row)[-100:]]))
    # Display the generated image:
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

#-----------------------------------------------------------#
#                            MAIN                           #
#-----------------------------------------------------------#

if __name__ == "__main__":

    #-----------------------------------------------------------#
    #                 Dataframe  reading                        #
    #-----------------------------------------------------------#

    # define paths
    dir = os.getcwd()
    model_dir = dir +'/Models/'

    df = pd.read_csv('./Abstracts/Processed_Abstracts.csv')
   
    # remove missing abstract
    df = df.dropna(how='any', subset=['Processed_abstract'])

    #-----------------------------------------------------------#
    #                         TF-IDF                            #
    #-----------------------------------------------------------#

    #Vectorization: convert it into a format that can be handled by our algorithms
    """
    TF-IDF : measure of how important each word is to the instance out of the literature as a whole.
    """
    corpus = df['Processed_abstract'].values
   
    Tfidf = TfidfVectorizer(max_features=2 ** 9) # i.e. max features = 512
    
    # X = tfidf Matrix
    X = Tfidf.fit_transform(corpus)
    
    #X = vectorize(corpus, min_df = 5, max_df = 0.95, max_features = 512)
    print(f'TF-IDF Matrix shape: {X.shape}')

    # list of the features used in the tf-idf matrix. This is a vocabulary 
    vocabulary = Tfidf.get_feature_names()

    # #-----------------------------------------------------------#
    # #                        PCA                                #
    # #-----------------------------------------------------------#
    # # pca = PCA(n_components=0.95, random_state=0)
    # # X_reduced= pca.fit_transform(X.toarray())
    # # print(f'TF-IDF Matrix shape  after PCA: {X_reduced.shape}')

    #-----------------------------------------------------------#
    #                  Kmeans  clustering                       #
    #-----------------------------------------------------------#
    """ Indicate a number of optimal clusters k and load the corresponding K-means model
    or perfom a sihlouette score analysis in order to find it and run k-means with that number"""
    
    optimal_k = 14
    if os.path.isfile(f'{model_dir}Pubmed_Kmeans_{optimal_k}_clusters.pkl'): 
        k_means = joblib.load(f'{model_dir}Pubmed_Kmeans_{optimal_k}_clusters.pkl')
        clusters = k_means.labels_.tolist()
        df['Cluster'] = clusters
    else:
        """
        How many clusters? 
        To choose optimal number of clusters silhouette scores as well as computation of the sum of squared distances from each point to its assigned center are used
        """
        optimal_k =  K_means_optimal_K(X)

        # Run K-means with that number of clusters
        k_means= KMeans(n_clusters=optimal_k, random_state=0).fit(X)
        clusters = k_means.labels_.tolist()
        df['Cluster'] = clusters
        
        #-----save model-----#
        joblib.dump(k_means, f'{model_dir}./Pubmed_Kmeans_{optimal_k}_clusters.pkl')


    #-----clusters-----#
    palette =sns.color_palette('Spectral',optimal_k)
    print('# of publications per cluster')
    print(df['Cluster'].value_counts()) 
    sns.countplot(x="Cluster", data=df, palette=palette)
    plt.title('# of publications per cluster')
    plt.savefig(model_dir + "nb_of_publications_per_cluster.png")
    plt.show()
    plt.close()


    grouped = df.groupby(['Cluster','Category']).size()
    print(grouped)
    palette =sns.color_palette('Spectral',optimal_k)
    sns.countplot(x="Cluster", hue='Category', data=df, palette=palette)
    plt.legend(loc='best')
    plt.title('Distribution of categories per cluster')
    plt.savefig(model_dir + 'category_per_cluster.png')
    plt.show()
    plt.close()

    #-----------------------------------------------------------#
    #       Dimensionality Reduction with t-SNE                 #
    #-----------------------------------------------------------#
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE) reduces dimensionality
    while trying to keep similar instances close and dissimilar instances apart.
    It is mostly used for visualization, in particular to visualize clusters 
    of instances in high-dimensional space
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(verbose=1, perplexity=100, random_state=0)
    X_embedded = tsne.fit_transform(X.toarray())

    # sns settings
    sns.set(rc={'figure.figsize':(15,15)})

    # plot
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=clusters, legend='auto',  palette=palette)
    plt.title('t-SNE with Kmeans Labels')
    plt.savefig(model_dir + "t-sne_pubmed_kmeans.png")
    plt.show()
    plt.close()
    
    
    #-----------------------------------------------------------#
    #                   Top 10  words per cluster               #
    #-----------------------------------------------------------#

    print("Top 10 terms per cluster:")
    get_top_keywords(X, clusters, vocabulary, 10)

    #-----------------------------------------------------------------#
    # Topic modeling on each cluster with Latent Dirichlet Allocation #
    #-----------------------------------------------------------------#
    """
    For topic modeling => Latent Dirichlet Allocation
    In LDA, each document can be described by a distribution of topics
    and each topic can be described by a distribution of words.
    """

    """
    Gensim creates a unique id for each word in the document.
    The produced corpus  is a mapping of (word_id, word_frequency).
    For example, (0, 2) below implies, word id 0 occurs twice in the first document. 
    """

    #vectorized_data = df['Processed_abstract'].tolist()
    def LDA_model(vectorized_data, num_topics=20):
        # Create Dictionnary
        texts = [text.split() for text in vectorized_data]
        id2word = corpora.Dictionary(texts)
        #remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
        id2word.filter_extremes(no_below=1, no_above=0.8)
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        # quick view 
        #print(corpus[:1])
        # Model
        lda_model = models.LdaModel(corpus, num_topics=num_topics, 
                        id2word=id2word, 
                        update_every=1, 
                        chunksize=100, 
                        passes=10,
                        random_state =0,
                        per_word_topics=True)
      
        topics_per_cluster =  lda_model.show_topics(formatted=False, num_words=20)
        All_topics =[]
        for idx, topic in enumerate(topics_per_cluster):
            print('Topic: {} \nWords: {}'.format(idx, '|'.join([w[0] for w in topic[1]])))
            All_topics.append([w[0] for w in topic[1]])
            print('----next topic-----')
   
        vis = pyLDAvis.gensim.prepare(lda_model,corpus, id2word)
        pyLDAvis.show(vis)

    LDA_model(df.loc[df['Cluster'] == 6]['Processed_abstract'].tolist(), 3)

## pyLDAvis.show(vis) # open a server: http://127.0.0.1:8891/  
 
                
