# -*- coding: utf-8 -*-
# Article_clustering_app.py

# homemade package
import sys
sys.path.append('./src')

# Api package 
import streamlit as st

# basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os 
import re
import glob
from  collections import Counter
from pathlib import Path
import time

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

# Word Cloud
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#topic modeling
from gensim import corpora, models, similarities 
import gensim, spacy, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import TfidfModel
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim

warnings.filterwarnings("ignore",category=DeprecationWarning)

# visualization settings
fontsize = 8
sns.set_context("paper", rc={'figure.figsize':(1,1),'legend.fontsize': 3.0, "font.size":4.0,"axes.titlesize":8.0,"axes.labelsize":6.0, 'xtick.labelsize': 4.0, 'ytick.labelsize': 4.0})   
st.set_option('deprecation.showPyplotGlobalUse', False)


def display_app_header(main_txt,sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    Parameters
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <div style = "background.color:orange ; padding:15px">
    <h2 style = "color:white; text_align:center;"> {main_txt} </h2>
    <p style = "color:white; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)

def display_header(header):
    """
    function to display minor headers at user interface main pannel 
    Parameters
    ----------
    header: str -> the major text to be displayed
    """
    #view clean data
    html_temp = f"""<h2 style = "color:orange;text_align:center;"> {header} </h2>"""
    st.markdown(html_temp, unsafe_allow_html = True)


# page setting
display_app_header(main_txt='Genomic Medicine Literature Clustering',sub_txt='')
st.set_option('deprecation.showfileUploaderEncoding', False)

# Litterature categories available
categories = {
'Adeno-Associated-Virus': 'AAV',
'Epigenetics':'epigenetics',
'Gene Therapy':'gene_therapy', 
'Genome Engineering': 'genome_engineering',
'Immunology': 'immunology',
'Post-Translational Modification':'post_translational_modification',
'Regulatory Element': 'regulatory_element',
'Sequence' : 'sequence',
'Transfection' : 'transfection',
'Tropism' : 'tropism',
'Variant' : 'variant',
}

def category_selector(categories:dict):
    categories = [key for key in categories.keys()]
    selected_categories = st.sidebar.multiselect('Select categorie(s) to perform publication clustering and click on "Valid selection"', categories, key=1)
    st.write('You selected:', selected_categories)
    return selected_categories


def load_data(data_file, categories:dict, selected_categories:list, nrows=None):
    selected_categories_values = [categories[key] for key in selected_categories]
    data = pd.read_csv(data_file, nrows=nrows)
    data = data.dropna(how='any', subset=['Processed_abstract'])
    subset =  data[data['Category'].isin(selected_categories_values)]
    st.write(f'Total Number of publications selected : {subset.shape[0]}')
    return subset

def tfidf(df):
    #Vectorization :  convert it into a format that can be handled by our algorithms
    """
    TF-IDF : measure of how important each word is to the instance out of the literature as a whole.
    """
    corpus = df['Processed_abstract'].values
   
    Tfidf = TfidfVectorizer(max_features=2 ** 9)# i.e. max features = 512
    # X = tfidf Matrix
    X = Tfidf.fit_transform(corpus)
    #X = vectorize(corpus, min_df = 5, max_df = 0.95, max_features = 8000)
    st.write(f'TF-IDF Matrix shape : {X.shape}')
    # list of the features used in the tf-idf matrix. This is a vocabulary 
    vocabulary = Tfidf.get_feature_names()
    return corpus, X, vocabulary

def K_means_optimal_K(X):
    K = range(2, 5)
    summary = pd.DataFrame(columns=['k','silhouette_score','distortion'])
    #Sum_of_squared_distances =[]
    Silhouette_scores =[]
    K_list=[]
    for k in K:
        k_means = KMeans(n_clusters=k, random_state=0).fit(X)
        k_means.fit(X)
        #Sum_of_squared_distance = k_means.inertia_
        labels= k_means.predict(X)
        Silhouette_score = silhouette_score(X, labels, metric='euclidean')
        print("For n_clusters = {}, silhouette score is {}".format(k,Silhouette_score))
        #print("For n_clusters = {}, sum of squared distances is {}".format(k,Sum_of_squared_distance))
        K_list.append(k)
        #Sum_of_squared_distances.append(Sum_of_squared_distance)
        Silhouette_scores.append(Silhouette_score)
    summary['k'] = K_list
    summary['silhouette_score']= Silhouette_scores
    #summary['distortion']= Sum_of_squared_distances
    optimal_k = summary.iloc[summary['silhouette_score'].idxmax()]['k']
    st.write(f'Optimal number of clusters is {optimal_k}')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary['k'], y=summary['silhouette_score'],
    mode='lines+markers'))

    fig.update_layout(
    title="Silhouette Score plot",
    xaxis_title="K",
    yaxis_title="Score")

    st.plotly_chart(fig, use_container_width=True)
    return int(optimal_k)

def run_K_means(X,k):
    model = KMeans(n_clusters=k, random_state=0).fit(X)
    clusters = model.labels_.tolist()
    return model, clusters

def Nb_of_publications_per_cluster(df):
    df_count = df['Cluster'].value_counts().reset_index()
    df_count.columns = ['Cluster', 'Count']
    print(df_count) 
    data = [go.Bar(x=df_count['Cluster'], y=df_count['Count'])]
    layout = go.Layout(
    title="Number of publications per cluster",
    xaxis_title="Cluster",
    xaxis_type = "category",
    yaxis_title="Count")
    fig = go.Figure(data = data, layout = layout)
    st.plotly_chart(fig, use_container_width=True)

def Distribution_categories_per_cluster(df):
    categories = list(df['Category'].unique())
    grouped= df.groupby(['Cluster','Category']).size().unstack(level=-1)
    print(grouped)
    print(grouped.columns)
    df_count = grouped.reset_index()
    print(df_count)
    print(list(df_count['Cluster']))
    fig = go.Figure()
    colors  = px.colors.qualitative.Plotly
    for i, category in enumerate(categories,0):
        print(i, category)
        fig.add_trace(
            go.Bar(x=list(df_count['Cluster']), y=list(df_count[category]), name = category, marker_color=colors[i])
        )
    fig.update_layout(title="Distribution of categories per cluster", autosize=True,width=800,height=600, legend_title_text='Category')
    st.plotly_chart(fig, use_container_width=False)

def show_nb_pubs():
    st.sidebar.write('Show number of publication per cluster?')
    check_nb_pubs = st.sidebar.checkbox('Yes',value = False, key=1)
    return check_nb_pubs

def show_distribution_category():
    st.sidebar.write('Display category distribution per cluster?')
    check_dist_cat = st.sidebar.checkbox('Yes',value = False, key=2)
    return check_dist_cat

def Tsne(X, df, categories):
    tsne = TSNE(verbose=1, perplexity=100, random_state=0)
    X_embedded = tsne.fit_transform(X.toarray())
    df['x'] = X_embedded[:,0]
    df['y'] =  X_embedded[:,1]

    nb_category  = df['Category'].nunique()
    symbols = ['circle','square','diamond','hexagon','pentagon','cross','octagon','star','triangle-se','triangle-sw','x']
    symbol_map  = dict( zip( [value for value in categories.values()], symbols) )
    colors  = px.colors.qualitative.Plotly
    #sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=clusters, legend='auto',  palette=sns.color_palette("Set2"))
    fig = go.Figure()
    for k in range(0,df['Cluster'].nunique()):
        sub_df=df[df['Cluster']==k]
       
        fig.add_trace(go.Scatter(x=sub_df['x'], y=sub_df['y'],  mode='markers', marker_color=colors[k], hovertext = "Title: " + sub_df["Title"].astype(str) + "<br>Category: " + sub_df["Category"].astype(str), legendgroup=f"Cluster_{k}", name = f'Cluster_{k}'))
    #fig.add_trace(go.Scatter(x=X_embedded[:,0], y=X_embedded[:,1],  mode='markers', marker_color=df['Cluster'], marker_symbol = df['Category'], hovertext = "Title: " + df["Title"].astype(str) + "<br>Category: " + df["Category"].astype(str)  ))
    fig.update_layout(title="t-SNE scatter plot with K-means labels", autosize=True,width=800,height=600, legend_title_text='Cluster')
    # plt.savefig(save_dir + "t-sne_pubmed_kmeans.png")
    st.plotly_chart(fig, use_container_width=False)
    return X_embedded

def get_top_keywords(tfidf_matrix, clusters, vocabulary, n_terms):
    tfidf_df = pd.DataFrame(X.todense())
    tfidf_df['Cluster']=clusters
    grouped = tfidf_df.groupby('Cluster').mean()
    for index,row in grouped.iterrows():
        st.write(f'\nCluster {index}')
        st.write(', '.join([vocabulary[t] for t in np.argsort(row)[-n_terms:]]))
        wordcloud = WordCloud(max_words=50).generate(' '.join([vocabulary[t] for t in np.argsort(row)[-100:]]))
        # Display the generated image:
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
        st.pyplot()

def cluster_selector():
    cluster_selection = st.sidebar.number_input('Choose a cluster on which to perfom Topic Modeling',min_value=0,value=0,step =1)
    return int(cluster_selection)

def number_of_topics_LDA_selector():
    nb_of_topics= st.sidebar.number_input('Select a number of topics for topic modeling analysis',min_value=1,value=3,step =1)
    return int(nb_of_topics)


#vectorized_data = df['Processed_abstract'].tolist()
def LDA_model(vectorized_data, num_topics=20):
    # Create Dictionnary
    texts = [text.split() for text in vectorized_data]
    id2word = corpora.Dictionary(texts)
    #remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
    id2word.filter_extremes(no_below=1, no_above=0.8)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
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
        st.write('Topic: {} \n\n Key Words:  \n\n {}'.format(idx, ' | '.join([w[0] for w in topic[1]])))
        All_topics.append([w[0] for w in topic[1]])
        st.write('**--------Next Topic--------**')
    # vis = pyLDAvis.gensim.prepare(lda_model,corpus, id2word)
    # pyLDAvis.show(vis)


#################################################################
#################################################################
#                   Streamlit  main flow                        #      
#################################################################
#################################################################

# Path
dir = os.getcwd()
save_dir = os.path.join(dir, 'Article_clustering_output/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
data_file = os.path.join(dir, 'Abstracts', 'Processed_Abstracts.csv')

# category selection
display_header('Data')
selected_categories= category_selector(categories)

# data loading 
df = load_data(data_file, categories, selected_categories, nrows=None)

# Word Count statistics
#st.write(df['Abstract_word_count'].describe())

#if selected_categories :
button_selection = st.button('Valid selection')

# show more...
check_nb_pubs = show_nb_pubs()
check_dist_cat = show_distribution_category()

# LDA 
cluster_selection = cluster_selector()
num_topics = number_of_topics_LDA_selector()


if button_selection:
    display_header('TF-IDF')

    #-----------------------------------------------------------#
    #                         TF-IDF                            #
    #-----------------------------------------------------------#
    tfidf_state = st.text('TF-IDF vectorization...')
    # Load 10,000 rows of data into the dataframe.
    corpus, X, vocabulary = tfidf(df)
    # Notify the reader that the data was successfully loaded.
    tfidf_state.text('TF-IDF vectorization...done!')

    #-----------------------------------------------------------#
    #                  Kmeans  clustering                       #
    #-----------------------------------------------------------#
    # Optimal k
    display_header('K-means')
    st.markdown('How many clusters?')
    optimal_k_state = st.text('Silhouette score calculation...')
    optimal_k = K_means_optimal_K(X)
    optimal_k_state.text('Silhouette score...done!')
    
    # Run K-means
    st.markdown('Run K-means')
    model, clusters = run_K_means(X, optimal_k)

    # Add cluster assigments to new df_cluster
    df_cluster = df.copy()
    df_cluster['Cluster'] = clusters
    
    #st.write(df_cluster.head())

    if check_nb_pubs:
        Nb_of_publications_per_cluster(df_cluster)

    if check_dist_cat:
        Distribution_categories_per_cluster(df_cluster)
    
    #-----------------------------------------------------------#
    #       Dimensionality Reduction with t-SNE                 #
    #-----------------------------------------------------------#
    display_header('t-SNE')
    tsne_state = st.text('t-SNE...')
    X_embedded = Tsne(X, df_cluster,categories)
    tsne_state.text('t-SNE...done!')

    #-----------------------------------------------------------#
    #                top 10 words  and word cloud               #
    #-----------------------------------------------------------#
    display_header('Top words per cluster')
    get_top_keywords(X, clusters, vocabulary, 10)


    #-----------------------------------------------------------------#
    # Topic modeling on each cluster with Latent Dirichlet Allocation #
    #-----------------------------------------------------------------#
    display_header('Topic modeling on each cluster with Latent Dirichlet Allocation')
    LDA_model(df_cluster.loc[df_cluster['Cluster'] == 1]['Processed_abstract'].tolist(), 3)



