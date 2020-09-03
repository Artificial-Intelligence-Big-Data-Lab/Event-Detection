# -*- coding: utf-8 -*-
from json2table import convert
import pprint
import pymongo, pickle, os
from datetime import datetime, timedelta
import numpy as np
from gensim.parsing import preprocessing as pproc
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from nltk.cluster import cosine_distance
from sklearn.metrics.pairwise import cosine_distances
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.agglomerative import agglomerative, type_link
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from dunn_index import dunn
import matplotlib.pyplot as plt
import spacy

from create_lexicons import process_news_text, process_string, create_lexicons, fetch_lexicons, calc_lexicon_match
from utils import get_all_ids, fetch_previous_news, get_news_per_day, get_market_per_day, get_deltas_per_date, TEST_KEYWORDS_DICT
from plotter import plot_clusters, plot_statistics, print_boxplot, plot_price_chart

# We employ a word-embedding model pre-trained on Google news and adjusted in such a way, that
# makes embeddings more aware of the sentiments of the words.
# Reference: https://towardsdatascience.com/sentiment-preserving-word-embeddings-9bb5a45b2a5
# The model is loaded as a global variable to avoid reloading at each iteration,
# because it is a slow and memory-consuming operation
with open('../word2vec_data/google_word2vec_sentiment.bin', 'rb') as f:
    WORD_EMBEDDING_MODEL = pickle.load(f)
    
# used to filter the senteces of the documents based on the tense detection
#NLP_MODEL = spacy.load('en_core_web_lg')
NLP_MODEL = None
    
# If you want to use the classic model, without sentiment adjustment, use this instruction:
#with open('../word2vec_data/google_word2vec_classic.bin', 'rb') as f:
#    WORD_EMBEDDING_MODEL = pickle.load(f) 

#we use a global variable to store the initian centroids that have been computed already; this is done
#to ensure that on the same day, a cluster algorithm always returns the same output.
INITIAL_CENTROIDS = {}

# global variables used to pick the colors for the plots
ALL_COLORS = ['red', 'green', 'blue', 'gold', 'magenta', 'cyan', 'gray', 'orange', 'teal', 'pink', 'purple']


"""
Performs the evaluation of the alert-generation phase. The evaluation is performed for different configurations of parameters, chosen from
'day_tolerance_range', 'delta_threshold_range' and 'alert_threshold_range' (see below for reference).


*** Definition of the ground truth:

    'weekly_deltas_per_date' is a dict where keys are dates and values are the corresponding stock price variation computed on a 7-days horizon. 
    Ex: for 2018-01-01, the variation is given by this formula: 100 * (close(2018-01-08) - open(2018-01-01) / open(2018-01-01)).
    
    The events that are included in the ground truth are defined by a couple (event_start_date, event_end_date) and are selcted deterministically 
    by looking at the values of 'weekly_deltas_per_date' in the following way: if weekly_deltas_per_date[2018-01-01] >= delta_threshold, 
    then 2018-01-01 is marked as an event-day. In order to obtain the time interval associated to the event, we join the days marked as event-days that
    are contiguous or closer than 'day_tolerance'; when we reach the end of the chain of event-days, we get the event_end_date. We keep scrolling
    the dates in search for the other events.
    
    Ex: if day_tolerance = 2 and the event-days are 2018-01-01, 2018-01-02, 2018-01-04, 2018-01-06, 2018-01-15, 2018-01-16, our final list of events would be defined by the
    following pairs of (event_start_date, event_end_date) : [(2018-01-01, 2018-01-06), (2018-01-15,2018-01-16)]
    
    
*** Definition of the generated alerts

    'assigned_tweets_per_date' is a dict where keys are dates and values are the percentage of assigned tweets on that date.
    An alert is generated for date d if assigned_tweets_per_date[d] >= alert_threshold.
    

*** Evaluation

    We calculate precision and recall.
    
    Recall is defined as the percentage of events, among those included in the ground truth, that we spotted by means of the generated alerts.
    Specifically, for every (event_start_date, event_end_date) pair, we check if there is at least one alert generated between these two dates. If yes, then
    the event is marked as spotted. Finally, the recall is computed as number of spotted_events divided by number of events in the ground truth.
    
    Precision is defined as the percentage of generated alerts that are included between one (event_start_date, event_end_date) pair.

    
"""
def evaluate_alerts(weekly_deltas_per_date, assigned_tweets_per_date, day_tolerance_range=[1,3,7], delta_threshold_range=[2,3,4], alert_threshold_range=[1,2,3,4,5]):
    
    for day_tolerance in day_tolerance_range:
        for delta_threshold in delta_threshold_range:
            original_test_labels = {d:1 if abs(weekly_deltas_per_date[d]) >= delta_threshold else 0 for d in sorted(weekly_deltas_per_date)}
            
#            print('\nAll dates:')
#            for d in original_test_labels:
#                print(d)
            
#            print('\nOriginal event dates:')
#            for d in original_test_labels:
#                if original_test_labels[d] == 1:
#                    print(d)
                    
            extended_test_labels = {d:0 for d in original_test_labels}
            current_date = sorted(list(original_test_labels.keys()))[0]
            max_date = sorted(list(original_test_labels.keys()))[-1]
            while current_date <= max_date:
                if (current_date in original_test_labels) and (original_test_labels[current_date] == 1):
                    for td in range(-day_tolerance,day_tolerance+1):
                        if current_date + timedelta(days=td) in original_test_labels:
                            extended_test_labels[current_date + timedelta(days=td)] = 1
                current_date = current_date + timedelta(days=1)
                
#            print('\nExtended event dates:')
#            for d in extended_test_labels:
#                if extended_test_labels[d] == 1:
#                    print(d)
                            
            test_events = []
            previous_label = 0
#            print('\nForming events...')
            for d in extended_test_labels:
#                print()
#                print(d)
#                print('Current label:', extended_test_labels[d])
#                print('Previous label:', previous_label)
                if previous_label == 0 and extended_test_labels[d] == 1:
#                    print('\nEvent start on', d)
                    event_start_date = d
                elif previous_label == 1 and extended_test_labels[d] == 0:
#                    print('Event end on', d)
                    event_end_date = d
                    test_events.append((event_start_date, event_end_date))
                previous_label = extended_test_labels[d]
            
            if previous_label == 1 and extended_test_labels[d] == 1:
                test_events.append((event_start_date, event_end_date))
                    
#            print('\nEvent ranges:')
#            for e in test_events:
#                print(e)
                            
            
            for alert_threshold in alert_threshold_range:
                alert_dates = [d for d in assigned_tweets_per_date if assigned_tweets_per_date[d] >= alert_threshold]
            
                print('\n**************\nDelta Threshold:', delta_threshold)
                print('Alert Threshold:', alert_threshold)
                print('Day Tolerance:', day_tolerance)
                print('\nN. events in test:', len(test_events))
                print('N. alerts:', len(alert_dates))
                
#                print('\nComputing recall...')
                spotted_events_in_test = 0
                for event_start_date,event_end_date in test_events:
                    for d in alert_dates:
                        if d >= event_start_date and d <= event_end_date:
                            spotted_events_in_test += 1
#                            print('Hit:', d, 'between', event_start_date, 'and', event_end_date)
                            break
                if len(test_events) > 0:
                    recall = spotted_events_in_test / len(test_events)
                else:
                    recall = 'No computable, because 0 events were selected for the ground truth'
                print('\nRecall:', recall)
                
                relevant_events_in_alerts = 0
                for d in alert_dates:
                    for event_start_date,event_end_date in test_events:
                        if d >= event_start_date and d <= event_end_date:
                            relevant_events_in_alerts += 1
#                            print('Hit:', d, 'between', event_start_date, 'and', event_end_date)
                            break
                if len(alert_dates) > 0:
                    precision = relevant_events_in_alerts / len(alert_dates)
                else:
                    precision = 'No computable, because 0 alerts were generated'
                print('\nPrecision:', precision)
                

    
    
"""
Given a list of test_keywords defined a priori, this function computes:
    - percentage of test_keywords that appear among the relevant words, for each cluster;
    - percentage of articles that contain at least one test keyword, for each cluster.
    
In particular, this function is used to verify the match between the obtained clusters and an event known a priori (Brexit, Trump's election, etc).
For example, to test the result on Brexit referendum, the user should set test_keyword to something like ['brexit', 'uk', 'britain', 'leave', 'polls'].
The length is arbitrary.
"""
def run_keywords_test(cluster_relevant_words, cluster_articles, test_keywords, n_top_words=10):
    
    stemmed_test_keywords = list(set([pproc.stem(w) for w in test_keywords]))
        
    keywords_in_relevant_words = [0 for c in range(len(cluster_relevant_words))]
    keywords_in_titles = [0 for c in range(len(cluster_relevant_words))]
    keywords_in_snippets = [0 for c in range(len(cluster_relevant_words))]
    for c in range(len(cluster_relevant_words)):
        stemmed_relevant_words = []
        for w,s in cluster_relevant_words[c]:
            sw = pproc.stem(w)
            if sw not in stemmed_relevant_words:
                stemmed_relevant_words.append(sw)
            if len(stemmed_relevant_words) >= n_top_words:
                break
    
        keywords_in_relevant_words[c] = 100 * (sum([1 for w in stemmed_relevant_words if w in stemmed_test_keywords]) / n_top_words)
        cluster_titles = [process_string(title, stemming=True) for i,date,title,snippet,dist in cluster_articles[c]]
        keywords_in_titles[c] = 0
        for title in cluster_titles:
            for tk in stemmed_test_keywords:
                if tk in title:
                    keywords_in_titles[c] += 1
                    break
        keywords_in_titles[c] = 100 * (keywords_in_titles[c] / len(cluster_titles))
        
        cluster_snippets = [process_string(snippet, stemming=True) for i,date,title,snippet,dist in cluster_articles[c]]
        keywords_in_snippets[c] = 0
        for snippet in cluster_snippets:
            for tk in stemmed_test_keywords:
                if tk in snippet:
                    keywords_in_snippets[c] += 1
                    break
        keywords_in_snippets[c] = 100 * (keywords_in_snippets[c] / len(cluster_snippets))
    
    return keywords_in_relevant_words, keywords_in_titles, keywords_in_snippets


"""
Given a list of relevant words for each cluster, this function computed the average overlapping between them in this way:
    - compute the Jaccard index between the first 'n_top_words' relevant words each pair of clusters
    - average the results to obtain a global score.
    
The Jaccard Index between two lists is defined as the size of the intersection divided by the size of the union.
"""
def calc_relevant_words_overlap(cluster_relevant_words, n_top_words=10):
    
    overlaps = []
    for i in range(len(cluster_relevant_words)-1):    
        list1 = [w for w,s in cluster_relevant_words[i][:n_top_words]]            
        for j in range(i+1, len(cluster_relevant_words)):
            list2 = [w for w,s in cluster_relevant_words[j][:n_top_words]]
            overlap = 0
            for w in list1:
                if w in list2:
                    overlap += 1
            overlap = overlap / len(set(list1+list2))
            overlaps.append(overlap)
#            print('\nLIST 1:', cluster_relevant_words[i][:n_top_words])
#            print('LIST 2:', cluster_relevant_words[j][:n_top_words])
#            print('Overlap:', overlap)
    return np.average(overlaps)

    
"""
Given an array of documents, each represented by a dict as retreived from the mongo database, returns the
texts, relevant companies, dates, ids, titles and snippets for each document, stored in distinct arrays.
All the lists have the same length; at the same position of each list, you find the information about the same document.

Params:
    - documents_array : array of documents, each represented by a dict as retreived from the mongo database
    - companies : list of companies we are interested in (NOTE if interested in SP500 index, set companies = ['SP500'])
    - industry : industry we are interested in (NOTE if interested in SP500 index, set industry = 'SP500')
    - relevance_mode : if 'about', retrieves a document iff one of 'companies' is the main focus of the document
                       if 'relevant', retrieves a document if one of 'companies' is somehow relevant to the document
                       if 'both', retrieves a document in one of the two cases above
                       This parameter depends on DNA metadata; if interested in SP500, simply set this to 'both'
    - use_tense_detection : if True, filters the sentences in each document, keeping only those that employ at least one
                            verb in the future tense
                       
Returns:
    - processed_texts : list of texts, filtered out of stopwords but NOT stemmed 
    - associated_companies : list of relevant companies associated to each document (it's a list of lists)
    - dates : list of dates associated to each document
    - ids : list of ids associated to each document
    - titles : list of titles associated to each document
    - snippets : list of snippets associated to each document
"""
def organize_documents(documents_array, companies, industry, relevance_mode='both', use_tense_detection=False):
    
    processed_texts = []
    associated_companies = []
    dates = []
    ids = []
    titles = []
    snippets = []
    for i in range(len(documents_array)):
        item = documents_array[i]
        if relevance_mode == 'both':
            companies_set = list(set(item['about_companies'] + item['relevant_companies']))
        elif relevance_mode == 'about':
            companies_set = item['about_companies']
        elif relevance_mode == 'relevant':
            companies_set = item['relevant_companies']
        
        #check if at least one of the companies we are interested in is contained in the document-companies
        for c in companies_set:
            if c in companies:
                text = process_news_text(item, stemming=False, remove_stopwords=True, use_tense_detection=use_tense_detection, nlp_model=NLP_MODEL if use_tense_detection else None)
                if len(text) > 1 and text not in processed_texts:   # we need this check, because some tweets have duplicates (spam)
                    processed_texts.append(text)
                    if industry == 'SP500':
                        associated_companies.append(companies_set)
                    else:
                        associated_companies.append([comp for comp in companies_set if comp in companies])
                    dates.append(datetime.strptime(item['converted_ingestion_datetime_utc-5'], '%Y-%m-%d %H:%M:%S').date())
                    ids.append(item['an'])
                    if 'title' in item:
                        titles.append(item['title'])
                    else:   #this happens with tweets, that don't have keys 'title' and 'snippet', but only 'body'
                        titles.append(item['body'])
                    
                    if 'snippet' in item:
                        snippets.append(item['snippet'])
                    else:   #this happens with tweets, that don't have keys 'title' and 'snippet', but only 'body'
                        snippets.append('not_found')
                    
                    break
    
    return processed_texts, associated_companies, dates, ids, titles, snippets
    

"""
Given an array of vectors and the corresponding labels indicating the cluster, computes and returns:
    - the average silhouette score
    - the silhouette score for each vector
    - the silhouette score averages for each cluster.
    
Refer to function 'cluster_news' for further reference on parameter.


References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html#sklearn.metrics.silhouette_samples
"""
def calc_silhouette(vectors, labels):
    
    silhouette_avg = silhouette_score(vectors, labels, metric='cosine')
    samples_silhouette_values = silhouette_samples(vectors, labels, metric='cosine')
    silhouette_per_cluster = []
    for l in np.unique(labels):
        silhouette_per_cluster.append(np.average(samples_silhouette_values[labels == l]))
    
    return silhouette_avg, samples_silhouette_values, silhouette_per_cluster

"""
Given an array of vectors and the corresponding labels indicating the cluster, computes and returns the dunn score,
using the external module dunn_index.py.

Refer to function 'cluster_news' for further reference on parameter.

Reference: https://www.geeksforgeeks.org/dunn-index-and-db-index-cluster-validity-indices-set-1/
"""
def calc_dunn(vectors, labels):
    
    cluster_list = [[] for c in range(max(labels)+1)]
    for i in range(len(labels)):
        cluster_list[labels[i]].append(vectors[i])
        
    cluster_list = [cl for cl in cluster_list if len(cl) > 0]
    for c in range(len(cluster_list)):
        cluster_list[c] = np.array(cluster_list[c])
        
    return dunn(cluster_list)
        


"""
Detects the points that, based on one or two metrics, are considered outliers.
More precisely, the points whose score is below the percentile defined by percentile_cutoff are marked as outliers.
Normally, the metrics used to compute the scores are silhouette or proximity (defined as 1 - cosine(point, centroid))

Example:
    scores_per_sample_1 is a list of silhouette scores, while scores_per_sample_2 is a list of proximities.
    If percentile_cutoff == 25, all the points that have a silhouette score below the 25th percentile of scores_per_sample_1
    OR a proximity below the 25th percentile of scores_per_sample_2 are marked as outliers.

Params:
    - scores_per_sample_1 : list/array of scores, each for each point
    - scores_per_sample_2 (optional) : another list/array of scores, each for each point
    - percentile_cutoff : percentile below which a point is marked as outlier.
    
Returns:
    - list of the same length as scores_per_samples_1; contains 1s at the positions where the outliers are and 0s elsewhere.
"""
def find_outliers(scores_per_sample_1, scores_per_sample_2=[], percentile_cutoff=25):
    
    threshold_1 = np.percentile(scores_per_sample_1, percentile_cutoff)
    if len(scores_per_sample_2) > 0:
        threshold_2 = np.percentile(scores_per_sample_2, percentile_cutoff)
    else:
        scores_per_sample_2 = np.zeros(len(scores_per_sample_1))
        threshold_2 = -1
    is_outlier = np.array([1 if s1 < threshold_1 or s2 < threshold_2 else 0 for s1,s2 in zip(scores_per_sample_1, scores_per_sample_2)])

    return is_outlier

"""
Used in rare cases in which, after outlier removal, one or more clusters are suppressed.
This function renames labels so that they always range from 0 to (n_clusters-1), without jumps.
It assumes that labels corresponding to outliers have been set to -1 (i.e., is_outlier[i] == 1 <=> labels[i] == -1)

Examples:
    before outlier removal, we have labels [0,0,1,2,0]
    after outlier removal, we have labels [0,0,-1,2,0]
    after rename_labels, we have labels [0,0,-1,1,0]
"""
def rename_labels(labels):
    
    labels = np.array(labels)
    new_label_set = range(len(np.unique(labels[labels != -1])))
    old_label_set = np.unique(labels[labels != -1])
    for ol,nl in zip(old_label_set,new_label_set):
        labels[labels == ol] = nl
    return labels

"""
This function is used mainly in two cases:
    - for agglomerative algorithm, which does not rely on the notion of centroid, so we need to compute one at the end
    - for recomputing the centroids after outlier removal (for all kinds of algorithm)
    
Possible values for method are:
    - mean : the centroid is the average of all the points of the cluster
    - median : the centroid is the median of all the points of the cluster
"""
def compute_centroids(vectors, labels, method='mean'):
    
    centroids = []
    for c in np.unique(labels):
        indices = [k for k in range(len(labels)) if labels[k] == c]
        if method == 'mean':
            new_centroid = np.average(vectors[indices], axis=0)
        elif method == 'median':
            new_centroid = np.median(vectors[indices], axis=0)
        centroids.append(new_centroid)
    centroids = np.array(centroids)
    return centroids
    

"""
Function that finds the best clustering algorithm for the data passed as parameter (news_vectors, which contains the vectors we want to cluster)
by calling recursively 'cluster_news'.
The ensemble algorithm works in the following way:
    1. apply 3 different cluster algorithm; each algorithm returns a number of clusters k
    2. use majority voting to find the best k (e.g. if 2 algorithms find 6 clusters, then we select k = 6);
      in case of ties, bigger k are preferred.
    3. among the algorithms that won the previous vote, select the one with higer 'search_score'.
    
Possible values for search_score are 'silhouette' and 'dunn'. 
This parameter is used for 2 purposes:
    - finding the best candidate algorithm, as described in step 3;
    - finding the best k for each individual algorithm, as indicated in step 1. See 'cluster_news' for further reference on this point.

Returns output in the same form as 'cluster_news' (see for reference)
"""
def ensemble_manager(news_vectors, date, search_score='silhouette'):
    
    #STEP 1
    output_per_alg = {}
    n_centroids_count = np.zeros(10)
    for alg in ['kmeans', 'kmedians', 'agglomerative']:
        labels, centroids, _a = cluster_news(news_vectors, date, cluster_alg=alg, k_search_score=search_score)
        n_centroids_count[len(centroids)] += 1
        if search_score == 'silhouette':
            score = silhouette_score(news_vectors, labels, metric='cosine')
        elif search_score == 'dunn':
            score = calc_dunn(news_vectors, labels)
        
        output_per_alg[alg] = {'labels' : labels, 
                               'centroids' : centroids, 
                               'score' : score,
                               'n_centroids': len(centroids)}
    
    #STEP 2
    selected_n_centroids = max(np.argwhere(n_centroids_count == max(n_centroids_count)).flatten().tolist())
    
    #STEP 3
    best_score = 0
    best_algorithm = ''
    for alg in output_per_alg:
        if output_per_alg[alg]['n_centroids'] == selected_n_centroids:
            if output_per_alg[alg]['score'] > best_score:
                best_score = output_per_alg[alg]['score']
                best_algorithm = alg
    
    return output_per_alg[best_algorithm]['labels'], output_per_alg[best_algorithm]['centroids'], best_algorithm




    
"""
Applies the clustering algorithm defined by 'cluster_alg' on the data defined by 'news_vectors'.

Params:
    - news_vectors : array of vectors, each representing a news. This is the data we want to clusterize
    - date : date on which the clustering is being computed (only used to store the INITIAL CENTROID, see note above)
    - cluster_alg : one of 'kmeans', 'kmedians', 'kmedoids', 'agglomerative', 'dbscan', 'ensemble' (see function above)
    - k_search_method : one of 'silhouette', 'dunn'; metric used to evaluate a clustering, in order to find the best number of clusters
    - n_clusters : number of clusters to compute (ignored if k_search_method is not None)
    - centroid_type : one of 'mean', 'median'; used if cluster_alg == 'agglomerative' (see function 'compute_centroids' for reference)
    
Returns:
    - labels : an array of the same length as news_array; labels[i] contains the label assigned to news_vectors[i]
    - centroids : an array of k elements, where k is the number of clusters extracted by the algorithm; 
      label values correspond to positions in centroid array (if labels[i] == 2, then the centroid for news_vectors[i] is at centroids[2])
    - cluster_alg : return value used just for compliance with the output of 'ensemble_manager', which returns the algorithm selected by the ensemble
    
For reference on clustering algorithms:
    - https://pyclustering.github.io/docs/0.9.0/html/db/de0/classpyclustering_1_1cluster_1_1center__initializer_1_1kmeans__plusplus__initializer.html
    - https://pyclustering.github.io/docs/0.9.0/html/da/d22/classpyclustering_1_1cluster_1_1kmeans_1_1kmeans.html
    - https://pyclustering.github.io/docs/0.9.0/html/df/d68/classpyclustering_1_1cluster_1_1kmedians_1_1kmedians.html
    - https://pyclustering.github.io/docs/0.9.0/html/d0/dd3/classpyclustering_1_1cluster_1_1kmedoids_1_1kmedoids.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
    
For reference on the silhouette method to select best k:
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    
"""
def cluster_news(news_vectors, date, cluster_alg='kmeans', k_search_score='silhouette', n_clusters=5, centroid_type='mean', vocabulary=None):
    
    # custom function that wraps the cosine_distances function provided by scikit-learn.
    # This is necessary in order to use the cosine as metric for kmeans funtion by py_clustering library
    def my_cosine(point1, point2):
        point1 = np.array(point1)
        point2 = np.array(point2)
        d = cosine_distances(point1.reshape(1,-1), point2.reshape(1,-1))[0][0]
        return d
     
    # use this if you want to visualize the clustering for every choice of k, for example during debugging
#    vocabulary_word_embeddings = np.array([WORD_EMBEDDING_MODEL[w] for w in vocabulary])
#    all_vectors = np.append(vocabulary_word_embeddings, news_vectors, axis = 0)
#    all_tsne_vectors = TSNE(n_components=2, metric='cosine', perplexity=30, early_exaggeration=3, random_state=0).fit_transform(all_vectors)
#    news_tsne_vectors = all_tsne_vectors[-len(news_vectors):]
        
    if cluster_alg == 'ensemble':
        return ensemble_manager(news_vectors, date, search_score=k_search_score)
    
    # the silhouette method does not apply to dbscan algorithm
    if cluster_alg == 'dbscan':
        k_search_score = None
        n_clusters = 0
    
    #if k_search_score is None, then simply n_clusters are extracted
    if not k_search_score:
        min_n_clusters = n_clusters
        max_n_clusters = n_clusters + 1
    # otherwise, we test each clustering for k in range from 2 to 10
    else:
        min_n_clusters = 2
        max_n_clusters = min(10, len(news_vectors)) # make sure that k < number of news
        
    best_score = -9999  # initialize the score to lowest value
    
    # we test the clustering for all k in this range
    for k in range(min_n_clusters, max_n_clusters):
        
        # we need to compute the initial centroids for kmeans, kmeans or kmedoids
        # If we have already computed them in a previous iteration, then simply retrieve them from the
        # global dict (to make sure that the output will be the same)
        if cluster_alg in ('kmeans', 'kmedians', 'kmedoids'):
            if (date,cluster_alg,k) in INITIAL_CENTROIDS:
                initial_centroids = INITIAL_CENTROIDS[(date, cluster_alg, k)]
            else:
                initial_centroids = kmeans_plusplus_initializer(news_vectors, k).initialize()
                for c in range(len(initial_centroids)):
                    initial_centroids[c] = np.array(initial_centroids[c])
                initial_centroids = np.array(initial_centroids)
                INITIAL_CENTROIDS[(date, cluster_alg, k)] = initial_centroids
        
        # some differences in tratement between algorithms are due to the fact that
        # kmeans, kmedians and kmedoids come from py_clustering library,
        # while agglomerative and dbscan come from scikit-learn
        if cluster_alg == 'kmeans':
            clusterer = kmeans(news_vectors, initial_centroids, metric=distance_metric(type_metric.USER_DEFINED, func=my_cosine, numpy_usage=True))
        elif cluster_alg == 'kmedians':
            clusterer = kmedians(news_vectors, initial_centroids, metric = distance_metric(type_metric.USER_DEFINED, func=my_cosine, numpy_usage=True))
        elif cluster_alg == 'kmedoids':
            # kmedoids requires real vectors picked from news_vectors as initial centroid;
            # so, for each initial centroid previously computed, we need to find the closest vector 
            initial_medoids = []
            for ic in initial_centroids:
                closest_index = 0
                shortest_distance = 1
                for i in range(len(news_vectors)):
                    d = cosine_distance(ic, news_vectors[i])
                    if d < shortest_distance:
                        closest_index = i
                        shortest_distance = d
                initial_medoids.append(closest_index)
            clusterer = kmedoids(news_vectors, initial_medoids, metric = distance_metric(type_metric.USER_DEFINED, func=my_cosine, numpy_usage=True))
        elif cluster_alg == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='average')
        elif cluster_alg == 'dbscan':
            # these commented lines would be useful to find the best choice of parameter 'eps' for DBSCAN, but need some testing and adjustments
            # (https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc)
            # At the moment, we are discarding dbscan anyways
#            neighbors = NearestNeighbors(n_neighbors=2, metric='cosine').fit(news_vectors)
#            distances, indices = neighbors.kneighbors(news_vectors)
#            distances = np.sort(distances, axis=0)[:,1]
#            differences = [distances[i] - distances[i-3] for i in range(3, len(distances))]
#            plt.plot(distances)
#            plt.show()
#            plt.plot(differences)
#            plt.show()
#            candidate_eps = max(differences)
#            print('\nCandidate EPS:', candidate_eps)
            clusterer = DBSCAN(eps=0.5, min_samples=3, metric='cosine')
        
        if cluster_alg in ('kmeans', 'kmedians', 'kmedoids'):
            clusterer.process()
            clusters = clusterer.get_clusters()
            labels = np.zeros(len(news_vectors), dtype=np.int8)
            for c in range(len(clusters)):
                labels[clusters[c]] = c
            if cluster_alg == 'kmeans':
                centroids = clusterer.get_centers()
            elif cluster_alg == 'kmedians':
                centroids = clusterer.get_medians()
            elif cluster_alg == 'kmedoids':
                centroids = [news_vectors[index] for index in clusterer.get_medoids()]    
            centroids = np.array([np.array(centroids[c]) for c in range(len(centroids))])
        
        # these algorithms require that we manually compute the centroids after the clustering is done
        elif cluster_alg in ('agglomerative', 'dbscan'):
            clusterer.fit(news_vectors)
            labels = clusterer.labels_
            centroids = compute_centroids(news_vectors[labels != -1], labels[labels != -1], method=centroid_type)
            
        if not k_search_score:
            return labels, centroids, cluster_alg
        
        elif k_search_score == 'silhouette':
            score = silhouette_score(news_vectors, labels, metric='cosine')
        # you can ignore davies_bouldin; it's an alternative metric, but we are discarding it
        elif k_search_score == 'davies_bouldin':
            score = -1 * davies_bouldin_score(news_vectors, labels)  #we multiply for -1, so that 0 is the maximum value, not the minimum
        elif k_search_score == 'dunn':
            score = calc_dunn(news_vectors, labels)
            
        if score > best_score:
#            best_k = k
            best_score = score
            best_labels = labels
            best_centroids = centroids
            
    # use these lines if you want to visualize the clustering for every choice of k, for example during debugging 
#        print('\n***\nAlgorithm:', cluster_alg)
#        print('\nN. clusters:', k)
#        print('Score:', score)
#        for l in np.unique(labels):
#            indices = [k for k in range(len(labels)) if labels[k] == l]
#            plt.scatter(news_tsne_vectors[indices,0], news_tsne_vectors[indices,1], c=ALL_COLORS[l], label='Cluster #'+str(l))
#        plt.title('N. of clusters: ' + str(k) + ' - Silhouette score: ' + str(round(score,2)))
#        plt.legend()
#        plt.show()
#    
#    print('\nBest k:', best_k)   
    labels = best_labels
    centroids = best_centroids
                    
    return labels, centroids, cluster_alg

"""
Given some news clusters and a set of tweets, this function tries to assign each tweet to the most similar cluster,
in case there is one within a certain threshold of similarity.
Each tweet is represented in the same way in which news are represented: the words are filtered using a specific lexicon and a vector is
computed by averaging the word-embeddings of the words that remain. Subsequently, each tweet-model is compared to each centroid using 
the cosine distance metric and is assigned to the centroid with the shortest distance, if this distance is smaller than 'distance_threshold'.

Params:
    - tweet_processed_texts : array of tweet texts, filtered out of stopwords but NOT stemmed
    - tweet_original_texts : array of original texts of tweets
    - tweet_dates : array of the date corresponding to each tweet
    - tweet_ids : array of the id corresponding to each tweet 
      (the position i of each of the first four parameters refer to the same tweet)
    - centroids : array of k centroids, where k is the number of news clusters
    - lexicon : specific lexicon used to filter the words of the tweets, if lexicon_filter is True
    - lexicon_filter : read comment above
    - distance_threshold : if the closest centroid to a tweet is at distance d and d > distance_threshold, then
      the tweet is discarded (not assigned to any news centroid)
    - scale_features : scale features of embeddings using statistics that are robust to outliers 
      (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler)
      
Returns:
    - tweet_ids_per_cluster : all the non-discarded tweet ids, grouped according to the cluster they've been assigned to 
    - tweet_dates_per_cluster : same as above
    - tweet_texts_per_cluster : same as above
    - tweet_vectors_per_cluster : same as above
    - tweet_distances_per_cluster : same as above
    - tweet_labels : array of the same length as tweet_processed_texts, where tweet_labels[i] contains the news cluster assigned to the tweet at position i;
      tweet_labels[i] == -1 means that the tweet at position i has been discarded (not assigned to any cluster)
    
    
Each of the first 5 parameters is a list of lists, where at position i you find the list of ids/dates/texts/vectors/distances assigned to cluster i.
"""
def assign_tweets_to_clusters(tweet_processed_texts, tweet_original_texts, tweet_dates, tweet_ids, centroids, 
                             lexicon, lexicon_filter=True, distance_threshold=0.5, scale_features=True):
               
    # build a word-embedding representation for each tweet
    tweet_vectors = []
    for t in tweet_processed_texts:
        tweet_embedding = np.zeros(300)
        n_words = 0
        for w in t.split():
            # if lexicon_filter is active, filter out the words not contained in the lexicon
            if ((not lexicon_filter or (lexicon_filter and pproc.stem(w) in lexicon)) 
                and (w in WORD_EMBEDDING_MODEL)):
                # sum up the word embeddings of the words that remain...
                tweet_embedding += WORD_EMBEDDING_MODEL[w]
                n_words += 1
        # ... and compute the average
        if n_words > 0:
            tweet_embedding = tweet_embedding / n_words
        tweet_vectors.append(tweet_embedding)
    
    tweet_vectors = np.array(tweet_vectors)
    if scale_features:
        tweet_vectors = RobustScaler().fit_transform(tweet_vectors)
    
    # assign each tweet to a cluster, if within distance_threshold
    tweets_per_cluster = [[] for c in range(len(centroids))]
    tweet_labels = []
    for i in range(len(tweet_vectors)):
        closest_centroid = 0
        shortest_dist = 1
        for c in range(len(centroids)):
            dist = cosine_distance(tweet_vectors[i], centroids[c])
            if dist < shortest_dist:
                closest_centroid = c
                shortest_dist = dist
        if shortest_dist < distance_threshold:
            tweets_per_cluster[closest_centroid].append((tweet_ids[i],tweet_vectors[i],tweet_dates[i],tweet_original_texts[i],shortest_dist))
            tweet_labels.append(closest_centroid)
        else:
            tweet_labels.append(-1)
    
    # group up tweet ids, vectors, dates, texts and distances to news centroid,
    # according to the cluster the tweets are assigned to.
    tweet_ids_per_cluster = [[] for c in range(len(centroids))]
    tweet_vectors_per_cluster = [[] for c in range(len(centroids))]
    tweet_dates_per_cluster = [[] for c in range(len(centroids))]
    tweet_texts_per_cluster = [[] for c in range(len(centroids))]
    tweet_distances_per_cluster = [[] for c in range(len(centroids))]
    for c in range(len(tweets_per_cluster)):
        tweets_per_cluster[c] = sorted(tweets_per_cluster[c], key = lambda x : x[-1])
        tweet_ids_per_cluster[c] = [tid for tid,tvec,tdat,ttext,dist in tweets_per_cluster[c]]
        tweet_vectors_per_cluster[c] = [tvec for tid,tvec,tdat,ttext,dist in tweets_per_cluster[c]]
        tweet_dates_per_cluster[c] = [tdat for tid,tvec,tdat,ttext,dist in tweets_per_cluster[c]]
        tweet_texts_per_cluster[c] = [ttext for tid,tvec,tdat,ttext,dist in tweets_per_cluster[c]]
        tweet_distances_per_cluster[c] = [dist for tid,tvec,tdat,ttext,dist in tweets_per_cluster[c]]
    
    return tweet_ids_per_cluster, tweet_dates_per_cluster, tweet_texts_per_cluster, tweet_vectors_per_cluster, tweet_distances_per_cluster, tweet_labels
    

"""
This is the main function of the script. It iterates through a period defined between 'min_date' and 'max_date' and, for each day, extracts a clustering
of the news published in a timespan tracing back to 'news_look_back', about the specified 'industry' and 'companies'. After applying outlier removal,
the function assigns the tweets posted in a timespan tracing back to 'tweets_look_back' to the most similar news cluster, 
if within a certain 'tweet_distance_threshold'. The function contains calls to'plot_clusters' and 'plot_statistics' to visualize the daily output.

Params:
    - industry : set this to 'SP500'; the industry we are interested in (e.g. 'Information Technology'). 
    - companies: set this to ['SP500']; the companies we are interested in 
      (for convenience, SP500 is trated as a company in this code and in the mongo database)
    - news_collection_name : name of the mongodb collection where the news relevant to 'industry' are stored
    - tweets_collection_name : name of the mongodb collection where the tweets relevant to 'industry' are stored
    - positive_lexicons : dict where the keys are dates of class datetime.date and format '%Y-%m-%d'; values are lists of terms
      having a positive impact on the market in some time interval preceding the key
    - negative_lexicons : same as above, for negative words
    - min_date : first date on which we want to apply clustering; however, the news will be collected starting from min_date - news_look_back
      and the tweets from min_date - tweets_look_back
    - max_date : last date on which we want to apply clustering
    - news_look_back : length of the time interval on which news are collected, for every day 
      (e.g. if current_date = 2018-01-10 and news_look_back = 7, the news to clusterize are collected from 2018-01-02 to 2018-01-09)
    - tweets_look_back : same as above, but for tweets
    - relevance_mode : if 'about', retrieves a document iff one of 'companies' is the main focus of the document
                       if 'relevant', retrieves a document if one of 'companies' is somehow relevant to the document
                       if 'both', retrieves a document in one of the two cases above
                       This parameter depends on DNA metadata; if interested in SP500, simply set this to 'both'.
    - cluster_algorithm : one of 'kmeans', 'kmedians', 'kmedoids', 'agglomerative', 'ensemble' (see function 'cluster_news' for reference)
    - centroid_type : one of 'mean', 'median' (see function 'compute_centroid' for reference)
    - lexicon_filter : if True, filter the words of the documents, keeping only those that appear in the positive or negative lexicon for the current day
    - scale_features : scale features of embeddings using statistics that are robust to outliers (you should set this to True)
      (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler)
    - outlier_method : metric used to the detect the outliers 
      (one of 'silhouette', 'proximity', 'silhouette+proximity'; see function 'find_outliers' for reference)
    - outlier_cutoff : percentile used to find a threshold below which a news is considered an outlier
      (see function 'find_outliers' for reference)
    - tweet_distance_threshold : threshold used to decide if a tweet should be discarded
      (see function 'assign_tweets_to_clusters'; the parameter is called simply 'distance_threshold')
    - n_top_words : how many words should be considered as relevant for each cluster, picking from the ranked list of words
    - test_keywords (optional) : if True, runs the function 'run_keywords_test' (see for reference)
    - use_tense_detection : if True, filters the sentences in each news (NOT tweets), keeping only those sentences that employ at least one
                            verb in the future tense
    - path_to_save : if not None, indicates the path to the folder where the daily output should be saved.
      The paths for each day will be created automatically, using this scheme: path_to_save/cluster_algorithm/date.
      So, for example, if you run 'detect_events' several times with different values of 'cluster_algorithm',
      but with the same 'path_to_save', you will obtain something like:
          path_to_save /
              kmeans /
                  2017-01-01
                  2017-01-02
                  2017-01-03
              agglomerative /
                  2017-01-01
                  2017-01-02
                  2017-01-03
    
Returns:
    - metrics_per_day : dict that stores the metrics obtained day by day; in particular, silhouette score, dunn index,
                        number of clusters and chosen_algorithm (if cluster_algorithm == 'ensemble').
                        the keys are dates of class datetime.date and format '%Y-%m-%d'; the values are dicts with
                        keys 'silhouette', 'dunn', 'n_clusters', %discarded_tweets and 'chosen_algorithm' (if cluster_algorithm == 'ensemble').
"""
def detect_events(industry, companies, news_collection_name, tweets_collection_name, positive_lexicons, negative_lexicons,
                  min_date='2009-01-01', max_date='2019-01-01', news_look_back=7, tweet_look_back=7, relevance_mode='both',
                  cluster_algorithm='kmeans', k_search_score='silhouette', n_clusters=None, centroid_type='mean',
                  lexicon_filter=True, scale_features=True, outlier_method='silhouette', outlier_cutoff=10, tweet_distance_threshold=0.5,
                  n_top_words=30, keywords_test=False, use_tense_detection=False, path_to_save=False):
    
    # in case we want to save the output to files, we create a folder named like 'cluster_algorithm',
    # that will contain all the output obtained with this configuration
    # see reference above about saving system
    if path_to_save:
        if cluster_algorithm not in os.listdir(path_to_save):
            os.mkdir(path_to_save + '/' + cluster_algorithm)
        path_to_save = path_to_save + '/' + cluster_algorithm
        
    min_date = datetime.strptime(min_date, '%Y-%m-%d').date()
    max_date = datetime.strptime(max_date, '%Y-%m-%d').date()
        
    # get all news relevant to the companies, in the time interval
    news_ids, news_dates = get_all_ids(companies=companies, min_date=min_date-timedelta(days=news_look_back), max_date=max_date, relevance_mode=relevance_mode, mongo_collection=news_collection_name)
    client = pymongo.MongoClient()
    all_news = client.financial_forecast.get_collection(news_collection_name).find({'an': {"$in": list(news_ids)}}).sort([('ingestion_datetime',1)])
    all_news = np.array(list(all_news))
    # organize the news in arrays (see organize_documents for reference)          
    news_processed_texts, news_associated_companies, news_dates, news_ids, news_titles, news_snippets = organize_documents(all_news, companies, industry=industry, relevance_mode=relevance_mode, use_tense_detection=use_tense_detection)
        
    # get all tweets relevant to the companies, in the time interval
    tweet_ids, tweet_dates = get_all_ids(companies=companies, min_date=min_date-timedelta(days=tweet_look_back), max_date=max_date, relevance_mode=relevance_mode, mongo_collection=tweets_collection_name)
    client = pymongo.MongoClient()
    all_tweets = client.financial_forecast.get_collection(tweets_collection_name).find({'an': {"$in": list(tweet_ids)}}).sort([('ingestion_datetime',1)])
    all_tweets = np.array(list(all_tweets))
    # organize the tweets in arrays (see organize_documents for reference) 
    # for tweets, the last return value of 'organize_documents' can be ignored; that's why it's called _xxx
    tweet_processed_texts, tweet_associated_companies, tweet_dates, tweet_ids, tweet_original_texts, _xxx = organize_documents(all_tweets, companies, industry=industry, relevance_mode=relevance_mode)

    # initialize the dict that will contain the daily metrics
    metrics_per_day = {}
    
    # initialize the dict that will contain the important words to train the CNN
    relevant_words_per_day = {}
    
    # initialize the dict that will contain all the words used on each day 
    # (the super-lexicon can be created as the union of all these lists)
    all_words_per_day = {}
    
    keyword_test_results = []
    
    # iterate over the time interval between min_date and max_date
    current_date = min_date
    start_index = 0
    while current_date <= max_date:
        
#        if current_date.strftime('%Y-%m-%d') not in TEST_KEYWORDS_DICT:
#            current_date = current_date + timedelta(days=1)
#            continue
        
        # find indexes i and j such that news_ids[i:j] are the news published in the time interval 
        # between (current_date - news_look_back) and current_date
        for i in range(start_index, len(news_processed_texts)):
            if news_dates[i] >= current_date - timedelta(days=news_look_back):
                start_index = i
                break
        for j in range(i, len(news_processed_texts)):
            if news_dates[j] >= current_date:
                break
        
        # we select from the right time interval, that we want to clusterize
        selected_news_processed_texts = news_processed_texts[i:j]
        selected_news_associated_companies = news_associated_companies[i:j]
        selected_news_dates = news_dates[i:j]
        selected_news_ids = news_ids[i:j]
        selected_news_titles = news_titles[i:j]
        selected_news_snippets = news_snippets[i:j]
        
        # in case we want to save the output to files, we create a subfolder for each current_date,
        # that will contain all the output obtained on that day; all the daily subfolder are saved
        # inside the folder named like 'cluster_algorithm'
        # see reference above about saving system
        if path_to_save:
            if current_date.strftime('%Y-%m-%d') == '2016-11-08':
                extra = ' (trumps election)'
            elif current_date.strftime('%Y-%m-%d') == '2016-06-23':
                extra = ' (brexit referendum)'
            else:
                extra = ''
            if current_date.strftime('%Y-%m-%d')+extra not in os.listdir(path_to_save):
                os.mkdir(path_to_save +'/' + current_date.strftime('%Y-%m-%d') + extra)
            full_path_to_save = path_to_save + '/' + current_date.strftime('%Y-%m-%d') + extra
        else:
            full_path_to_save = None
        
        # retrieve the lexicons for the current day
        current_pos_lexicon = {w:s for w,s in positive_lexicons[current_date]}
        current_neg_lexicon = {w:s for w,s in negative_lexicons[current_date]}
        print('\n\n********************************')
        print(current_date)
        
        metrics_per_day[current_date.strftime('%Y-%m-%d')] = {}
        
        # create a word-embedding representation for every news
        document_vectors = []           # list of the news-embeddings, obtained through the average of the words that each article contains,
                                        # after filtering it with the lexicon in case lexicon_filter == True
        vocabulary_for_tfidf = []       # whole set of words contained in the corpus, after applying lexicon filter
                                        # (it's a superset of document_words)
        
        for d,i in zip(selected_news_processed_texts, selected_news_ids):
            document_embedding = np.zeros(300)
            n_words = 0
            words = []
            for w in d.split():
                if ((not lexicon_filter or (lexicon_filter and pproc.stem(w) in (list(current_pos_lexicon.keys()) + list(current_neg_lexicon.keys())))) 
                    and (w in WORD_EMBEDDING_MODEL)):
                    document_embedding += WORD_EMBEDDING_MODEL[w]
                    vocabulary_for_tfidf.append(w)
                    words.append(w)
                    n_words += 1
            if n_words > 0:
                document_embedding = document_embedding / n_words
            
            document_vectors.append(document_embedding)
        
        document_vectors = np.array(document_vectors)
                    
        # scale features
        if scale_features:
            document_vectors = RobustScaler().fit_transform(document_vectors)
        vocabulary_for_tfidf = list(set(vocabulary_for_tfidf))
        
        # apply tf-idf on the whole current set of news that we want to clusterize 
        # (only in time interval between current_date - news_look_back and current_date)
        # This will be used later to extract the relevant words for each cluster
        vectorizer = TfidfVectorizer(vocabulary=vocabulary_for_tfidf)               #train tf-idf model using only the words that we kept after filtering
        tfidf_matrix = vectorizer.fit_transform(selected_news_processed_texts)      #tfidf_matrix[i] contains the tf-idf vector for selected_news_processed_texts[i] 
        tfidf_feature_names = np.array(vectorizer.get_feature_names())              #list of words used as features in the tf-idf model
                        
        # apply clustering to the selected set of news (see function 'cluster_news' for reference)
        labels, original_centroids, chosen_algorithm = cluster_news(document_vectors, current_date, cluster_alg=cluster_algorithm, 
                                                                    k_search_score=k_search_score, n_clusters=n_clusters, vocabulary=vocabulary_for_tfidf)

        # dbscan is the only algorithm that might return even 0 or 1 clusters
        # moreover, the outlier removal process is intrinsic to the algorithm itself
        if cluster_algorithm == 'dbscan':
            if len(original_centroids) == 0:
                print('DBSCAN has not detected any cluster in the data.')
                continue
            is_outlier = np.zeros(len(document_vectors))
            is_outlier[labels == -1] = 1        # for convenience, we want this equivalence always satisfied: is_outlier[i] == 1 <=> labels[i] == -1
            centroids = original_centroids
        else:
            # evaluate the quality of the cluster by silhouette and dunn index
            # please note that these metrics are valid only if number of clusters > 1
            silhouette_avg, samples_silhouette_values, silhouette_per_cluster = calc_silhouette(document_vectors, labels)
            dunn_index = calc_dunn(document_vectors, labels)
#            db_score = davies_bouldin_score(document_vectors, labels)   # still another clustering metric; you can ignore it
            
            print('\nNumber of clusters:', len(original_centroids))
#            print('Davies-Bouldin score (miniumum 0, the lower the better):', db_score)
            print('Dunn Index (the higher the better):', dunn_index)
            print('Average silhouette score (ranges from -1 to 1, the higher the better):', silhouette_avg)
            print('\nSILHOUETTE SCORE PER CLUSTER:')
            for c in range(len(silhouette_per_cluster)):
                print('Cluster #', c, ': ', silhouette_per_cluster[c], sep='')
                    
            # apply outlier removal, if desired
            # see function 'find_outliers' for reference
            if not outlier_method:
                is_outlier = np.zeros(len(document_vectors))
            elif outlier_method == 'silhouette':
                is_outlier = find_outliers(samples_silhouette_values, percentile_cutoff = outlier_cutoff)
            elif outlier_method == 'proximity':
                proximities = [(1 - cosine_distance(document_vectors[i], original_centroids[labels[i]])) for i in range(len(labels))]
                is_outlier = find_outliers(proximities, percentile_cutoff = outlier_cutoff)
            elif outlier_method == 'silhouette+proximity' or outlier_method == 'proximity+silhouette':
                proximities = [(1 - cosine_distance(document_vectors[i], original_centroids[labels[i]])) for i in range(len(labels))]
                is_outlier = find_outliers(samples_silhouette_values, scores_per_sample_2=proximities, percentile_cutoff = outlier_cutoff)
            
            # recompute the centroids after outlier removal
            if chosen_algorithm == 'kmeans':
                c_type = 'mean'
            elif chosen_algorithm in ('kmedians', 'kmedoids'):
                c_type = 'median'
            else:
                c_type = centroid_type
            centroids = compute_centroids(document_vectors[is_outlier == 0], labels[is_outlier == 0], method=c_type)
            
        # this check is needed because, in rare extreme cases, after outlier removal all news might be marked as outliers and we might remain with 0 clusters
        if len(np.unique(labels[labels != -1])) > 1:
            # evaluate the quality of the clustering after outlier removal
            silhouette_avg, samples_silhouette_values, silhouette_per_cluster = calc_silhouette(document_vectors[is_outlier == 0], 
                                                                                                labels[is_outlier == 0])
            dunn_index = calc_dunn(document_vectors[is_outlier == 0], labels[is_outlier == 0])
#            db_score = davies_bouldin_score(document_vectors[is_outlier == 0], labels[is_outlier == 0])
            
            print('\nAFTER DENOISING:')
#            print('Davies-Bouldin score (miniumum 0, the lower the better):', db_score)
            print('Dunn Index (the higher the better):', dunn_index)
            print('Average silhouette score (ranges from -1 to 1, the higher the better):', silhouette_avg)
            print('\nSILHOUETTE SCORE PER CLUSTER:')
            for c in range(len(silhouette_per_cluster)):
                print('Cluster #', c, ': ', silhouette_per_cluster[c], sep='')
        else:
            print('\nOnly one cluster has been detected. For this reason, cluster quality metrics do not apply.')
            silhouette_avg, samples_silhouette_values, silhouette_per_cluster, dunn_index = None, None, None, None
        
        # prepare the vectors for 2D visualization of the clusters; this is done by TSNE algorithm, that allows to reduce the dimensionality of a set of vectors
        # (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)
        # The input to TSNE will be the concatanation of:
        #    - the word-embeddings of all the words in the current set of news, after lexicon filter
        #    - the news-embeddings
        #    - the centroids of the original clustering
        #    - the centroids re-computed after outlier removal
        # It is useful to add the word-embeddings of the individual words because, even though we will never plot them,
        # they make the TSNE model more stable and improve the quality of the 2D aproximation.
        # See function 'plot_clusters' for further reference.
        if path_to_save:
            vocabulary_word_embeddings = np.array([WORD_EMBEDDING_MODEL[w] for w in vocabulary_for_tfidf])
            all_centroids = np.append(original_centroids, centroids, axis=0)
            ndimensional_vectors = np.append(document_vectors, all_centroids, axis=0)
            ndimensional_vectors = np.append(vocabulary_word_embeddings, ndimensional_vectors, axis = 0)
            vectors_2D = TSNE(n_components=2, metric='cosine', perplexity=30, early_exaggeration=3, random_state=0).fit_transform(ndimensional_vectors)
            document_vectors_2D = vectors_2D[len(vocabulary_word_embeddings):len(vocabulary_word_embeddings)+len(document_vectors)]
            old_centroids_2D = vectors_2D[len(vocabulary_word_embeddings)+len(document_vectors):-len(centroids)]
            new_centroids_2D = vectors_2D[-len(centroids):]
            if outlier_method and cluster_algorithm != 'dbscan':
                plot_clusters(document_vectors_2D, old_centroids_2D, labels, path_to_save=full_path_to_save, chosen_algorithm=chosen_algorithm)
            plot_clusters(document_vectors_2D, new_centroids_2D, labels, is_outlier=is_outlier, plot_outliers=False, path_to_save=full_path_to_save, chosen_algorithm=chosen_algorithm)
            
            
        # update the dict with the current metrics
        metrics_per_day[current_date.strftime('%Y-%m-%d')]['n_clusters'] = len(centroids)
        if cluster_algorithm == 'ensemble':
            metrics_per_day[current_date.strftime('%Y-%m-%d')]['chosen_algorithm'] = chosen_algorithm
        if silhouette_avg:
            metrics_per_day[current_date.strftime('%Y-%m-%d')]['silhouette'] = round(silhouette_avg, 2)
            metrics_per_day[current_date.strftime('%Y-%m-%d')]['dunn'] = round(dunn_index, 2)
        else:
            metrics_per_day[current_date.strftime('%Y-%m-%d')]['silhouette'] = 'does not apply'
            metrics_per_day[current_date.strftime('%Y-%m-%d')]['dunn'] = 'does not apply'
        
        # for convenience, we want this equivalence always satisfied: is_outlier[i] == 1 <=> labels[i] == -1
        labels[is_outlier == 1] = -1  
        
        # in case some cluster was supressed after outlier removal, we need to rename the labels
        # see the function for reference
        labels = rename_labels(labels)
            
        # collect a set of properties (e.g. articles, relevant companies, relevant words) for every cluster
        # for every property, we have a list of lists, in which at position i the information about cluster i is stored
        cluster_articles = [[] for c in range(len(centroids))]
        cluster_companies = [[] for c in range(len(centroids))]
        cluster_stemmed_documents = [[] for c in range(len(centroids))]
        # we will compute the average td-idf vector for each cluster, by averaging the tf-idf vectors of all the news of the cluster
        cluster_tfidf = [np.zeros(len(tfidf_feature_names)) for c in range(len(centroids))] 
        discarded_news_dates = []   # useful to keep track of how many news we discard on each date (outliers)
        for i in range(len(labels)):
            if is_outlier[i] == 0:
                distance_from_centroid = cosine_distance(document_vectors[i], centroids[labels[i]])
                # for every article, we store the DNA-id of the news, the date, the title, the snippet and the distance from the centroid it belongs to
                cluster_articles[labels[i]].append((selected_news_ids[i], selected_news_dates[i], selected_news_titles[i], selected_news_snippets[i], distance_from_centroid))
                cluster_stemmed_documents[labels[i]].append(pproc.stem(selected_news_processed_texts[i]))
                cluster_companies[labels[i]].extend(selected_news_associated_companies[i])
                cluster_tfidf[labels[i]] += tfidf_matrix[i]
            else:
                discarded_news_dates.append(selected_news_dates[i])
        
        # how many news have been discarded on each date?
        discarded_news_per_date = [discarded_news_dates.count(d) for d in [current_date - timedelta(days=x) for x in range(news_look_back,0,-1)]]
                
        n_news_per_cluster = [0 for c in range(len(centroids))]
        n_news_per_cluster_per_date = [[] for c in range(len(centroids))]
        cluster_relevant_words_tfidf = [[] for c in range(len(centroids))]
        # how many words do the documents of each cluster share with the positive and negative lexicon on the current date?
        pos_lexicon_match_per_cluster = [0 for c in range(len(centroids))]
        neg_lexicon_match_per_cluster = [0 for c in range(len(centroids))]      
        for c in range(len(centroids)):
            pos_lexicon_match_per_cluster[c] = calc_lexicon_match(cluster_stemmed_documents[c], list(current_pos_lexicon.keys()))
            neg_lexicon_match_per_cluster[c] = calc_lexicon_match(cluster_stemmed_documents[c], list(current_neg_lexicon.keys()))
            
            # compute the average td-idf vector for each cluster, by averaging the tf-idf vectors of all the news of the cluster
            cluster_tfidf[c] = cluster_tfidf[c] / len(cluster_articles[c])
            cluster_tfidf[c] = np.asarray(cluster_tfidf[c]).flatten()
            sorted_indices = np.asarray(cluster_tfidf[c].argsort()).flatten()
            # words sorted in descending order, according to their relevance to the cluster
            cluster_relevant_words_tfidf[c] = [(tfidf_feature_names[index], round(cluster_tfidf[c][index],2)) for index in reversed(sorted_indices) if not np.isnan(cluster_tfidf[c][index])]
            
            cluster_companies[c] = sorted([(comp,round(cluster_companies[c].count(comp)/len(cluster_articles[c]),2)) for comp in set(cluster_companies[c])], key = lambda x: x[1], reverse=True)
            
            cluster_articles[c] = sorted(cluster_articles[c], key = lambda x : x[-1])
            n_news_per_cluster[c] = len(cluster_articles[c])
            n_news_per_cluster_per_date[c] = [[date for nid,date,tit,snip,dist in cluster_articles[c]].count(d) for d in [current_date - timedelta(days=x) for x in range(news_look_back,0,-1)]]
        
        
        metrics_per_day[current_date.strftime('%Y-%m-%d')]['relevant_words_overlap'] = calc_relevant_words_overlap(cluster_relevant_words_tfidf, n_top_words=30)
        
        
        # compute the total number of news published on each date in the current time interval, without considering the cluster they belong to
        # and including also the discarded ones
        date_range = [current_date - timedelta(days=x) for x in range(news_look_back,0,-1)]
        total_news_per_date = []
        for i in range(len(date_range)):
            total_news_per_date.append(sum([n_news_per_cluster_per_date[c][i] for c in range(len(n_news_per_cluster_per_date))]) + discarded_news_per_date[i])
        
        # compute the percentage of news associated to each cluster on each day, over the total 
        perc_news_per_cluster_per_date = [[] for c in range(len(n_news_per_cluster_per_date))]
        for c in range(len(n_news_per_cluster_per_date)):
            perc_news_per_cluster_per_date[c] = [100 * (n_news_per_cluster_per_date[c][i] / total_news_per_date[i]) if total_news_per_date[i] > 0 else 0 for i in range(len(total_news_per_date))]
            
        # compute the percentage of discarded news on each day, over total
        perc_discarded_news_per_date = []
        for i in range(len(date_range)):
            if total_news_per_date[i] > 0:
                perc_discarded_news_per_date.append(100* (discarded_news_per_date[i] / total_news_per_date[i]))
            else:
                perc_discarded_news_per_date.append(0)
    
    
        # select tweets posted in the time interval between (current_date - tweets_look_back) and current_date,
        # as we did previously with news (see above for reference)
        for h in range(len(tweet_processed_texts)):
            if tweet_dates[h] >= current_date - timedelta(days=tweet_look_back):
                break
        for k in range(h, len(tweet_processed_texts)):
            if tweet_dates[k] >= current_date:
                break
        
        selected_tweets_processed_texts = tweet_processed_texts[h:k]
        selected_tweets_ids = tweet_ids[h:k]
        selected_tweets_dates = tweet_dates[h:k]
        selected_tweets_original_texts = tweet_original_texts[h:k]
        
        # we use the same tf-idf-based method on tweet corpus to compute the importance of each
        # word to each group of tweets (after we will have assigned them to news clusters)
        tweets_vectorizer = TfidfVectorizer(vocabulary=vocabulary_for_tfidf)
        tweets_tfidf_matrix = tweets_vectorizer.fit_transform(selected_tweets_processed_texts)
        tweets_tfidf_feature_names = np.array(tweets_vectorizer.get_feature_names())
        
        # see function for reference
        (tweet_ids_per_cluster, tweet_dates_per_cluster, tweet_texts_per_cluster,
         tweet_vectors_per_cluster, tweet_distances_per_cluster, tweet_labels) = assign_tweets_to_clusters(selected_tweets_processed_texts, selected_tweets_original_texts, 
                                                                                                           selected_tweets_dates, selected_tweets_ids, centroids,
                                                                                                           lexicon = list(current_neg_lexicon.keys()) + list(current_pos_lexicon.keys()), 
                                                                                                           lexicon_filter=lexicon_filter,
                                                                                                           distance_threshold=tweet_distance_threshold, 
                                                                                                           scale_features=scale_features)
        
        # the following statistics and properties are collected and computed in the same way as we did for news
        # see above for reference
        n_tweets_per_cluster = [0 for c in range(len(centroids))]
        n_tweets_per_cluster_per_date = [[] for c in range(len(centroids))]
        for c in range(len(tweet_ids_per_cluster)):
            n_tweets_per_cluster[c] = len(tweet_ids_per_cluster[c])
            n_tweets_per_cluster_per_date[c] = [tweet_dates_per_cluster[c].count(d) for d in [current_date - timedelta(days=x) for x in range(tweet_look_back,0,-1)]]
        
        discarded_tweets_dates = [date for date,label in zip(selected_tweets_dates, tweet_labels) if label == -1]
        discarded_tweets_per_date = [discarded_tweets_dates.count(d) for d in [current_date - timedelta(days=x) for x in range(tweet_look_back,0,-1)]]
        
        total_tweets_per_date = []
        for i in range(len(date_range)):
            total_tweets_per_date.append(sum([n_tweets_per_cluster_per_date[c][i] for c in range(len(n_tweets_per_cluster_per_date))]) + discarded_tweets_per_date[i])
        
        perc_tweets_per_cluster_per_date = [[] for c in range(len(n_tweets_per_cluster_per_date))]
        for c in range(len(n_tweets_per_cluster_per_date)):
            perc_tweets_per_cluster_per_date[c] = [100 * (n_tweets_per_cluster_per_date[c][i] / total_tweets_per_date[i]) if total_tweets_per_date[i] > 0 else 0 for i in range(len(total_tweets_per_date))]
        
        perc_discarded_tweets_per_date = []
        for i in range(len(date_range)):
            if total_tweets_per_date[i] > 0:
                perc_discarded_tweets_per_date.append(100* (discarded_tweets_per_date[i] / total_tweets_per_date[i]))
            else:
                perc_discarded_tweets_per_date.append(0)
        
        metrics_per_day[current_date.strftime('%Y-%m-%d')]['%assigned_tweets'] = 100 - perc_discarded_tweets_per_date[-1]
        
        tweets_cluster_tfidf = [np.zeros(len(tweets_tfidf_feature_names)) for c in range(len(centroids))]
        for i in range(len(tweet_labels)):
            if tweet_labels[i] != -1:
                tweets_cluster_tfidf[tweet_labels[i]] += tweets_tfidf_matrix[i]
        
        tweets_cluster_relevant_words_tfidf = [[] for c in range(len(centroids))]
        for c in range(len(centroids)):
            if len(tweet_ids_per_cluster[c]) > 0:
                tweets_cluster_tfidf[c] = tweets_cluster_tfidf[c] / len(tweet_ids_per_cluster[c])
                tweets_cluster_tfidf[c] = np.asarray(tweets_cluster_tfidf[c]).flatten()
                sorted_indices = np.asarray(tweets_cluster_tfidf[c].argsort()).flatten()
                tweets_cluster_relevant_words_tfidf[c] = [(tweets_tfidf_feature_names[index], round(tweets_cluster_tfidf[c][index],2)) for index in reversed(sorted_indices) if not np.isnan(tweets_cluster_tfidf[c][index])]

        
        # check how large is the intersection between the list of top-30 relevant words for a cluster
        # and the top-30 list for the tweets assigned to the same cluster.
        # This should give us a measure of quality of the assignment procedure; low values should indicate noise in the tweets
        # and poor match between news and tweets
        relevant_words_match = [[] for c in range(len(centroids))]
        for c in range(len(centroids)):
            match = 0
            for w,s in tweets_cluster_relevant_words_tfidf[c][:n_top_words]:
                if pproc.stem(w) in [pproc.stem(v) for v,t in cluster_relevant_words_tfidf[c]][:n_top_words]:
                    match += 1
            match = 100 * (match / n_top_words)
            relevant_words_match[c] = match

        # run keywords test (see function for reference)
        if keywords_test and current_date.strftime('%Y-%m-%d') in TEST_KEYWORDS_DICT:
            keywords_in_relevant_words_tfidf, keywords_in_titles, keywords_in_snippets = run_keywords_test(cluster_relevant_words_tfidf, cluster_articles, TEST_KEYWORDS_DICT[current_date.strftime('%Y-%m-%d')], n_top_words=n_top_words)
            keyword_test_results.append(keywords_in_relevant_words_tfidf)
        else:
            keywords_in_relevant_words_tfidf, keywords_in_titles, keywords_in_snippets = [], [], []
                
        # save all the properties of the clusters to the daily folder
        if path_to_save:
            with open(full_path_to_save + '/4. NEWS titles per cluster.txt', 'w') as writer:
                for c in range(len(cluster_articles)):
                    writer.write('\n\n\n*** Cluster #' + str(c) + ' (' + str(len(cluster_articles[c])) + ' news items) ***\n\n')
                    for i,date,title,snippet,dist in cluster_articles[c]:
                        writer.write(title + '\n' + str(date) + '\tDist. from centroid: ' + str(dist) + '\n\n')
            
            with open(full_path_to_save + '/7. NEWS top '+ str(n_top_words) +' words per cluster (tf-idf scores on news).txt', 'w') as writer:
                for c in range(len(cluster_relevant_words_tfidf)):
                    writer.write('\n\n\n*** Cluster #' + str(c) + '***\n')
                    for w,s in cluster_relevant_words_tfidf[c][:n_top_words]:
                        writer.write(w + '\t' + str(s) + '\n')
                
            with open(full_path_to_save + '/10. attached tweet per cluster.txt', 'w', encoding='utf-8') as writer:
                for c in range(len(tweet_texts_per_cluster)):
                    writer.write('\n\n\n*** Cluster #' + str(c) + ' (' + str(len(tweet_texts_per_cluster[c])) + ' tweets) ***\n\n')
                    for text, date, dist in zip(tweet_texts_per_cluster[c], tweet_dates_per_cluster[c], tweet_distances_per_cluster[c]):
                        try:
                            writer.write(text + '\n' + str(date) + '\tDist. from centroid: ' + str(dist) + '\n\n')
                        except UnicodeEncodeError:
                            continue
            
            with open(full_path_to_save + '/11. tweet top '+str(n_top_words)+' words per cluster (tf-idf scores on tweets).txt', 'w') as writer:
                for c in range(len(tweets_cluster_relevant_words_tfidf)):
                    writer.write('\n\n\n*** Cluster #' + str(c) + '***\n')
                    for w,s in tweets_cluster_relevant_words_tfidf[c][:n_top_words]:
                        writer.write(w + '\t' + str(s) + '\n')
             
            with open(path_to_save + '/_parameters.txt', 'w') as w:
                w.write('\nAlgorithm: ' + cluster_algorithm)
                if k_search_score:
                    w.write('\nK-search method: ' + k_search_score)
                if n_clusters:
                    w.write('\nN. clusters: ' + str(n_clusters))
                if cluster_algorithm in ('dbscan', 'agglomerative'):
                    w.write('\nCentroid type: ' + centroid_type)
                if cluster_algorithm != 'dbscan':
                    w.write('\nOutlier method: ' + str(outlier_method))
                    w.write('\nOutlier cutoff: ' + str(outlier_cutoff))
                w.write('\nNews look back: ' + str(news_look_back))
                w.write('\nTweets look back: ' + str(tweet_look_back))
                w.write('\nLexicon filter: ' + str(lexicon_filter))       
                
            plot_statistics(n_news_per_cluster, n_news_per_cluster_per_date, perc_news_per_cluster_per_date, discarded_news_per_date, perc_discarded_news_per_date, total_news_per_date,
                            n_tweets_per_cluster, n_tweets_per_cluster_per_date, perc_tweets_per_cluster_per_date, discarded_tweets_per_date, perc_discarded_tweets_per_date, total_tweets_per_date,
                            pos_lexicon_match_per_cluster, neg_lexicon_match_per_cluster, silhouette_per_cluster, silhouette_avg, dunn_index, 
                            cluster_companies, cluster_relevant_words_tfidf,
                            keywords_in_relevant_words=keywords_in_relevant_words_tfidf, keywords_in_titles=keywords_in_titles, 
                            keywords_in_snippets=keywords_in_snippets, relevant_words_match=relevant_words_match,
                            date_range=[current_date - timedelta(days=x) for x in range(news_look_back,0,-1)],
                            path_to_save=full_path_to_save)
        
        # we save all the words used on the current date, filtered with the lexicon
        all_words_per_day[current_date] = vocabulary_for_tfidf
        
        # we extract the words from each cluster 
        relevant_words_per_day[current_date] = {}
        for c in range(len(centroids)):
            # a cluster gives contribution only if it has tweets assigned in the last day and (=> the event is hot)
            if perc_tweets_per_cluster_per_date[c][-1] > 0 :
                for w,s in cluster_relevant_words_tfidf[c][:n_top_words]:
                    # every word is assigned 3 different values:
                    #   - the relevance to the cluster it belongs to, calculated through tf-idf
                    #   - the percentage of tweets assigned to the cluster it belongs to
                    #   - the score in the specialized lexicon
                    stemmed_w = pproc.stem(w)
                    relevant_words_per_day[current_date][w] = {'relevance_to_cluster': s,
                                                               'tweets_percentage' : perc_tweets_per_cluster_per_date[c][-1],
                                                                'lexicon_score' : current_pos_lexicon[stemmed_w] if stemmed_w in current_pos_lexicon else (current_neg_lexicon[stemmed_w] if stemmed_w in current_neg_lexicon else 0)}
            
        current_date = current_date + timedelta(days=1)
    
    if keywords_test:
        print('\n\nAverage match in keyword test:', np.average(keyword_test_results))
        
    return metrics_per_day, relevant_words_per_day, all_words_per_day


if __name__ == "__main__":
    
    save_global_output = True         #True if you want to save the comparison boxplots and the tweet plot
    save_daily_output = False          #True if you want to save the plots for every single day
    save_words_for_CNN = False
    
    # algorithms that you want to compare;
    # NOTE: this does not affect the list of algorithms that are used within the ensemble strategy.
    algorithms_list = ['agglomerative']#, 'kmeans', 'kmedians', 'kmedoids']   
    
    # you can set this variable to one of these values: None, 'brexit, 'trump'
    # see below for reference
    event = None
    alert_threshold = 3
    
    industry = 'SP500'      #possible values are 'Financial', 'Industrials' and 'Information Technology', but no tweets are available for these industries yet
    companies=['SP500']     #if industry != 'SP500', use companies = get_companies_by_industry(industry)
    news_collection_name = 'sp500_news_2009-2020'      #change the collections name depending on the industry
    tweets_collection_name = 'sp500_tweets_2009-2020'
    
    # if you set event to 'brexit' or 'trump', the algorithm will run on those specific time intervals and will run the keywords_test with already defined keywords;
    # please modify the paths to your preference
    if event == 'brexit':
        min_date='2016-06-15'
        max_date='2016-07-30'
        test_keywords = ['brexit', 'uk', 'british', 'referendum', 'poll', 'stay', 'leave', 'eu', 'vote', 'remain']
        path_to_save = 'output_clustering/brexit'
    elif event == 'trump':
        min_date='2016-11-01'
        max_date='2016-11-15'
        test_keywords = ['trump','donald','clinton','hillary','vote','votes','election','elections','poll','polls','president']
        path_to_save = 'output_clustering/trump'
    elif event == 'coronavirus':
        min_date='2020-01-20'
        max_date='2020-02-05'
        test_keywords = ['coronavirus','virus','chinese','china','outbreak','death','infect','loss','spread','case']
        path_to_save = 'output_clustering/coronavirus'
    elif event == 'tradewar':
        min_date='2019-05-05'
        max_date='2019-05-20'
        test_keywords = ['china','us','trump','tariff','trade','war','deal','market','industry','beijing']
        path_to_save = 'output_clustering/tradewar'
        
    # if None, you should modify min_date and max_date to your preference; no keywords test will be executed;
    # a folder with the datetime of the moment of the execution will be created automatically inside output_clustering/prove
    # please modify the paths to your preference
    elif not event:
        min_date='2016-06-15'
        max_date='2016-06-30'
        test_keywords = []
        if save_global_output:
            folder_name = 'output_clustering/prove/'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            os.mkdir(folder_name)
            path_to_save = folder_name
    
    if not save_global_output:
        path_to_save = None
        
    
    # compute the ground-truth for the alert-generation evaluation
    weekly_deltas_per_date = get_deltas_per_date(min_date=min_date, max_date=max_date, delta_timespan=7)
    
    # Retrieve the lexicons with the desired parameters.
    # If you want to create a new set of lexicons with different parameters, substitute this instruction
    # with a call to create_lexicons.
    # See the script 'create_lexicons.py' for reference
    pos_lexicons, neg_lexicons = fetch_lexicons(industry=industry, collection_name=news_collection_name,
                                                min_date=min_date, max_date=max_date,
                                                look_back=28, ngram_range=(1,1), max_df=0.9, min_df=10,
                                                positive_percentile_range=(80,100), negative_percentile_range=(0,20))
    
    # keep track of the metrics returned by every algorithm in 'algorithm_list', for each day
    global_metrics_per_day = {}
    silhouette_scores_per_alg = {a:[] for a in algorithms_list}
    dunn_scores_per_alg = {a:[] for a in algorithms_list}
    relevant_words_overlap_per_alg = {a:[] for a in algorithms_list}
    n_clusters_per_alg = {a:[] for a in algorithms_list}
    assigned_tweets_per_day_per_alg = {a:[] for a in algorithms_list}
    chosen_algorithm_count = {'kmeans':0, 'kmedians':0, 'agglomerative':0}  #used only for ensemble, to count how many time each algorithm was selected
    
    # 'detect_events' is executed multiple times on the same time interval, changing only the clustering algorithm.
    # NOTE: global variable INITIAL_CENTROIDS ensures that, on a given day, the output of the kmeans component of ensemble (when cluster_algorithm == 'ensemble)
    # will be the same as the one obtained on the execution of 'detect_events' in which cluster_algorithm == 'kmeans'
    for ca in algorithms_list:
        metrics, relevant_words, all_words = detect_events(industry=industry, companies=companies, news_collection_name=news_collection_name, tweets_collection_name=tweets_collection_name,
                                                           positive_lexicons=pos_lexicons, negative_lexicons=neg_lexicons,
                                                           min_date=min_date, max_date=max_date, news_look_back=7, tweet_look_back=7, relevance_mode='both', scale_features=True,
                                                           cluster_algorithm=ca, n_clusters=None, k_search_score='silhouette', centroid_type='median',
                                                           outlier_cutoff=25, outlier_method='silhouette+proximity', use_tense_detection=False,
                                                           lexicon_filter=True, tweet_distance_threshold=0.5, keywords_test=False,
                                                           path_to_save=path_to_save if save_daily_output else None)
        
        if save_words_for_CNN:
            with open('../words_for_CNN/'+ca+'_from_'+min_date+'_to_'+max_date+'.csv', 'w') as word_writer:
                for date in relevant_words:
                    word_writer.write(date.strftime('%Y-%m-%d')+'\t'+str(relevant_words[date])+'\n')
            with open('../words_for_super_lexicon/from_'+min_date+'_to_'+max_date+'.csv', 'w') as word_writer:
                for date in all_words:
                    word_writer.write(date.strftime('%Y-%m-%d')+'\t'+str(all_words[date])+'\n')
                
        for date in metrics:
            if date not in global_metrics_per_day:
                global_metrics_per_day[date] = {}
            global_metrics_per_day[date][ca] = metrics[date]
            silhouette_scores_per_alg[ca].append(metrics[date]['silhouette'])
            dunn_scores_per_alg[ca].append(metrics[date]['dunn'])
            relevant_words_overlap_per_alg[ca].append(metrics[date]['relevant_words_overlap'])
            n_clusters_per_alg[ca].append(metrics[date]['n_clusters'])
            assigned_tweets_per_day_per_alg[ca].append(metrics[date]['%assigned_tweets'])
            if ca == 'ensemble':
                chosen_algorithm_count[metrics[date]['chosen_algorithm']] += 1
            
    if save_global_output:
        
        # print the plot of discarded tweets on each day, highlighting only the dates where the rate is below 97% (hardcoded below)
        min_date = datetime.strptime(min_date, '%Y-%m-%d').date()
        max_date = datetime.strptime(max_date, '%Y-%m-%d').date()
        for alg in algorithms_list:
            
            if alg not in os.listdir(path_to_save):
                os.mkdir(path_to_save + '/' + alg)
            
            labels_for_assigned = []
            assigned_tweets_per_date = {}
#            os.mkdir(path_to_save+'/'+alg+'/price_plots_on_events')
            with open(path_to_save+'/'+alg+'/percentage of assigned TWEETS per day, detailed.csv', 'w') as writer:
                for td, atweets in zip(range((max_date - min_date).days+1), assigned_tweets_per_day_per_alg[alg]):
                    next_date = min_date + timedelta(days=td)
                    writer.write(next_date.strftime('%Y-%m-%d') + ',' + str(atweets) + '\n')
                    assigned_tweets_per_date[next_date] = atweets
                    if atweets >= alert_threshold:
                        labels_for_assigned.append(next_date)
#                        plot_price_chart('SP500', min_date=next_date-timedelta(days=14), max_date=next_date+timedelta(days=14), main_date=next_date.strftime('%Y-%m-%d'), path_to_save=path_to_save+'/'+alg+'/price_plots_on_events')
                    else:
                        labels_for_assigned.append('')
            fig, ax = plt.subplots(figsize = (24, 10))
            x = np.arange(len(assigned_tweets_per_day_per_alg[alg]))
            ax.plot(x, assigned_tweets_per_day_per_alg[alg], color='black')       
            ax.set_ylabel('% of assigned tweets', fontsize=12)
            ax.set_title('Percentage of assigned tweets per day (% over total daily tweets)', fontsize=14)
            ax.set_xticks([])
#            ax.set_xticklabels([min_date + timedelta(x) for x in range((max_date - min_date).days)])
            plt.axhline(y=3, color='red', label='alert-threshold')
            for i in range(len(labels_for_assigned)):
                ax.annotate(str(labels_for_assigned[i]), (x[i], assigned_tweets_per_day_per_alg[alg][i]), color='red', fontsize=10)
            plt.legend()
            plt.savefig(path_to_save+'/'+alg+'/percentage of assigned TWEETS per day.png')
            
            evaluate_alerts(weekly_deltas_per_date, assigned_tweets_per_date)
            
        # print boxplots to compare the metrics obtained by each algorithm
        # see function 'print_boxplot' for reference
        print_boxplot(silhouette_scores_per_alg, algorithms_list, 'Silhouette scores', path_to_save=path_to_save + '/comparison silhouette scores.png')
        print_boxplot(dunn_scores_per_alg, algorithms_list, 'Dunn scores', path_to_save=path_to_save + '/comparison dunn scores.png')
        print_boxplot(relevant_words_overlap_per_alg, algorithms_list, 'Relevant words overlap', path_to_save=path_to_save + '/comparison relevant words overlap.png')
        print_boxplot(n_clusters_per_alg, algorithms_list, 'Number of clusters', path_to_save=path_to_save + '/comparison number of clusters.png')
    
        # print a bar plot with the count of the times each algorithm was selected in ensemble
        if 'ensemble' in algorithms_list:
            print('\nChosen algorithm count:')
            counts = []
            algs = []
            for alg in chosen_algorithm_count:
                print(alg, ':', chosen_algorithm_count[alg])
                algs.append(alg)
                counts.append(chosen_algorithm_count[alg])
                
            x = np.arange(3)
            fig, ax = plt.subplots()
            ax.bar(x, counts, width=0.35, color=ALL_COLORS[-4:])
            ax.set_title('Count of selected clustering algorithms in ensemble')
            ax.set_xticks(x)
            ax.set_xticklabels(algs)
            plt.savefig(path_to_save + '/algorithm_count_in_ensemble.png')
        
        # print the dict of all the metrics as an html table 
        pp = pprint.PrettyPrinter(indent=4)
#        pp.pprint(global_metrics_per_day)
        build_direction = "LEFT_TO_RIGHT"
        table_attributes = {"style" : "width:100%", "border": "1px solid black", "border-collapse": "collapse", "text-align": "center"}
        html = convert(global_metrics_per_day, build_direction=build_direction, table_attributes=table_attributes)
        
        html = """<p>NOTA BENE: Alcune apparenti incongruenze nella tabella sono dovute al fatto che, mentre
                     l'algoritmo di ensemble seleziona il clustering migliore <b>prima</b> dell'outlier removal,
                     nella tabella sono mostrati le silhouette e i dunn <b>dopo</b> l'outlier removal.
                  </p><br>""" + html + """<ul>
                                             <li>Silhouette score: ranges from -1 (worst) to 1 (best)</li>
                                             <li>Dunn index: ranges from 0 (worst) to 1 (best)</li>
                                         </ul>"""
        
        
        
        with open(path_to_save+ '/results_by_day.html', 'w') as table_writer:
            table_writer.write(html)
