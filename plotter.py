# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas
import numpy as np

# global variables used to pick the colors for the plots
ALL_COLORS = ['red', 'green', 'blue', 'gold', 'magenta', 'cyan', 'gray', 'orange', 'teal', 'pink', 'purple']

"""
Utility function that plots the close price for the company passed as parameter in the defined time period.
"""
def plot_price_chart(company, min_date='2016-06-18', max_date='2016-06-29', main_date=None, path_to_save=None):

    
    if type(min_date) == type('str'):
        min_date = datetime.strptime(min_date, '%Y-%m-%d').date()
        max_date = datetime.strptime(max_date, '%Y-%m-%d').date()
        
        
    prices = pandas.read_csv('../SP500_market_data/'+company+'.csv')
    prices['Date'] = [datetime.strptime(d, '%Y-%m-%d').date() for d in prices['Date']]
    
    dates = prices['Date'].tolist()
    closes = prices['Adj Close'].tolist()
    
    for i in range(len(dates)):
        if dates[i] >= min_date:
            break
    for j in range(i,len(dates)):
        if dates[j] >= max_date:
            break
    
    fig, ax = plt.subplots(figsize=(28,10))
    ax.set_ylabel('close price')
    ax.set_title('Close prices for '+str(company)) 
    ax.plot(dates[i:j], closes[i:j], linestyle='-', marker='o')
    ax.set_xticks(dates[i:j])
    ax.set_xticklabels(dates[i:j])
    ax.vlines(dates[i:j], ymin=min(closes[i:j]), ymax=max(closes[i:j]), linestyle='--', color='gray')
    for i in range(i,j):
        delta = round(100*((closes[i]-closes[i-1])/closes[i-1]), 2)
        ax.annotate(str(delta)+'%', (dates[i], closes[i]), color='g' if delta >= 0 else 'r', fontsize=14)
    if path_to_save:
        plt.savefig(path_to_save+'/'+main_date+'_price_plot.png')
    else:
        plt.show()
        
        
"""
Prints or saves (if path_to_save' is not None) the charts of metrics obtained by 'detect_events' (see this function for reference on the parameters).
"""
def plot_statistics(n_news_per_cluster, n_news_per_cluster_per_date, perc_news_per_cluster_per_date, discarded_news_per_date, perc_discarded_news_per_date, total_news_per_date,
                    n_tweets_per_cluster, n_tweets_per_cluster_per_date, perc_tweets_per_cluster_per_date, discarded_tweets_per_date, perc_discarded_tweets_per_date, total_tweets_per_date,
                    pos_lexicon_match_per_cluster, neg_lexicon_match_per_cluster, silhouette_per_cluster, silhouette_avg, dunn_index, 
                    cluster_companies, cluster_relevant_words, vocabulary_tsne, 
                    keywords_in_relevant_words=[], keywords_in_snippets=[], keywords_in_titles=[], relevant_words_match=[], date_range=[],
                    path_to_save=None):
    
    x = np.arange(len(n_news_per_cluster))
    fig, ax = plt.subplots()
    ax.bar(x, n_news_per_cluster, width=0.35, color=ALL_COLORS[:len(n_news_per_cluster)])
    ax.set_ylabel('Number of news')
    ax.set_title('Number of news in each cluster')
    ax.set_xticks(x)
    ax.set_xticklabels(['Cluster #'+str(c) for c in range(len(n_news_per_cluster))])
    if path_to_save:
        plt.savefig(path_to_save+'/3a. n NEWS per cluster.png')
        plt.close()
    else:
        plt.show()
        
    fig, ax = plt.subplots(figsize = (20, 10))
    for c in range(len(n_news_per_cluster_per_date)):
        x = np.arange(len(n_news_per_cluster_per_date[c]))
        ax.plot(x, n_news_per_cluster_per_date[c], color=ALL_COLORS[c], label='Cluster #'+str(c))       
    ax.set_ylabel('Number of news')
    ax.set_title('Number of news per cluster in time-span (absolute value)')
    ax.set_xticks(x)
    ax.set_xticklabels(date_range)
    ax.legend()
    if path_to_save:
        plt.savefig(path_to_save+'/3b. number of NEWS per day per cluster (absolute value).png')
        plt.close()
    else:
        plt.show()
            
    fig, ax = plt.subplots(figsize = (20, 10))
    for c in range(len(perc_news_per_cluster_per_date)):
        x = np.arange(len(perc_news_per_cluster_per_date[c]))
        ax.plot(x, perc_news_per_cluster_per_date[c], color=ALL_COLORS[c], label='Cluster #'+str(c))       
    ax.set_ylabel('% of news')
    ax.set_title('Percentage of news per cluster in time-span (% over total daily news)')
    ax.set_xticks(x)
    ax.set_xticklabels(date_range)
    ax.legend()
    if path_to_save:
        plt.savefig(path_to_save+'/3c. percentage of NEWS per day per cluster.png')
        plt.close()
    else:
        plt.show()
               
    fig, ax = plt.subplots(figsize = (20, 10))
    x = np.arange(len(discarded_news_per_date))
    ax.plot(x, discarded_news_per_date, color='black')       
    ax.set_ylabel('Number of discarded news')
    ax.set_title('Number of discarded news per day')
    ax.set_xticks(x)
    ax.set_xticklabels(date_range)
    if path_to_save:
        plt.savefig(path_to_save+'/3e. number of discarded NEWS per day.png')
        plt.close()
    else:
        plt.show()
        
    fig, ax = plt.subplots(figsize = (20, 10))
    x = np.arange(len(perc_discarded_news_per_date))
    ax.plot(x, perc_discarded_news_per_date, color='black')       
    ax.set_ylabel('% of discarded news')
    ax.set_title('Percentage of discarded news per day (% over total daily news)')
    ax.set_xticks(x)
    ax.set_xticklabels(date_range)
    if path_to_save:
        plt.savefig(path_to_save+'/3f. percentage of discarded NEWS per day.png')
        plt.close()
    else:
        plt.show()
    
    if silhouette_avg:
        x = np.arange(len(silhouette_per_cluster))
        fig, ax = plt.subplots()
        ax.bar(x, silhouette_per_cluster, width=0.35, color=ALL_COLORS[:len(silhouette_per_cluster)])
        ax.set_ylabel('Silhouette score')
        ax.set_title('Silhouette score, averaged per cluster')
        ax.set_xticks(x)
        ax.set_xticklabels(['Cluster #'+str(c) for c in range(len(silhouette_per_cluster))])
        plt.axhline(y=silhouette_avg, color='red', label='Avg. silhouette')
        plt.axhline(y=dunn_index, color='orange', label='Dunn index')
        plt.legend()
        if path_to_save:
            plt.savefig(path_to_save+'/5. silhouette per cluster.png')
            plt.close()
        else:
            plt.show()
    
    x = np.arange(len(pos_lexicon_match_per_cluster))  # the label locations
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, pos_lexicon_match_per_cluster, width=0.35, label='Positive lexicon')
    ax.bar(x + width/2, neg_lexicon_match_per_cluster, width=0.35, label='Negative lexicon')
    ax.set_ylabel('Lexicon match (%)')
    ax.set_title('Lexicon match for clusters')
    ax.set_xticks(x)
    ax.set_xticklabels(['Cluster #'+str(c) for c in range(len(pos_lexicon_match_per_cluster))])
    ax.legend()
    if path_to_save:
        plt.savefig(path_to_save+'/6. lexicon match per cluster.png')
        plt.close()
    else:
        plt.show()
    
    
    if keywords_in_relevant_words:
        x = np.arange(len(keywords_in_relevant_words))
        fig, ax = plt.subplots()
        ax.bar(x, keywords_in_relevant_words, width=0.35, color=ALL_COLORS[:len(keywords_in_relevant_words)])
        ax.set_ylabel('% of test-keywords')
        ax.set_title('Percentage of test-keywords among\nthe top relevant words, for each cluster')
        ax.set_xticks(x)
        ax.set_xticklabels(['Cluster #'+str(c) for c in range(len(keywords_in_relevant_words))])
        if path_to_save:
            plt.savefig(path_to_save+'/8a. test keywords among relevant words.png')
            plt.close()
        else:
            plt.show()
            
    if keywords_in_titles:
        x = np.arange(len(keywords_in_titles))  # the label locations
        width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x - width/2, keywords_in_titles, width=0.35, label='Titles')
        ax.bar(x + width/2, keywords_in_snippets, width=0.35, label='Snippets')
        ax.set_ylabel('% of news with test-keywords')
        ax.set_title('Percentage of relevant news containing\nat least 1 test-keyword, for each cluster')
        ax.set_xticks(x)
        ax.set_xticklabels(['Cluster #'+str(c) for c in range(len(keywords_in_titles))])
        ax.legend()
        if path_to_save:
            plt.savefig(path_to_save+'/8b. test keywords in relevant news.png')
            plt.close()
        else:
            plt.show()
            
    
    if n_tweets_per_cluster:
        
        x = np.arange(len(n_tweets_per_cluster))
        fig, ax = plt.subplots()
        ax.bar(x, n_tweets_per_cluster, width=0.35, color=ALL_COLORS[:len(n_news_per_cluster)])
        ax.set_ylabel('Number of tweets')
        ax.set_title('Number of tweets in each cluster')
        ax.set_xticks(x)
        ax.set_xticklabels(['Cluster #'+str(c) for c in range(len(n_tweets_per_cluster))])
        if path_to_save:
            plt.savefig(path_to_save+'/9a. number of TWEETS per cluster.png')
            plt.close()
        else:
            plt.show()
            
        fig, ax = plt.subplots(figsize = (20, 10))
        for c in range(len(n_tweets_per_cluster_per_date)):
            x = np.arange(len(n_tweets_per_cluster_per_date[c]))
            ax.plot(x, n_tweets_per_cluster_per_date[c], color=ALL_COLORS[c], label='Cluster #'+str(c))       
        ax.set_ylabel('Number of tweets')
        ax.set_title('Number of TWEETS per cluster in time-span')
        ax.set_xticks(x)
        ax.set_xticklabels(date_range)
        ax.legend()
        if path_to_save:
            plt.savefig(path_to_save+'/9b. number of TWEETS per day per cluster.png')
            plt.close()
        else:
            plt.show()
            
        fig, ax = plt.subplots(figsize = (20, 10))
        for c in range(len(perc_tweets_per_cluster_per_date)):
            x = np.arange(len(perc_tweets_per_cluster_per_date[c]))
            ax.plot(x, perc_tweets_per_cluster_per_date[c], color=ALL_COLORS[c], label='Cluster #'+str(c))       
        ax.set_ylabel('% of tweets')
        ax.set_title('Percentage of tweets per cluster in time-span (% over total daily tweets)')
        ax.set_xticks(x)
        ax.set_xticklabels(date_range)
        ax.legend()
        if path_to_save:
            plt.savefig(path_to_save+'/9c. percentage of TWEETS per day per cluster.png')
            plt.close()
        else:
            plt.show()
        
        fig, ax = plt.subplots(figsize = (20, 10))
        x = np.arange(len(total_tweets_per_date))
        ax.plot(x, total_news_per_date, color='darkorange')
        ax.plot(x, total_tweets_per_date, color='indigo')       
        ax.set_ylabel('n. of tweets and news')
        ax.set_title('Number of total news and tweets per day')
        ax.set_xticks(x)
        ax.set_xticklabels(date_range)
        if path_to_save:
            plt.savefig(path_to_save+'/9d. number of total NEWS and TWEETS per day.png')
            plt.close()
        else:
            plt.show()
        
        fig, ax = plt.subplots(figsize = (20, 10))
        x = np.arange(len(discarded_tweets_per_date))
        ax.plot(x, discarded_tweets_per_date, color='black')       
        ax.set_ylabel('Number of discarded tweets')
        ax.set_title('Number of discarded tweets in time-span')
        ax.set_xticks(x)
        ax.set_xticklabels(date_range)
        if path_to_save:
            plt.savefig(path_to_save+'/9e. number of discarded TWEETS per day.png')
            plt.close()
        else:
            plt.show()
        
        fig, ax = plt.subplots(figsize = (20, 10))
        x = np.arange(len(perc_discarded_tweets_per_date))
        ax.plot(x, perc_discarded_tweets_per_date, color='black')       
        ax.set_ylabel('% of discarded tweets')
        ax.set_title('Percentage of discarded tweets per day (% over total daily tweets')
        ax.set_xticks(x)
        ax.set_xticklabels(date_range)
        if path_to_save:
            plt.savefig(path_to_save+'/9f. percentage of discarded TWEETS per day.png')
            plt.close()
        else:
            plt.show()
            
        x = np.arange(len(relevant_words_match))
        fig, ax = plt.subplots()
        ax.bar(x, relevant_words_match, width=0.35, color=ALL_COLORS[:len(n_news_per_cluster)])
        ax.set_ylabel('common words %')
        ax.set_title('Percentage of common relevant words (top 30)\nbetween news and tweets, per cluster')
        ax.set_xticks(x)
        ax.set_xticklabels(['Cluster #'+str(c) for c in range(len(relevant_words_match))])
        if path_to_save:
            plt.savefig(path_to_save+'/12. percentage of common relevant words (top 30) between news and tweets, per cluster.png')
            plt.close()
        else:
            plt.show()
        
    
"""
Plots or saves (if path_to_save is not None) the points defined by vectors_2D and centroids_2D, coloring them according to labels
(which indicate the cluster each point belongs to).
is_outlier is a vector of the same length as vectors_2D; is_outlier[i] == 1 <=> vectors_2D[i] is an outlier.
If plot_outliers is True, the outlier points are plotted and colored black.

Refer to function 'cluster_news' for further reference on 'labels' parameter.
"""
def plot_clusters(vectors_2D, centroids_2D, labels, is_outlier=[], plot_outliers=True, chosen_algorithm=None, path_to_save=None):

    if is_outlier == []:
        is_outlier = np.zeros(len(vectors_2D))
    
    fig, ax = plt.subplots()
    for l,c in zip(np.unique(labels[labels != -1]), range(len(centroids_2D))):
        indices = [k for k in range(len(labels)) if labels[k] == l and is_outlier[k] == 0]
        ax.scatter(vectors_2D[indices,0], vectors_2D[indices,1], c=ALL_COLORS[l], label='Cluster #'+str(l))    
        if len(indices) > 0:
            ax.scatter(centroids_2D[c,0], centroids_2D[c,1], s=50, c=ALL_COLORS[l], marker='+')
    
    if is_outlier.any():
        if plot_outliers:
            outlier_indices = [k for k in range(len(labels)) if is_outlier[k] == 1]
            ax.scatter(vectors_2D[outlier_indices,0], vectors_2D[outlier_indices,1], c='black', label='outliers')
        title = 'Clustering after outlier removal - ' + chosen_algorithm
        file_name = '2. clustering after outlier removal.png'
    else:
        title = 'Original clustering - ' + chosen_algorithm
        file_name = '1. original clustering.png'
        
    ax.set_title(title)
    ax.legend()
    if path_to_save:
        plt.savefig(path_to_save+'/' + file_name)
        plt.close()
    else:
        plt.show()
        
        
"""
Prints the boxplot of some scores obtained by 'detect_events'.
"""
def print_boxplot(scores_by_algorithm, algorithms_list, title, path_to_save=None):
    
    x = [2*k for k in range(len(algorithms_list))]
    fig, ax = plt.subplots()
    ax.boxplot([scores_by_algorithm[alg] for alg in algorithms_list], widths=0.6, positions=x, whis=[25,75])
    ax.set_xticks(x)
    ax.set_xticklabels([alg + '\n(Avg: '+str(round(np.average(scores_by_algorithm[alg]),2))+')' for alg in algorithms_list])
    ax.set_title(title)
    plt.savefig(path_to_save)
    plt.close()
        