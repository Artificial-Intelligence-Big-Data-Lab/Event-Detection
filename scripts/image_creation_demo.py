# -*- coding: utf-8 -*-
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import matplotlib.pyplot as plt

from create_lexicons import fetch_lexicons

# the image will be a square of size img_size x img_size
img_size = 100



# load the pre-trained word-embedding model
with open('../word2vec_data/google_word2vec_sentiment.bin', 'rb') as f:
    WORD_EMBEDDING_MODEL = pickle.load(f)



# fetch the lexicons (this is done just for convenience in this script, to create the super lexicon)
# could be removed in the final implementation
pos_lexicons, neg_lexicons = fetch_lexicons(industry='SP500', collection_name='sp500_news_2009-2019',
                                            min_date='2016-01-01', max_date='2018-12-30',
                                            look_back=28, ngram_range=(1,1), max_df=0.9, min_df=10,
                                            positive_percentile_range=(50,100), negative_percentile_range=(0,50))

# In this demo, the super-lexicon is given by the union of all the positive and negative lexicons between min_date and max_date
# NOTE: since the lexicons contain only stemmed words, also the super-lexicon contains only stemmed words; 
# however, the WORD_EMBEDDING_MODEL contains only the extended forms of the words. 
# For this reason, we should think of a different way to obtain the super-lexicon, because it must contain extended forms
# (for example, the super lexicon could be a return value of the function 'detect_events')
super_lexicon = []
for date in pos_lexicons:
    current_pos_lexicon = [w for w,s in pos_lexicons[date]] 
    current_neg_lexicon = [w for w,s in neg_lexicons[date]]
    for w in current_pos_lexicon + current_neg_lexicon:
        if w in WORD_EMBEDDING_MODEL and w not in super_lexicon:
            super_lexicon.append(w)
            
print('\nSize of super-lexicon:', len(super_lexicon))

# at position i, we store the 300-d word embedding of the i-th word in super_lexicon
embeddings_300d = np.array([WORD_EMBEDDING_MODEL[w] for w in super_lexicon])

# apply tsne for dimensionality reduction to obtain simple (x,y) coordinates for each word in the super_lexicon
# NOTE: n_components could be set also to 3 or higher values, in case CNNs are trained with 3-dimensional images or more
embeddings_2d = TSNE(n_components=2, metric='cosine', perplexity=30, early_exaggeration=3).fit_transform(embeddings_300d)
# we scale all the coordinates in the range between 0 and img_size
embeddings_2d = MinMaxScaler(feature_range=(0,img_size)).fit_transform(embeddings_2d)

# In the final implementation, the code up to here should be performed only once for each training set.
# In fact, both the super-lexicon and the TSNE transformation of words are valid for the whole training set.

# In the continuation of the script, you'll find the code that needs to be applied to every single day in the training set.

# example of words for a given day; the scores will be computed within detect_events
todays_words = {'brexit' : 1,
                'referendum' : 1,
                'britain' : 0.9,
                'poll': 0.8,
                'vote' : 0.8,
                'uk' : 0.8,
                'leave' : 0.8,
                'remain' : 0.7,
                'england' : 0.7,
                'scotland' : 0.7,
                'ireland' : 0.6,
                'stock' : 0.4,
                'money' : 0.4}

# find the indices of our list of words inside the super_lexicon
todays_words_indices = []
todays_words_scores = []        # this will be used to assign the value of the pixel
todays_words_list = []
for i in range(len(super_lexicon)):
    if super_lexicon[i] in todays_words:
        todays_words_indices.append(i)
        todays_words_scores.append(todays_words[super_lexicon[i]])
        todays_words_list.append(super_lexicon[i])


# create the image
print('\nWords found in super-lexicon:')
image = np.zeros((img_size,img_size))     #at first, the image is just a matrix of 0s   
for index,score in zip(todays_words_indices, todays_words_scores):
    # we get the x and y coordinates from the 2D embeddings
    # we need to cast to int to be able to address the locations of the matrix
    x_coord = int(embeddings_2d[index][0])
    y_coord = int(embeddings_2d[index][1])
    print()
    print(super_lexicon[index])
    print('x:', x_coord, 'y:',y_coord)
    print('Score:', score)
    # we fill the pixel with the score assigned to the corresponding word
    image[x_coord,y_coord] = score

# at this stage, 'image' in a 2-dimensional numpy array that can be fed as input to the CNN
# the following code is just for visualization, but can be ignored for integration purposes 

# visualization of the activated pixels
# NOTE: the coordinates of the matrix 'image' and of the plot do not correspond, because in matplotlib
# the point (0,0) is the bottom-left corner, while in 'image' (0,0) means first row from the top, first column
color_levels = [(cl,0,0) for cl in todays_words_scores]
fig, ax = plt.subplots()
ax.scatter(embeddings_2d[todays_words_indices,0], embeddings_2d[todays_words_indices,1], c = color_levels)
ax.set_xlim(0, img_size)
ax.set_ylim(0, img_size)
for i in range(len(todays_words_indices)):
    ax.annotate(todays_words_list[i], (embeddings_2d[todays_words_indices[i],0], embeddings_2d[todays_words_indices[i],1]), color='black', fontsize=10)   
plt.show()
    
    

