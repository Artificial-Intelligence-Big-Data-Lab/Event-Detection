# AIBD Event Detection

## 0. Content

This repository contains the scripts to execute the event detection algorithm for financial forecasting, described in the paper "Event detection in finance using hierarchical clustering algorithms on news and tweets" (S. Carta, S. Consoli, L. Piras, A.S. Podda, D. Reforgiato Recupero).

The root of the project should contain the following subfolders: `DNA and tweets`, `lexicons/3 classes`, `scripts`, `SP500_market_data`, `word2vec_data`. The content of each folder is the following:

- DNA and tweets : 
	- `sp500_news_example.json`: example news published in 2020 about SP500. An article was included in the collection iff the keyword `Standard & Poor`, `SPX` or similar appeared at least once in the text of the article (title, snippet or body)
	- `sp500_tweets_example.json`: example tweets posted on StocTwits in 2020 about SP500. A tweet was included in the collection iff the cashtag `$SPX`was included in the text of the tweet

- lexicons : used when creating new lexicons. Each subfolder inside `3 classes` is named according to the parameters passed to create the lexicons; similarly, a function in `scripts/create_lexicons.py` uses the same naming system to retrieve the lexicons based on the parameters. Each subfolder contains a list of csv files, each representing the lexicon created for the day indicated in the file name. Read documentation in "`scripts/create_lexicons.py`" for further reference on the saving system.

- scripts:
	- `output_clustering` : subfolder where the output of 'event_detector.py' will be saved. See this function for reference on the saving system.
	- `create_lexicons.py` : functions used for creating new lexicons and for fetching previously created lexicons
	- `dunn_index.py` : copied from some github repo, useful to compute the dunn index given a clustering
	- `event_detector.py` : main script
	- `plotter.py` : functions to plot 2D visualizations of clusters and various statistics
	- `utils.py` : utility functions

- SP500_market_data : daily time series for all companies included in the SP500 index. Also the time series for the global prices of SP500 are included in the folder (file `SP500.csv`)

- word2vec_data :
	- `google_word2vec_classic.bin` : word-embedding model pre-trained on Google news (download here: https://code.google.com/archive/p/word2vec/)
	- `google_word2vec_sentiment.bin` : word-embedding model based on google_word2vec_classic.bin; adjusted in such a way, that makes embeddings more aware of the sentiments of the words (reference: https://towardsdatascience.com/sentiment-preserving-word-embeddings-9bb5a45b2a5).
	

Please note that, due to licensing constraints, we cannot publish any news document contained in the Dow Jones' Data, News and Analytics dataset, which was employed in the experimental framework of the paper. However, we hereby make available a small example set of news and tweets for illustrative purpose. To allow the reproducibility of the experiments, your data should follow the same format.

Furthermore, a set of pre-computed lexicons is already available in the folder `lexicons`. In case you want to use lexicons with different parameters, you need to create new lexicons from scratch, based on the small example data. Please read the documentation on the script `create_lexicons.py` for reference.

For size constraint, it was not possible to upload the word-embedding model to this repository. We recommend that the user download the model from the link indicated above (https://code.google.com/archive/p/word2vec/) or, alternatively, generate word-embeddings using the interface offered by the gensim library: https://radimrehurek.com/gensim/models/word2vec.html. It is crucial that the final model may be read as a classic Python dictionary, in which keys are words and values are vectors.



## 1. MongoDB configuration

Install mongodb community edition: https://www.mongodb.com/try/download/community
I recommend also installing Robo3T, to manage databases and collections from graphic interface: https://robomongo.org/download

Once the installation is complete, create a database called "financial_forecast" (this operation is very easy from Robo3T interface).

Now let's create mongodb collections for news and tweets for SP500, with the respective indexes.
- unzip `event_detection_repo.zip`
- Open a shell or windows command line 
- Go to directory `event_detection_repo/DNA` and tweets
- run command >
```
mongoimport --db financial_forecast --collection sp500_news_2009-2020 --file sp500_news_example.json --legacy
```
- run command > 
```
mongoimport --db financial_forecast --collection sp500_tweets_2009-2020 --file sp500_tweets_example.json --legacy
```
- verify on Robo3T that collections have been created successfully
- on Robo3T command line, run `db.getCollection('sp500_news_2009-2020').createIndex( { ingestion_datetime: 1 } )`
- on Robo3T command line, `run db.getCollection('sp500_tweets_2009-2020').createIndex( { ingestion_datetime: 1 } )`


## 2. Python modules installation

In order to run the code, you must have the following Python modules installed and updated:
- sklearn : https://scikit-learn.org/stable/install.html
- numpy : https://numpy.org/install/
- pymongo : https://pymongo.readthedocs.io/en/stable/installation.html
- gensim : https://radimrehurek.com/gensim/index.html
- nltk : https://www.nltk.org/install.html
- pyclustering : https://pyclustering.github.io/docs/0.9.0/html/index.html#install_sec
- json2table : https://pypi.org/project/json2table/


## 3. Executing the code

The main script to run is `scripts/event_detector.py`. 
You can simply run this command from shell:
`event_detection_repo/scripts` > `python event_detector.py`

For efficient memory management, I recommend running the script from the system shell and not from the IDE. For example, Spyder IDE does not clean the RAM after each execution and this causes fast memory overload.

The other files contain functions that are called by `event_dectector.py` for specific tasks. Read the documentation in each single file for further reference.
