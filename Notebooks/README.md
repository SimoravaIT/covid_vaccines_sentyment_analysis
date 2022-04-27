# READ ME

## Notebooks Reading Order

- first_data_cleansing
- sentiment
- merge_cases_deaths_complete
- merge_events
- data_ingestion

(Notebooks Stored in the Notebook test are just operation made on
small segment of data in order to get a better understanding of the
packages and are not used in the overall project)


## Information about generate_sentiment_classifier.py
### Acknowledgements
### Datasource
The training data was taken from

[http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)

### Machine Learning
The proper algorithm and parameters to use and feature extraction was mostly taken from the following  article.

[https://www.kaggle.com/ngyptr/lstm-sentiment-analysis-keras](https://www.kaggle.com/ngyptr/lstm-sentiment-analysis-keras)


### Run Classifier Generation
To run the classifier generation first make sure a directory called `out` exists and then  run
>```python generate_sentiment_classifier.py <path-to-sentiment-140.csv> <optional: TRAINING_SIZE> ```

The output will be placed in the `out`  folder and will have names `TRAINING_SIZE-sentiment_classifier.h5` and `TRAINING_SIZE-tokenizer.pickle`. Both of these files are needed for the sentiment server.

(Note on MacBook Pro (15-inch, 2018), 2.2 GHz Intel Core i7 running generate_sentiment_classifier with TRAINING_SIZE=200000 took around 25 min)
