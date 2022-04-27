# Sentiment Analysis Covid Vaccine

### Who we are
This project is developed by 4 students of USI "Visual analytics" course:

* Simone Rava (ravasimoneit@gmail.com)
* Valdet Ismaili (ismaiv@usi.ch)
* Federico Lombardo (lombafe@usi.ch)
* Felici Rocco (felicr@usi.ch)

### A brief description
* The aim is to show some interesting patterns found out analyzing the sentiment of the population during the COVID vaccination campaign and visualize them in a dashboard
* For our scope we analyzed the sentiment based on the text of tweet posts. 

### The dashboard 
*__This is the [link](http://195.176.181.168:5601/goto/d1a6e50ef01eb3f512d06a68b8918ffe) for the dashboard.__*

The dashboard presents some overall metrics of the data taken into account for the visualization: 

* Total number of tweets 
* Countries 

Notice that these metrics change interactively when a selection or a filtering is done by the user.

The dashboard presents some visualizations: 

* Two bar-chart: one representing the number of tweets per sentiment for the three top country (in terms of number of tweets); the other regarding the number of retweet and favorites per sentiment.
* A time serie of the number of tweets per sentiment. The time precision is the day. 
* A words-cloud regarding the most cited hashstags.

Moreover it is possible to explore the raw data directly on a table. 
Features available are:

* date in format month day, year @ hour:minute:second
* text of the tweet
* user_country
* link regarding an important event occurred in the moment the tweet was written and in the country the tweet was located

It is possible to filter data simply interacting with the dashboard (selecting portion of time, single tweet from the table et cetera) or by a set of predefined filters:

* Important events
* Countries
* Number of deaths per day
* Number of cases per day
* User verified
* Number of followers
* Number of retweets



### Research question
In December 2019, the first cases of pneumonia were reported. Shortly afterwards they were identified that a new form of coronavirus. To deal with the emergency, in June 2020, the first vaccines were administered. Today, in the midst of the vaccination campaigns, social networks have become a point of exchange of opinions on the subject. Our work focuses on the tweets to *understand the perception, and in particular the sentiment, of users* regarding the events of this latter period.

### Modus operandi and technologies
The data was collected by twitter via the **tweepy API** based on keywords and hashtags such as 'Moderna', 'Pfizer', 'Vaccines'... to ensure that they were related to the topic of study. Sentiment (positive, negative or neutral) was extracted from the natural language of the tweets using **nltk** and **textblob** for the model. Quantitative data regarding the number of cases, deaths and daily vaccines were then integrated from the **OurWorldInData repository** provided by the Oxford University.
The data were preprocessed with **Python**, then **Elastic** was used for data storage and ingestion and **Kibana** for visualisation.
