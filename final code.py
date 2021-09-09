# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:58:15 2021

@author: user
"""

#import the necessary libraries 
import pandas as pd 
import numpy as np
import re
import string 
import matplotlib.pyplot as plt
%matplotlib inline
import datetime
import seaborn as sns
import nltk
from pylab import rcParams
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os
from os import path
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


######TWEET EXTRACTION######
#Creating a list to append tweet data to
tweets = []

#Usage of TwitterSearchScraper to scrape data that contain the words
#"Dogecoin", "DOGE" and "$DOGE"
#and append the extracted tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper
                ('Dogecoin, DOGE $DOGE lang:en since:2020-08-01 until:2021-05-31').get_items()):
    if i>200000: #limit of extracted tweets
        break
    tweets.append([tweet.date, tweet.content])
    
# Creating a dataframe to store the above tweets list
final = pd.DataFrame(tweets, columns=['Datetime', 'Text'])


#read the financial historical data
historical = pd.read_csv('C:/Python/historical.csv') 

#fix the dates 
historical['Date'] = pd.to_datetime(historical['Date'], format='%m/%d/%Y') #convert to datetime

# Create new columns for month and year
historical['month'] = historical['Date'].dt.month
historical['year'] = historical['Date'].dt.year

#remove hours
historical['Date'] = historical['Date'].dt.date 

#make a plot to see the historical data
rcParams['figure.figsize'] = 18, 8 #change the dimensions for better visualisation
sns.lineplot(data=historical, x="Date", y="Close/Last", color='red')
plt.title('Dogecoin Price') 

###DATA PREPROCESSING
#function that preprocesses every tweet 
def CleanText(text):
#remove usernames (@user)   
    text = re.sub(r'@[A-Za-z0-9]+', '', text) 
#remove hashtags (#word)
    text = re.sub(r'#', '', text) #idk if i should remove the whole word after the hashtag
#remove retweet symbol (RT)
    text = re.sub(r'RT[\s]+','', text)
#remove urls
    text = re.sub(r'https?:\/\/\S+', '', text)
#remove line breaks
    text = text.replace('\n', ' ')
#remove quotation marks
    text = text.replace('"', '')
#remove tickers
    text = re.sub(r'\$\w*', '', text)
#remove whitespace (including new line characters)
    text = re.sub(r'\s\s+', ' ', text)
#remove single space remaining at the front of the tweet
    text = text.lstrip(' ')

    return text

#apply the function at the tidy tweet column
final['tidy_tweet'] = final['Text'].apply(CleanText)


######SENTIMENT ANALYSIS######
analyzer = SentimentIntensityAnalyzer()

#create lists to store the scores
scores = []
compound_list = []
positive_list = []
negative_list = []
neutral_list = []

#estimate compound score, negative, positive, neutral tweets
for i in range(final['tidy_tweet'].shape[0]):
    compound = analyzer.polarity_scores(final['tidy_tweet'][i])["compound"]
    pos = analyzer.polarity_scores(final['tidy_tweet'][i])["pos"]
    neu = analyzer.polarity_scores(final['tidy_tweet'][i])["neu"]
    neg = analyzer.polarity_scores(final['tidy_tweet'][i])["neg"]
    
    scores.append({"Compound": compound,
                       "Positive": pos,
                       "Negative": neg,
                       "Neutral": neu
                  })

#Convert dictionary to dataframe 
sentiments_score = pd.DataFrame.from_dict(scores)

#join the dataframes
final = final.join(sentiments_score)

#fix dates
final['Datetime'] = pd.to_datetime(final['Datetime'], format='%Y-%m-%d') 
final['Datetime'] = final['Datetime'].dt.date #remove hours from date

#i want to aggregate the compound scores of each day at the sentiment analysis
new_df = final.groupby('Datetime').mean().reset_index() #mean to take into account all data values

#make a plot to see the compound score
rcParams['figure.figsize'] = 18, 8 #change the dimensions for better visualisation
sns.lineplot(data=final, x="Compound", y='Datetime')

plt.title('Compound Score') 
plt.draw() 
plt.show() 

#invert the dates to have the similar flow as the dates at the dataframe with sentiment scores
historical = historical.iloc[::-1].reset_index()

#delete unecessary columns
del historical['index']
del historical['Volume'] 
del historical['Open']
del historical['High']
del historical['Low']

#merge the datasets
merged = pd.DataFrame(new_df['Datetime'])
merged['compound'] = new_df['Compound']
merged['close'] = historical['Close/Last']
merged['pos'] = new_df['Positive']
merged['neg'] = new_df['Negative']
merged['neu'] = new_df['Neutral']

#plot the compound score
time_btc = pd.Series(data=merged['compound'].values,index=merged.Datetime)
time_btc.plot(figsize=(16, 4), label="Compound Score", legend=True,color='b')
plt.show()

#histograms with sentiment scores
plt.title("Compound Score from Tweets on Dogecoin - Averaged Weekly")
sns.histplot(data=merged, x="compound")

plt.title("Positive Score from Tweets on Dogecoin - Averaged Weekly")
sns.histplot(data=merged, x="pos")

plt.title("Negative Score from Tweets on Dogecoin - Averaged Weekly")
sns.histplot(data=merged, x="neg")

plt.title("Neutral Score from Tweets on Dogecoin - Averaged Weekly")
sns.histplot(data=merged, x="neu")

#wordcloud code
# Create and generate a word cloud image:
wordcloud = WordCloud(width=900,
                      height=500,
                      max_words=500,
                      max_font_size=100,
                      relative_scaling=0.5,
                      colormap='Blues',
                      normalize_plurals=True).generate(' '.join(final['tidy_tweet']))
plt.figure(figsize=(17,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#remove stopwords - most commonly used words
#Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["doge", "dogecoin", "wine", "dogearmy", "trading", "group"])

#Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, width=900,
                      height=500,
                      max_words=500,
                      max_font_size=100,
                      relative_scaling=0.5,
                      colormap='Blues',
                      normalize_plurals=True).generate(' '.join(final['tidy_tweet']))
plt.figure(figsize=(17,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


final['number_tweets']=''
#calculate the number of tweets per day 
def tweets_per_day(df):
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d')
    return df[['number_tweets']].groupby(df['Datetime'].dt.date).count()
  
number_tweets = tweets_per_day(final)
number_tweets.hist()
number_tweets.reset_index(drop=True, inplace=True) #drop the indexing to achieve the concat
#store number of tweets along with the other useful data
merged['number_tweets'] = number_tweets

#number of tweets plot
time_btc = pd.Series(data=merged['number_tweets'].values,index=merged.Datetime)
time_btc.plot(figsize=(16, 4), label="Number of Tweets", legend=True,color='b')
plt.show()

#create a graph with both close price and number of tweets
# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(historical.Date, historical['Close/Last'], color="red")
# set x-axis label
ax.set_xlabel("Date",fontsize=14)
# set y-axis label
ax.set_ylabel("Close Price",color="red",fontsize=14)
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(historical.Date, time_btc,color="blue")
ax2.set_ylabel("Number of Tweets",color="blue",fontsize=14)
plt.show()

#estimate the corelation 
corr = merged.corr() 

#heatmap for correlation
sns.heatmap(corr,annot=True,linewidths=.5,cmap="YlGnBu")

#scatterplot for correlation
plt.plot(merged.number_tweets,merged.close,'o',markersize=2, color='blue')
plt.xlabel('Number of Tweets')
plt.ylabel('Close')
plt.show()


#######LOGISTIC REGRESSION######

#Define Predictor/Independent Variables
merged['S_10'] = merged['close'].rolling(window=10).mean()
merged['Corr'] = merged['close'].rolling(window=10).corr(merged['S_10'])

#delete unecessary data
merged = merged.dropna()
X = merged.iloc[:,:9]
del X['Datetime']
del X['number_tweets']

#Define dependent variable - make it binary
y = np.where(merged['close'].shift(-1) > merged['close'],1,0)

#split The dataset
split = int(0.7*len(merged))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

#apply logistic regression 
model = LogisticRegression()
model = model.fit (X_train,y_train)

#examine coefficients
pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))

#calculate class probabilities
probability = model.predict_proba(X_test)
print(probability)

#predict class labels
probability = model.predict_proba(X_test)
print(probability)
predicted = model.predict(X_test)

####Evaluate The Model
#Confusion Matrix
print(metrics.confusion_matrix(y_test, predicted))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#plot the confusion matrix
plot_confusion_matrix(model, X_test, y_test)  
plt.title('Confusion Matrix')
plt.colorbar()
plt.show() 

#estimate model accuracy
print(model.score(X_test,y_test))

#classification report
print(metrics.classification_report(y_test, predicted))
print("Accuracy:",metrics.accuracy_score(y_test, predicted))

#plot with the binary outcomes
sns.countplot(predicted, palette="Blues_d")


######RANDOM FOREST REGRESSION######

#create the merged dataframe again since it was altered during the logistic regression
merged = pd.DataFrame(new_df['Datetime'])
merged['compound'] = new_df['Compound']
merged['close'] = historical['Close/Last']
merged['pos'] = new_df['Positive']
merged['neg'] = new_df['Negative']
merged['neu'] = new_df['Neutral']
merged['number_tweets']= number_tweets

#train test split
X = merged.drop(['close', 'Datetime'], axis='columns')
y = merged.close
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#apply rf regression
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#evaluating the regression
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#estimate the regression score
regressor.score(X_train, y_train)
regressor.score(X_test, y_test)

#apply rf again with only high correlated aka negatives and number of tweets
#train test split
X1 = merged.drop(['close','compound', 'pos', 'neu','Datetime'], axis='columns')
y = merged.close
X_train1, X_test1, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=0)

#feature scaling
X_train1 = sc.fit_transform(X_train1)
X_test1 = sc.transform(X_test1)

#apply rf regression
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train1, y_train)
y_pred = regressor.predict(X_test1)

#evaluating the regression
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#estimate the regression score
regressor.score(X_train1, y_train)
regressor.score(X_test1, y_test)

#apply rf again with just with twitter volume
#train test split
X2 = merged.drop(['close','compound', 'pos', 'neu', 'neg','Datetime'], axis='columns')
y = merged.close
X_train2, X_test2, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=0)

#feature scaling
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train2)
X_test2 = sc.transform(X_test2)

#apply rf regression
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train2, y_train)
y_pred = regressor.predict(X_test2)

#evaluating the regression
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#estimate the regression score
regressor.score(X_train2, y_train)
regressor.score(X_test2, y_test)
