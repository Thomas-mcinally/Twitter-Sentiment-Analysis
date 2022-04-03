import tweepy
import config
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt


def convert_tweepy_data_to_dataframe(tweet_data:list) -> DataFrame:
    '''
    Extracts useful info from tweets downloaded from twitter api, and put results in a pandas dataframe

    Parameters:
        tweet_data (list) - List of tweepy tweet objects. Downloaded using API

    Returns:
        df (pandas dataframe) - Dataframe containing datetime, retweet_count, reply count, like count for each tweet in tweet_data
    '''
    datetimes = [tweet.created_at for tweet in tweet_data]
    texts = [tweet.text for tweet in tweet_data]
    retweet_counts = [tweet.public_metrics['retweet_count'] for tweet in tweet_data]
    reply_counts = [tweet.public_metrics['reply_count'] for tweet in tweet_data]
    like_counts = [tweet.public_metrics['like_count'] for tweet in tweet_data]

    df = pd.DataFrame(data = {'datetime':datetimes, 'text':texts, 'retweet_count':retweet_counts, 'reply_count':reply_counts, 'like_count':like_counts})

    return df


def get_BERT_sentiment(text:str, model:str) -> int:
    '''
    Calculates sentiment for text, using pre-trained model from Hugging Face

    Parameters:
        text (str) - String to calculate sentiment for
        model (str) - Location to download pre-trained model from
            e.g. model = 'cardiffnlp/twitter-roberta-base-sentiment' uses model downloaded from https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
        
    Returns:
        sentiment (int) - Sentiment score for the text. Takes values -1, 0 or +1 (negative, neutral or positive)
    '''
    #Load model and tokenizer
    model=AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    encoded_tweet = tokenizer(text, return_tensors='pt')
    output = model(encoded_tweet['input_ids'],encoded_tweet['attention_mask'])
    scores=np.array(output[0][0].tolist())
    if max(scores) == scores[1]:
        #Neutral tweet
        sentiment=0
    elif max(scores) == scores[0]:
        #Negative tweet
        sentiment=-1
    elif max(scores) == scores[2]:
        #Positive tweet
        sentiment=1
    return sentiment

def get_weighted_sentiment(row:Series) -> float:
    '''
    Calculates the weighted sentiment score of a tweet, using its like count

    Parameters:
        row (pandas Series) - Series containing sentiment and like_count for a tweet
    
    Returns:
        weighted_sentiment (float) - Weighted sentiment score
    '''
    if row['like_count'] > 0:
        weighted_sentiment = row['like_count'] * row['sentiment']   
    else:
        #like_count=0
        weighted_sentiment = row['sentiment']
    
    return weighted_sentiment


def clean_text(df:DataFrame) -> DataFrame:
    '''
    Cleans up text column of tweet dataframe

    Parameters:
        df (Pandas DataFrame) - Dataframe containing data for downloaded tweets
    
    Returns:
        clean_df (Pandas DataFrame) - Input dataframe with cleaned up text column. 
                                      If text column is empty after cleaning then row is dropped.
    '''
    #Copy to not modify original df object
    clean_df = df.copy(deep=True)
    
    #Remove wordle result emojis
    clean_df['text'] = clean_df['text'].str.replace('🟩', '')
    clean_df['text'] = clean_df['text'].str.replace('⬛', '')
    clean_df['text'] = clean_df['text'].str.replace('🟨', '')
    clean_df['text'] = clean_df['text'].str.replace('⬜', '')
    clean_df['text'] = clean_df['text'].str.replace('🟧', '')
    clean_df['text'] = clean_df['text'].str.replace('🟦', '')
    clean_df['text'] = clean_df['text'].str.replace('🟥', '')
    clean_df['text'] = clean_df['text'].str.replace('🟪', '')
    clean_df['text'] = clean_df['text'].str.replace('🟫', '')

    #Remove standard text when people shrae results
    clean_df['text'] = clean_df['text'].str.replace('Wordle\s\(ES\)\s#\d+\s\d\/\d', '', regex=True)
    clean_df['text'] = clean_df['text'].str.replace('Wordle\s\d+\s\d\/\d', '', regex=True)

    #Substitute links with 'http' and twitter user handles with '@user', as model trained using this
    clean_df['text'] = clean_df['text'].str.replace('((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', 'http', regex=True) 
    clean_df['text'] = clean_df['text'].str.replace('@[^\s]+', '@user', regex=True)

    #Drop rows with empty text column
    for index, row in clean_df.iterrows():
        #drop empty text fields
        if re.search('$^', row['text']):
            clean_df = clean_df.drop([index])

    return clean_df

def main():
    #==================Collect data==============================
    #Download tweets using twitter API full archive search (available under academic license)
    client = tweepy.Client(bearer_token=config.academic_bearer_token, wait_on_rate_limit=True)

    query = 'wordle place_country:GB -is:retweet -has:media -is:nullcast lang:en'
    start_time = '2021-12-01T00:00:00Z'
    end_time = '2022-03-15T00:00:00Z'
    tweet_fields=['text','created_at','public_metrics']

    tweets_3month=[]
    for tweet in tweepy.Paginator(client.search_all_tweets, query=query, max_results=500, start_time = start_time, end_time=end_time, tweet_fields = tweet_fields).flatten(limit=1000000):
        tweets_3month.append(tweet)

    #Convert tweet data to dataframe
    df_3month = convert_tweepy_data_to_dataframe(tweets_3month)
    df_3month.to_csv('wordle_3month.csv', index=False)


    #=============Sentiment analysis using roBERTa model trained on twitter data===============
    #Clean data
    clean_df = clean_text(df_3month)

    #Model of choice
    roberta = 'cardiffnlp/twitter-roberta-base-sentiment'

    #Calculate sentiment
    clean_df['sentiment'] = clean_df.text.apply(get_BERT_sentiment, model=roberta) 
    clean_df['weighted_sentiment'] = clean_df.apply(get_weighted_sentiment, axis=1)
    clean_df['datetime']= pd.to_datetime(clean_df.datetime)
    clean_df['date'] = clean_df['datetime'].dt.strftime('%Y-%m-%d')

    #Group by day and calculate volume and aggregate sentiment values
    result_df = clean_df.groupby('date').agg({'text':'count', 
                                            'sentiment':'mean', 
                                            'weighted_sentiment':'mean'})
    result_df.to_csv('results_3month_BERT.csv')


    #==================Visualize results===========================================
    fig = plt.figure()

    plt.subplot(2,1,1)
    plt.xticks(rotation='vertical')
    plt.xlabel('date')
    plt.ylabel('mean sentiment')
    plt.bar(result_df['date'], result_df['sentiment'])

    plt.subplot(2,1,2)
    plt.xticks(rotation='vertical')
    plt.xlabel('date')
    plt.ylabel('volume')
    plt.bar(result_df['date'], result_df['text'])

    plt.show()

if __name__ == '__main__':
    main()