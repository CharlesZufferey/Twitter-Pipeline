
##XYE6df2BnVqFDFb34il4tT8aF (API key)
##ARxt2nJERWKVJQVcueWJDsA2waNNLfaiL2J6dfGWzNGHcurqxz (API secret key)

##532241948-dkgYk3bB7ypnhpjVKsYHdbTzZTKFf5Q4jNODPgQ4 (Access token)

##jVOfRqJUhTlkxkBnQ5Eo4yq0UqvdNM00h9lxxYsT3OP4x (Access token secret)
#%%
import json
import tweepy
import logging
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pymongo
import pandas as pd

#%%
KEYWORDS = ['opinion']

consumer_key = 'XYE6df2BnVqFDFb34il4tT8aF'
consumer_secret = 'ARxt2nJERWKVJQVcueWJDsA2waNNLfaiL2J6dfGWzNGHcurqxz'
access_token = '532241948-dkgYk3bB7ypnhpjVKsYHdbTzZTKFf5Q4jNODPgQ4'
access_token_secret = 'jVOfRqJUhTlkxkBnQ5Eo4yq0UqvdNM00h9lxxYsT3OP4x'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)
user = api.me()


class StreamListener(tweepy.StreamListener):
    def __init__(self, limit, callback):
        super().__init__()
        self.limit = limit
        self.counter = 0
        self.callback = callback

    def on_error(self, status_code):
        '''stops stream when error 420 occurs'''
        if status_code == 420:
            return False

    def get_media(self, t):
        media_url = []
        if 'extended_tweet' in t and 'media' in t['extended_tweet']['entities']:
            for media in t['extended_tweet']['entities']['media']:
                media_url.append(media['media_url_https'])
                media_type = media['type']
        else:
            media_url = None
            media_type = ''
        return media_url, media_type

    def get_hashtags(self, t):
        hashtags = []
        if 'extended_tweet' in t:
            for hashtag in t['extended_tweet']['entities']['hashtags']:
                hashtags.append(hashtag['text'])
        elif 'hashtags' in t['entities'] and len(t['entities']['hashtags']) > 0:
            hashtags = [item['text'] for item in t['entities']['hashtags']]
        else:
            hashtags = []
        return hashtags

    def get_tweet_dict(self, t):
        '''extract information from the tweet'''
        if 'extended_tweet' in t:
            text = t['extended_tweet']['full_text']
        else:
            text = t['text']
        hashtags = self.get_hashtags(t)
        media_url, media_type = self.get_media(t)
        tweet = {'created_at': t['created_at'],
                 'id': t['id_str'],
                 'text': text,
                 'username': t['user']['screen_name'],
                 'followers':t['user']['followers_count'],
                 'user_favorites_count': t['user']['favourites_count'],
                 'retweets': t['retweet_count'],
                 'favorites': t['favorite_count'],
                 'hashtags': hashtags,
                 'media_url': media_url,
                 'media_type': media_type,
                 'interesting': 0}
        return tweet

    def on_data(self, data):
        '''collect, filter and parse the tweets from twitter API'''
        t = json.loads(data)
        if t['retweeted'] == False and 'RT' not in t['text'] and t['in_reply_to_status_id'] == None:
            tweet = self.get_tweet_dict(t)
            self.callback(tweet)
            self.counter += 1
            if self.counter == self.limit:
                return False


def get_tweets(limit, callback):
    stream_listener = StreamListener(limit, callback)
    stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
    stream.filter(track=KEYWORDS, languages=['en'])
#%%
txt = [] 

def show_text(tweet):
    #print(tweet['text'])
    logging.critical(tweet['text'])
    txt.append(tweet['text'])
    #r = txt.to_dict()
    return txt

    
#%%

#%%
txt.append('What i really like in you is your intelligence and your complete lack of self-awareness')
#%%
r = ()
if __name__ == '__main__':

    get_tweets(5, show_text)
    df = pd.DataFrame()
    ls = []
    for sentence in txt:
        p = (sentence)#,vs)
        ls.append(p)
    df = pd.DataFrame(ls) 
    df.columns =  ['Tweet']#,'Score'] 
    r = df.to_dict(orient='records')
    client = pymongo.MongoClient('mongodb:27017')
    db = client.tweets
    collection = db.tweets_data
    db.tweets_data.insert_many(r)

#%%    

#df = pd.DataFrame()
#ls = []
###analyzer = SentimentIntensityAnalyzer()
#for sentence in txt:
##    #vs = analyzer.polarity_scores(sentence)
##    #score = ("{:-<65} {}".format(sentence, str(vs)))
#    p = (sentence)#,vs)
#    ls.append(p)
#df = pd.DataFrame(ls) 
#df.columns =  ['Tweet']#,'Score'] 
##  
###%%
##r = {}
#r = df.to_dict(orient='records')
##r.to_csv(r'data/tweets.csv')
##
###%%
### connect to local MongoDB
##client = pymongo.MongoClient()#'mongodb:27017')
##db = client.tweets #use tweetw
#client = pymongo.MongoClient('mongodb:27017')
#db = client.tweets
#collection = db.tweets_data
#db.tweets_data.insert_many(r)
#%%
#%%


#%%


