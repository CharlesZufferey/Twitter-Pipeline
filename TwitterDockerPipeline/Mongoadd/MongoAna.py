# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:06:16 2020

@author: Charles
"""
import pymongo
#from sqlalchemy import create_engine
#import pandas as pd
#import psycopg2
#%%
client = pymongo.MongoClient('mongodb:27017')
db = client.tweets #use tweetw
db.tweets_data.find()

#df = pd.DataFrame(a)
#
#conns = f'postgres://postgres:postgres@localhost/postgres'
#engine = create_engine(conns, enconding = 'latin1',echo=False)
#df.to_sql('Tweets', engine)