import pandas as pd
import pymongo

#%%
df = pd.read_csv('C:/Users/Charles/Downloads/pokemon.csv')
#%%
print(df)
# remove columns with dots and strange chars
#%%
for c in df:
    if '.' in c or '#' in c:
        del df[c]

# create a list of dictionaries
r = df.to_dict(orient='records')

# connect to local MongoDB
client = pymongo.MongoClient()
db = client.pokemon #use pokemon database

# write to a collection called pokemon_data
db.pokemon_data.insert_many(r)
#%%
# read
for x in db.pokemon_data.find({'Name': {'$in': ['Pikachu', 'Bulbasaur']}
}):
    print(x)

#%%
agr = [ {'$group': {'_id': "$Type 1", 'mean':{'$avg': "$Speed"}}}]
#%%
val = list(db.pokemon_data.aggregate([ {'$group': {'_id':
    "$Type 1", 'mean':{'$avg': "$Speed"}}}]))
print(val)
