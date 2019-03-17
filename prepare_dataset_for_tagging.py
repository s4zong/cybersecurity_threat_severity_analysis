

from utils.io import *

## read download text
downloaded_tweets = readTSVFile('./dataset/downloaded.tsv')
print('num lines', len(downloaded_tweets))

downloaded_tweets_dict = {}
for each_tweet in downloaded_tweets:
    downloaded_tweets_dict[each_tweet[0]] = each_tweet[-1].replace('…', ' ')

## filling
dataset = readJSONFile('./dataset/dataset_template.json')
for each_tweet in dataset:
    each_tweet['text'] = downloaded_tweets_dict[each_tweet['id']]

## write output
writeJSONLine('./dataset/dataset.json', dataset, verbose=True)



