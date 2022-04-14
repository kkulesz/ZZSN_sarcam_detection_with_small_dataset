import pandas as pd
import re
import preprocessor as p

import html
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer


test_or_train = 'train'
input_file = f'data/{test_or_train}_downloaded.csv'
output_file = f'data/{test_or_train}_preprocessed.csv'

if __name__ == '__main__':
    df = pd.read_csv(input_file)
    tweets = df['text'].astype('string')

    # TODO: hashtags probably should not be removed
    # remove hashtags, urls, mentions, emojis etc.
    for i, v in enumerate(tweets):
        tweets.loc[i] = p.clean(v)

    # remove digits
    tweets = tweets.replace('\d+', '')

    # make everything lowercase
    tweets = tweets.str.lower()

    # remove punctuation
    for i, v in enumerate(tweets):
        tweets.loc[i] = re.sub(r'[^\w\s]', '', v)

    # remove unnecessary whitespaces and newlines
    for i, v in enumerate(tweets):
        tweets.loc[i] = re.sub(" +", " ", v)
        tweets.loc[i] = re.sub("\n", " ", v)
        tweets.loc[i] = re.sub("\t", " ", v)

    # remove empty tweets
    # tweets = tweets.drop(tweets[tweets == ""].index)

    # tokens = []
    # tokenizer = TweetTokenizer()
    # print(tweets.head())
    # for i, v in enumerate(tweets):
    #     tokens.append(tokenizer.tokenize(v))

    # nltk.download('stopwords')
    # stop_words = set(stopwords.words('english'))
    # print(stop_words)

    # df['preprocessed'] = tweets
    # df.to_csv(output_file)
