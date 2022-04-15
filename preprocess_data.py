import pandas as pd
import re
import preprocessor as p

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

test_or_train = 'train'
input_file = f'data/{test_or_train}_downloaded.csv'
output_file = f'data/{test_or_train}_ready.csv'
"""
    TODO:
        1. decide what to do with hashtags:
            - remove them?
            - if no - should the '#' be present? how to tokenize it?
        2. set limit of tokens in single row?
        3. padding?
"""
p.set_options(
    p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.SMILEY, p.OPT.NUMBER, p.OPT.RESERVED
)  # remove everything instead of hashtags

if __name__ == '__main__':
    df = pd.read_csv(input_file)
    df['text'] = df['text'].astype('string')

    # remove urls, mentions, emojis etc.
    for i, tweet_text in enumerate(df['text']):
        df['text'].loc[i] = p.clean(tweet_text)

    # remove digits
    df['text'] = df['text'].replace('\d+', '')

    # make everything lowercase
    df['text'] = df['text'].str.lower()

    # remove punctuation
    for i, tweet_text in enumerate(df['text']):
        df['text'].loc[i] = re.sub(r'[^\w\s#]', '', tweet_text)  # DO NOT REMOVE HASH

    # remove unnecessary whitespaces and newlines
    for i, tweet_text in enumerate(df['text']):
        df['text'].loc[i] = re.sub(" +", " ", tweet_text)
        df['text'].loc[i] = re.sub("\n", " ", df['text'].loc[i])
        df['text'].loc[i] = re.sub("\t", " ", df['text'].loc[i])

    # remove empty tweets
    df = df.drop(df['text'][df['text'] == ""].index)

    # tokenize
    df['tokens'] = [[]] * len(df)
    tokenizer = TweetTokenizer()
    pd.options.mode.chained_assignment = None  # default='warn'
    for i, tweet_text in enumerate(df['text']):
        df['tokens'].loc[i] = tokenizer.tokenize(tweet_text)

    # remove english stop words eg. 'a', 'the', ...
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    for i, tweet_tokens in enumerate(df['tokens']):
        df['tokens'].loc[i] = [token for token in tweet_tokens if token not in stop_words]

    # set labels as zeros and ones
    for i, label in enumerate(df['sarcasm_label']):
        df['sarcasm_label'].loc[i] = 0 if label == 'not_sarcastic' else 1

    # remove tweets without any token
    df = df[df['tokens'].map(lambda r: len(r)) > 0]

    df = df[['sarcasm_label', 'tokens']]
    df.to_csv(output_file)
