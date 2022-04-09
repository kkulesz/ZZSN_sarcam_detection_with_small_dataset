import pandas as pd

test_or_train = 'test'
file = f'data/{test_or_train}_ready.csv'

if __name__ == '__main__':
    df = pd.read_csv(file)
    print(df.iloc[4].text)
