import pandas as pd
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('No entry file selected!')
        exit(0)

    df = pd.read_csv(sys.argv[1])
    print(df['sarcasm_type'].value_counts())
