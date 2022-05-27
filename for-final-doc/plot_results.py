from matplotlib import pyplot as plt
import pandas as pd

bert = 'BERT_results.csv'
t5 = 'T5_results.csv'

if __name__ == '__main__':
    curr_model = t5
    results = pd.read_csv(curr_model)

    plt.plot(results['Learning dataset size'], results['f1 value'])
    plt.xlabel('Number of training rows')
    plt.ylabel('F1 score')
    plt.title(curr_model)
    f1_plot = plt.gcf()
    plt.show()
    plt.draw()
    f1_plot.savefig(f'{curr_model}-f1.png')

    plt.plot(results['Learning dataset size'], results['recall'])
    plt.xlabel('Number of training rows')
    plt.ylabel('Recall score')
    plt.title(curr_model)
    recall_plot = plt.gcf()
    plt.show()
    plt.draw()
    recall_plot.savefig(f'{curr_model}-recall.png')
