import numpy as np
import pandas as pd
from collections import Counter


Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'

def get_AAC_encoding(filepath):
    data_list = []

    with open(filepath, encoding='utf-8') as f:

        for line in f.readlines():
            sequence, label = list(line.strip('\n').split(','))
            data_list.append((sequence, label))

    result_seq_data = []
    result_seq_labels = []

    for seq,label in data_list:
        one_seq=[]
        counter=Counter(seq)
        for key in counter:
            counter[key] = round(counter[key] / len(seq), 3)

        for amino_acid in Amino_acid_sequence:
            one_seq.append(counter[amino_acid])


        result_seq_data.append(one_seq)
        result_seq_labels.append(int(label))

    return np.array(result_seq_data),np.array(result_seq_labels)

if __name__ == '__main__':

    train_filepath = '../../Datasets/Khib_train.csv'
    test_filepath = '../../Datasets/Khib_test.csv'
    seq_data, labels = get_AAC_encoding(test_filepath)

    print(seq_data.shape)