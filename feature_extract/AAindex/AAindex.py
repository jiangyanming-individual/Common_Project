import pandas as pd
import numpy as np

Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'
def AAindex_encoding(filepath):

    result_AAindex = pd.read_csv('./AAindex_12.csv')
    print(result_AAindex['A'])
    pro_name_list = result_AAindex['AccNo'].tolist()

    data_list = []

    with open(filepath, encoding='utf-8') as f:

        for line in f.readlines():
            sequence, label = list(line.strip('\n').split(','))
            data_list.append((sequence, label))

    result_seq_data = []
    result_seq_labels = []

    AAindex_dict = {}
    for aa in Amino_acid_sequence:
        if aa == 'X':
            AAindex_dict['X'] = [0] * 12
            continue
        AAindex_dict[aa] = result_AAindex[aa].tolist()

    for seq, label in data_list:

        one_seq = []
        for aa in seq:
            one_seq.extend(AAindex_dict.get(aa))
        result_seq_data.append(one_seq)
        result_seq_labels.append(int(label))

    return np.array(result_seq_data), np.array(result_seq_labels, dtype=np.int64)

if __name__ == '__main__':

    train_filepath = '../../Datasets/Khib_train.csv'
    test_filepath = '../../Datasets/Khib_test.csv'
    seq_data,seq_labels=AAindex_encoding(test_filepath)

    print(seq_data.shape)