import numpy as np
import pandas as pd
from collections import Counter

def get_EGAAC_encoding(filepath):
    data_list = []

    with open(filepath, encoding='utf-8') as f:

        for line in f.readlines():
            sequence, label = list(line.strip('\n').split(','))
            data_list.append((sequence, label))

    result_seq_data = []
    result_seq_labels = []

    group = {
        'Aliphatic group': 'GAVLMI',
        'Aromatic groups': 'FYW',
        'Positively charged groups': 'KRH',
        'Negatively charged groups': 'DE',
        'No charge group': 'STCPNQ'
    }

    groupKeys = group.keys()

    for seq,label in data_list:
        one_seq=[]
        groupCount_dict = {}
        counter=Counter(seq)
        for key in groupKeys:
            for aa in group[key]:
                groupCount_dict[key]=groupCount_dict.get(key,0)+counter[aa]

        for key in groupKeys:
            one_seq.append(round(groupCount_dict[key] / len(seq), 3))
        result_seq_data.append(one_seq)
        result_seq_labels.append(int(label))

    return np.array(result_seq_data),np.array(result_seq_labels)

if __name__ == '__main__':

    train_filepath='../../Datasets/Khib_train.csv'
    test_filepath='../../Datasets/Khib_test.csv'
    seq_data,labels=get_EGAAC_encoding(test_filepath)

    print(seq_data.shape)