import numpy as np
Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'

def BE_encoding(filepath):
    data_list = []

    with open(filepath, encoding='utf-8') as f:

        for line in f.readlines():
            sequence, label = list(line.strip('\n').split(','))
            data_list.append((sequence, label))

    result_seq_data = []
    result_seq_labels = []

    for data in data_list:
        seq,label=data[0],data[1]
        one_seq = []
        result_seq_labels.append(int(label))
        for amino_acid in seq:
            one_amino_acid = []
            for amino_acid_index in Amino_acid_sequence:
                if amino_acid_index == amino_acid:
                    flag = 1
                else:
                    flag = 0
                one_amino_acid.append(flag)
            one_seq.extend(one_amino_acid)
        result_seq_data.append(one_seq)
    return np.array(result_seq_data), np.array(result_seq_labels, dtype=np.int64)


if __name__ == '__main__':

    train_filepath='../../Datasets/Khib_train.csv'
    test_filepath='../../Datasets/Khib_test.csv'
    seq_data,labels=BE_encoding(train_filepath)

    print(seq_data.shape)