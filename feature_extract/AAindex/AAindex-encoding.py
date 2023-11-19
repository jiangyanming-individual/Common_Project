import numpy as np
import pandas as pd



def AAindex_encoding(filepath):
    data_list = []

    with open(filepath, encoding='utf-8') as f:

        for line in f.readlines():
            sequence, label = list(line.strip('\n').split(','))
            data_list.append((sequence, label))

    result_seq_data = []
    result_seq_labels = []

    with open('../features_encode/AAindex/AAindex_normalized.txt', mode='r') as f:
        records=f.readlines()[1:]
        f.close()

    Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'

    AAindex_names = []
    AAindex = []

    for i in records:
        AAindex_names.append(i.rstrip().split()[0] if i.rstrip() != '' else None)
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)

    props = 'FINA910104:LEVM760101:JACR890101:ZIMJ680104:RADA880108:JANJ780101:CHOC760102:NADH010102:KYTJ820101:NAKH900110:GUYH850101:EISD860102:HUTJ700103:OLSK800101:JURD980101:FAUJ830101:OOBM770101:GARJ730101:ROSM880102:RICJ880113:KIDA850101:KLEP840101:FASG760103:WILM950103:WOLS870103:COWR900101:KRIW790101:AURR980116:NAKH920108'.split(':')

    if props:
        tempAAindex_names = []
        tempAAindex = []
        for p in props:
            if AAindex_names.index(p) != -1:
                tempAAindex_names.append(p)
                tempAAindex.append(AAindex[AAindex_names.index(p)])

        if len(tempAAindex_names) != 0:
            AAindex_names = tempAAindex_names
            AAindex = tempAAindex

    seq_index = {}
    for i in range(len(AA)):
        seq_index[AA[i]] = i

    for seq,label in data:
        one_code=[]
        for aa in seq:
            if aa == 'X':
                for aaindex in AAindex:
                    one_code.append(0)
                continue
            for aaindex in AAindex:
                one_code.append(float(aaindex[seq_index.get(aa)]))
        X.append(one_code)
        y.append(int(label))

    X=np.array(X)
    n,seq_len=X.shape

    y=np.array(y)
    return X,y

if __name__ == '__main__':


    train_filepath='../../Datasets/Khib_train.csv'
    test_filepath='../../Datasets/Khib_test.csv'
    result=AAindex_encoding(test_filepath)