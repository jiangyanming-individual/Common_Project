import numpy as np
import pandas as pd
import math
Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'

#61种密码子：
myCodons = {
    'A': 4,
    'C': 2,
    'D': 2,
    'E': 2,
    'F': 2,
    'G': 4,
    'H': 2,
    'I': 3,
    'K': 2,
    'L': 6,
    'M': 1,
    'N': 2,
    'P': 4,
    'Q': 2,
    'R': 6,
    'S': 6,
    'T': 4,
    'V': 4,
    'W': 1,
    'Y': 2,
    'X': 1
}

#X 算不算一个密码子？
def DDE_encoding(filepath):

    data_list = []

    with open(filepath, encoding='utf-8') as f:

        for line in f.readlines():
            sequence, label = list(line.strip('\n').split(','))
            data_list.append((sequence, label))

    result_seq_data = []
    result_seq_labels = []

    # 二肽组成：
    diPeptides = [aa1 + aa2 for aa1 in Amino_acid_sequence for aa2 in Amino_acid_sequence]

    # TM
    myTM = []
    for pair in diPeptides:
        myTM.append((myCodons[pair[0]] / 62) * (myCodons[pair[1]] / 62))

    Amino_acid_Dict = {}
    for i in range(len(Amino_acid_sequence)):
        Amino_acid_Dict[Amino_acid_sequence[i]] = i

    for seq,label in data_list:
        one_seq=[]
        #二肽种类
        myDC = [0] * 441
        # 肽的长度-1 ，因为最后一个残基不能构成二肽：
        for j in range(len(seq) - 1):
            myDC[Amino_acid_Dict[seq[j]] * 21 + Amino_acid_Dict[seq[j + 1]]] = myDC[Amino_acid_Dict[seq[j]] * 21 + Amino_acid_Dict[seq[j + 1]]] + 1

        #计算概率
        if sum(myDC)!=0:
            myDC=[i/sum(myDC) for i in myDC]

        # TV
        myTV = []
        for k in range(len(myTM)):
            myTV.append(myTM[k] * (1 - myTM[k]) / (len(seq) - 1))

        # DDE
        for j in range(len(myDC)):
            myDC[j] = (myDC[j] - myTM[j]) / math.sqrt(myTV[j])
        # print(myDC)
        one_seq=one_seq + myDC
        result_seq_data.append(one_seq)
        result_seq_labels.append(int(label))

    # print(result_seq_data)
    return np.array(result_seq_data),np.array(result_seq_labels,dtype=np.int32)


if __name__ == '__main__':


    train_filepath = '../../Datasets/Khib_train.csv'
    test_filepath = '../../Datasets/Khib_test.csv'
    seq_data, labels = DDE_encoding(test_filepath)

    print(seq_data.shape)