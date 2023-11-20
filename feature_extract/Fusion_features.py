import numpy as np
import pandas as pd
import math
Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'


result_AAindex = pd.read_csv('./AAindex/AAindex_12.csv')
pro_name_list = result_AAindex['AccNo'].tolist()
AAindex_dict = {}
for aa in Amino_acid_sequence:
    if aa == 'X':
        AAindex_dict['X'] = [0] * 12
        continue
    AAindex_dict[aa] = result_AAindex[aa].tolist()


# one sequence
def get_BE_encoding(one_sequence):

    one_seq=[]
    for amino_acid in one_sequence:
        one_amino_acid = []
        for amino_acid_index in Amino_acid_sequence:
            if amino_acid_index == amino_acid:
                flag = 1
            else:
                flag = 0
            one_amino_acid.append(flag)
        one_seq.extend(one_amino_acid)

    return one_seq



def get_AAindex_encoding(one_sequence):


    one_seq = []
    for aa in one_sequence:
        one_seq.extend(AAindex_dict.get(aa))
    return one_seq



def get_BLOSUM62_encoding(one_sequence):

    #21 * 20
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
        'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # X
    }


    one_seq=[]
    for aa in one_sequence:
        # print("aa:",aa)
        one_seq.extend(blosum62.get(aa))
    return one_seq

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

def DDE_encoding(data_list):

    data_list = []

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



def fusion_features(filepath):

    data_frame=pd.read_csv(filepath,header=None)
    data_columns=data_frame.columns.tolist()

    sequences=data_frame[0].tolist()
    labels=data_frame[1].tolist()

    data_list=[]
    for item in zip(sequences,labels):
        #(seq,label)
        data_list.append(item)
    # print(data_list)

    seqs_encodings = []
    lables_encodings = []

    encodings=[]
    for seq, label in data_list:

        code=[seq,label]

        one_seq_list=[]


        BLOSUM62_encoding=get_BLOSUM62_encoding(seq)
        AAindex_encoding=get_AAindex_encoding(seq)
        BE_encoding=get_BE_encoding(seq)

        one_seq_list.extend(BLOSUM62_encoding)
        one_seq_list.extend(AAindex_encoding)
        one_seq_list.extend(BE_encoding)

        code.extend(one_seq_list)

        lables_encodings.append(int(label))

        # print("one_seq_list:",one_seq_list)
        # res=np.array(one_seq_list)
        # print(res.shape) #(1963,1)
        seqs_encodings.append(one_seq_list)
        encodings.append(code)

    #save csv file
    result_encoding=pd.DataFrame(encodings)
    result_encoding.to_csv('./train_example_features.csv',header=None)

    result_seqs=np.array(seqs_encodings).reshape((-1,37,53))
    return result_seqs,np.array(lables_encodings,dtype=np.int32)


if __name__ == '__main__':


    train_filepath = '../Datasets/Khib_train.csv'
    test_filepath = '../Datasets/Khib_test.csv'
    seqs_data,labels_data=fusion_features(test_filepath)

    print(seqs_data.shape)
    print(labels_data.shape)
