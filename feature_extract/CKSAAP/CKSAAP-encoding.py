import numpy as np
Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'



def create_empty_matrix():
    return np.zeros((21, 21))

def CKSAAP_encoding(filepath,k):

    data_list = []
    with open(file=filepath, mode='r',encoding='utf-8') as f:
        for line in f.readlines():
            sequence, label = list(line.strip('\n').split(','))
            data_list.append((sequence, label))

    Amino_acid_Dict = {}
    for i in range(len(Amino_acid_sequence)):
        Amino_acid_Dict[Amino_acid_sequence[i]] = i

    #400
    diamino_acid_pair=[aa1 + aa2 for aa1 in Amino_acid_sequence for aa2 in Amino_acid_sequence]
    print("diamino_acid_pair：",diamino_acid_pair)

    result_seq_data = []
    result_seq_labels = []
    for data in data_list:

        seq, label = data[0], data[1]
        result_seq_labels.append(int(label))
        peptein_length=len(seq)

        one_seq=[]
        for k_index in range(k+1):
            # print("k:",k_index)
            N_total=peptein_length - k_index -1
            myDC=[0]*441
            for i in range(peptein_length):
                myDC[Amino_acid_Dict[seq[i]] * 21 + Amino_acid_Dict[seq[i+k_index+1]]] = myDC[Amino_acid_Dict[seq[i]] * 21 + Amino_acid_Dict[seq[i+k_index+1]]] +1

                if i + k_index + 1 == (peptein_length - 1):
                    break

            #calculate vector
            if sum(myDC)!=0:
                myDC=[ j/N_total for j in myDC]

            one_seq.extend(myDC)
        result_seq_data.append(one_seq)

    result_data=np.array(result_seq_data)
    result_data=np.reshape(result_data, (-1, k+1, 441)) #reshape

    return result_data,np.array(result_seq_labels, dtype=np.int64)


if __name__ == '__main__':

    train_filepath = '../../Datasets/Khib_train.csv'
    test_filepath = '../../Datasets/Khib_test.csv'
    encoded_vector,labels = CKSAAP_encoding(test_filepath, 5)
    print("CKSAAP-encoded vector shape:", encoded_vector.shape)