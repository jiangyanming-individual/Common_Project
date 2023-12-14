import pandas as pd


def calculate_number(filepath):

    pd_file=pd.read_csv(filepath)

    neg_numbers=pd_file['1'].value_counts()[0]
    pos_numbers=pd_file['1'].value_counts()[1] + 1

    # print(pd_file['1'].value_counts())
    # print("neg:",pd_file['1'].value_counts()[0])
    # print("pos:",pd_file['1'].value_counts()[1])

    total_numbers=pos_numbers + neg_numbers
    return total_numbers,pos_numbers,neg_numbers

if __name__ == '__main__':

    train_filepath = '../../Datasets/Khib_train.csv'
    test_filepath = '../../Datasets/Khib_test.csv'

    total_numbers,pos_numbers,neg_numbers=calculate_number(train_filepath)