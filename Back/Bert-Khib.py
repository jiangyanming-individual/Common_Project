
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from torch import nn
import os
import numpy as np
import pickle
import math
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
from sklearn import metrics
from torch.utils.data import DataLoader
from sklearn.metrics import auc,roc_auc_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")


Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'

train_filepath= '../Datasets/Khib_train.csv'
test_filepath= '../Datasets/Khib_ind_test.csv'

dict_filepath= '../Datasets/word_dict/residue2idx.pkl'
def get_word2id_dict(dict_filepath):

    residue2idx = pickle.load(open(dict_filepath ,'rb'))
    word2id_dict=residue2idx
    return word2id_dict
word2id_dict=get_word2id_dict(dict_filepath)


def load_data(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            sequence, y_label = line.strip().split(',')
            data.append((sequence, y_label))
    return data


train_dataset=load_data(train_filepath)
test_dataset=load_data(test_filepath)

# word to id
class MyDataset(Dataset):

    def __init__(self, examples, word2id_dict):
        super(MyDataset, self).__init__()
        self.word2id_dict = word2id_dict
        self.examples = self.words_to_id(examples)

    def words_to_id(self, examples):
        temp_example = []
        for i, example in enumerate(examples):

            seq, label = example
            seq = [self.word2id_dict.get(AA) for AA in seq]
            label = int(label)
            temp_example.append((seq, label))

        return temp_example

    def __getitem__(self, idx):

        seq, label = self.examples[idx]
        seq = np.array(seq).astype('int32')
        return seq, label

    def __len__(self):
        return len(self.examples)

#train dataset：
train_set=MyDataset(train_dataset,word2id_dict)
#ind_test：
test_set=MyDataset(test_dataset,word2id_dict)



class FNN(nn.Module):
    """
    FFN
    """
    def __init__(self,ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(FNN, self).__init__()
        self.dense1=nn.Linear(ffn_num_input,ffn_num_hiddens)
        self.relu=nn.ReLU()
        self.dense2=nn.Linear(ffn_num_hiddens,ffn_num_outputs)

    def forward(self,X):

        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """
    AddNorm
    """
    def __init__(self,normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.LayerNorm=nn.LayerNorm(normalized_shape)
    def forward(self,X,Y):

        return self.LayerNorm(X + self.dropout(Y))

class EncoderBlock(nn.Module):


    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = nn.MultiheadAttention(key_size, num_heads, dropout=dropout)
        self.addnorm1=AddNorm(norm_shape,dropout)
        self.ffn=FNN(ffn_num_input,ffn_num_hiddens,num_hiddens)
        self.addnorm2=AddNorm(norm_shape,dropout)

    def forward(self,x,mask=None):

        att_out,_=self.attention(x,x,x,mask)
        Y=self.addnorm1(x,att_out)
        # residue connect
        return self.addnorm2(Y,self.ffn(Y))

class PositionalEncoding(nn.Module):

    def __init__(self,d_model, dropout=0.2, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)

        pos_mat=torch.arange(0,max_len,dtype=torch.float32).unsqueeze(dim=1)
        i_mat=torch.pow(10000,torch.arange(0,d_model,2,dtype=torch.float32)/ d_model)

        pe[:,0::2] = torch.sin(pos_mat / i_mat)
        pe[:,1::2] = torch.cos(pos_mat / i_mat)

        pe=pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe',pe)

    def forward(self, x):

        # print(self.pe[:x.size(0),:].shape)
        x=x+self.pe[:x.size(0),:]
        # print("x shape:",x.shape)
        return self.dropout(x)


class EncoderModel(nn.Module):

    #num_heads == d_model
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,
                 num_heads,num_layers,dropout,key_size=768,query_size=768,value_size=768, num_classes=2,**kwargs):
        super(EncoderModel,self).__init__()

        # self.token_embedding=nn.Embedding(vocab_size,num_hiddens)
        # self.segment_embedding=nn.Embedding(2,num_hiddens)
        self.vocab_size=vocab_size
        self.d_model=num_hiddens

        self.token_embedding=nn.Embedding(self.vocab_size,self.d_model)
        self.positional_embedding=PositionalEncoding(self.d_model)

        # encoder block
        self.blk=nn.Sequential()
        for i in range(num_layers):
            self.blk.add_module( f"{i}",EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        self.fc=nn.Sequential(
            # nn.AdaptiveAvgPool1d(16),
            # nn.Linear(768,128),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(37 * num_hiddens,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,num_classes),
        )


    def forward(self,x,valid_lens=None):

        token_out = self.token_embedding(x)
        x = self.positional_embedding(token_out.transpose(0, 1)).transpose(0, 1)  # (batch_size,seq_len,d_model)
        for blk in self.blk:
            x=blk(x,valid_lens)
        out=self.fc(x)

        return x,out

def Calculate_confusion_matrix(y_test_true,y_pred_label):

    conf_matrix = confusion_matrix(y_test_true, y_pred_label)
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    Pr = TP / (TP + FP)
    recall = metrics.recall_score(y_test_true, y_pred_label)
    F1Score = (2 * TP) / float(2 * TP + FP + FN)

    return (TN,TP,FN,FP),(SN,SP,ACC,MCC,Pr,recall,F1Score)


def train(model,epochs,train_loader,optimizer,train_criterion,device):
    epoch_loss = []
    epoch_acc = []
    epoch_auc = []

    model.train()
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader):

            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            _,y_predict = model(x_data)
            loss = train_criterion(y_predict,y_data)
            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),
                                         torch.argmax(y_predict, dim=1).detach().cpu().numpy())
            auc = metrics.roc_auc_score(y_data.detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_acc.append(acc)
            epoch_auc.append(auc)
            if (batch_id % 64 == 0):
                print("epoch is :{},batch_id is {},loss is {},acc is:{},auc is:{}".format(epoch+1, batch_id,
                                                                                          loss.detach().cpu().numpy(),
                                                                                          acc, auc))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # mean value
        avg_loss, avg_acc, avg_auc = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_auc)
        print("[train acc is:{}, loss is :{},auc is:{}]".format(avg_acc, avg_loss, avg_auc))

        if (epoch + 1) == epochs:
            # save model
            torch.save(model.state_dict(), '../DL_weights/Bert-Khib.pth')

def indepedent_test(model,test_loader,criterion,device):
    model.eval()
    with torch.no_grad():

        valid_acc = []
        valid_loss = []
        valid_auc = []

        y_true = []
        y_score = []
        y_predict_labels_list = []

        roc = []
        roc_auc = []
        for batch_id, data in enumerate(test_loader):

            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            _,y_predict = model(x_data)

            # the max possible label
            y_predict_label = torch.argmax(y_predict, dim=1)
            y_predict_labels_list.append(y_predict_label.detach().cpu().numpy())
            # calculate loss value
            loss = criterion(y_predict,y_data)
            # calculate acc
            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),
                                         torch.argmax(y_predict, dim=1).detach().cpu().numpy())

            # calculate auc：
            auc = roc_auc_score(y_data[:].detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            valid_loss.append(loss.detach().cpu().numpy())
            valid_acc.append(acc)
            valid_auc.append(auc)

            y_true.append(y_data[:].detach().cpu().numpy())
            y_score.append(y_predict[:, 1].detach().cpu().numpy())

            if (batch_id % 64 == 0):
                print("batch_id is {},loss is {},acc is:{}, auc is {}".format(batch_id, loss.detach().cpu().numpy(),
                                                                              acc, auc))

        avg_acc, avg_loss, avg_auc = np.mean(valid_acc), np.mean(valid_loss), np.mean(valid_auc)
        print("[test acc is:{},loss is:{},auc is:{}]".format(avg_acc, avg_loss, avg_auc))

        #concate data
        y_test_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        y_pred_label = np.concatenate(y_predict_labels_list)

        fpr, tpr, _ = metrics.roc_curve(y_test_true, y_score)
        res_auc = metrics.auc(fpr, tpr)

        # np.save('../np_weights/CapsNet_Khib_y_test_true(test).npy', y_test_true)
        # np.save('../np_weights/CapsNet_Khib_y_test_score(test).npy', y_score)
        # np.save('../np_weights/CapsNet_Khib_y_test_pred(test).npy', y_pred_label)

        #confusion matrix
        (TN,TP,FN,FP),(SN,SP,ACC,MCC,Pr,recall,F1Score)=Calculate_confusion_matrix(y_test_true,y_pred_label)
        print('--------------------------------------independent test---------------------------------------')

        print("[Valid TP is {},FP is {},TN is {},FN is {}]".format(TP, FP, TN, FN))
        print(
            "Valid : SN is {},SP is {},Pr is {},ACC is {},F1-score is {},MCC is {},\n AUC is {}, recall is {}".
            format(SN, SP, Pr, ACC, F1Score, MCC, res_auc, recall))
        print('---------------------------------------------------------------------------------------------')



class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25,gamma=2,reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction=reduction
    def forward(self, input, target):
        # input:size is N * 2. N　is the batch　size,
        # target:size is N. N is the batch size
        #claculate passibility:
        eps=1e-7
        pt = torch.softmax(input, dim=1)
        #positive passibility:
        p = pt[:, 1]
        loss = -self.alpha* torch.pow((1-p),self.gamma) * (target * torch.log(p + eps)) - \
               (1 - self.alpha) * torch.pow(p,self.gamma) * ((1 - target) * torch.log(1 - p + eps))
        if self.reduction == 'sum':
            loss=loss.sum() # sum
        else:
            loss=loss.mean() # mean
        return loss

if __name__ == '__main__':

    vocab_size = len(word2id_dict)
    d_model = 16

    # the number of epochs is 1000
    epochs =50
    batch_size = 256
    learn_rate = 0.001

    # paramters :
    hidden_size = 64
    num_heads=8
    num_classes = 2
    num_layers = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    # to DataLoader:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)

    model = EncoderModel(vocab_size=vocab_size,num_hiddens=d_model,norm_shape=d_model,ffn_num_input=d_model,ffn_num_hiddens=d_model * 2,
                         num_heads=num_heads,num_layers=num_layers,dropout=0.5,key_size=d_model,query_size=d_model,
                         value_size=d_model,num_classes=num_classes)
    model.to(device)
    print(model)

    #optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate)
    #focal loss
    train_criterion = FocalLoss(alpha=0.65,gamma=1)
    criterion = nn.CrossEntropyLoss()

    #training:
    train(model,epochs,train_loader,optimizer,train_criterion,device)

    # load model to test
    model_path = '../DL_weights/Bert-Khib.pth'
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    indepedent_test(model,test_loader,criterion,device)
