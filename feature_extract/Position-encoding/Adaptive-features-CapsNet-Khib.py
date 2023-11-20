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
import Count_samples
import warnings
warnings.filterwarnings("ignore")


Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'
class EarlyStopping:

    def __init__(self, save_path,patience=5, delta=0):

        self.petience = patience
        self.delta = delta
        self.counter = 0

        self.best_score = None
        self.early_stop = False
        self.save_path=save_path
        self.val_loss_min = np.Inf

    def __call__(self,model,val_loss):

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss,model)

        elif val_loss > self.best_score + self.delta:
            self.counter += 1

            if self.counter > self.petience:
                self.early_stop = True
        else:
            #the min val_loss
            self.best_score = val_loss
            self.save_checkpoint(val_loss,model)
            self.counter = 0

    #save model
    def save_checkpoint(self,val_loss,model):

        path=os.path.join(self.save_path)
        torch.save(model.state_dict(),path)
        self.val_loss_min=val_loss

train_filepath= '../../Datasets/Khib_train.csv'
test_filepath= '../../Datasets/Khib_test.csv'


dict_filepath='../../Datasets/residue2idx.pkl'
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


class Length(nn.Module):

    def __init__(self):
        super(Length, self).__init__()

    def forward(self,inputs):

        return torch.sqrt(torch.sum(torch.square(inputs),-1))

def squash(inputs,axis=-1):
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)

    return scale * inputs


class PrimaryCaps(nn.Module):

    def __init__(self,in_channels,out_channels,dim_vector, kernel_size, stride=1,padding=0):

        #out_channels= 16; dim_vector=8
        super(PrimaryCaps, self).__init__()

        # self.dim_vector=dim_vector #input vector
        #out_channels=>128
        self.conv1d=nn.Conv1d(in_channels=in_channels,out_channels=out_channels * dim_vector,kernel_size=kernel_size,stride=stride,padding=padding)

        self.dim_vector=dim_vector

    def forward(self,x):

        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: num_capsules * (batch_size, out_channels, height, weight)
        outputs=self.conv1d(x)
        batch_size=outputs.size(0)
        # Flatten out: (batch_size, num_capsules * height * weight, out_channels)
        # reshape:
        outputs=outputs.view(batch_size,-1,self.dim_vector)#[batch_size,num_caps,dim_vector]==>(batch,272,8)

        return squash(outputs)

class DigitCaps(nn.Module):

    def __init__(self,in_num_caps,in_dim_caps,out_num_caps,out_dim_caps,routing=3):

        super(DigitCaps, self).__init__()

        self.in_num_caps=in_num_caps #number of input capsule if digit layer 368
        self.in_dim_cpas=in_dim_caps # 8


        #DigitCaps==>[2,8]
        self.out_num_caps=out_num_caps # numbers of capsule in the capsule layer==> 2 classes
        self.out_dim_caps=out_dim_caps # 8

        self.routing=routing
        self.device=device
        #(1,2,272,8,8)
        self.weight=nn.Parameter(0.01 * torch.randn(1,out_num_caps,in_num_caps,out_dim_caps,in_dim_caps,requires_grad=True))

    def forward(self,x):


        batch_size=x.size(0)  #[400,8]
        # (batch_size, in_num_caps, in_dim_caps) -> (batch_size, 1, in_num_caps, in_dim_caps, 1)
        x=x.unsqueeze(1).unsqueeze(4)

        # (1, out_num_caps, in_num_caps, out_dim_caps, in_dim_caps) @ (batch_size, 1, in_num_caps, in_dim_caps, 1) =
        # (batch_size, out_num_caps, in_num_caps, out_dim_caps, 1) ==>(batch,2,272,8, 1) ???
        u_hat=torch.matmul(self.weight,x)

        u_hat=u_hat.squeeze(-1) # (batch,2,272,8) (batch_size, out_num_caps, in_num_caps, out_dim_caps)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat=u_hat.detach()#(batch_size, out_num_caps, in_num_caps, out_dim_caps)

        b=torch.zeros(batch_size,self.out_num_caps,self.in_num_caps,1).to(device) #[batch,2,272]

        for i in range(self.routing-1):

            c= b.softmax(dim=1)

            # element-wise multiplication
            # (batch_size, out_num_caps, in_num_caps, 1) * (batch_size, out_num_caps, in_num_caps, out_dim_caps) ->
            # (batch_size, out_num_caps, in_nmm_caps, out_dim_caps) sum across in_caps ->
            # (batch_size, out_num_caps, out_dim_caps) ==> 2 * 8

            s=(c * temp_u_hat).sum(dim=2)

            v=squash(s)#(batch_size, out_num_caps, out_dim_caps)

            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, out_num_caps, in_num_caps, out_dim_caps) @ (batch_size, out_num_caps, out_dim_caps, 1)
            # -> (batch_size, out_num_caps, in_num_caps, 1)
            uv=torch.matmul(temp_u_hat,v.unsqueeze(-1))
            b+=uv

        c=b.softmax(dim=1)

        s=(c * u_hat).sum(dim=2)
        v=squash(s)

        return v


class Self_Attention(nn.Module):

    def __init__(self, hidden_size, dropout_rate=0.0):
        super(Self_Attention, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout_rate)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        query = self.query(x).view(batch_size, seq_len, self.hidden_size)
        key = self.key(x).view(batch_size, seq_len, self.hidden_size)
        value = self.value(x).view(batch_size, seq_len, self.hidden_size)

        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)

        #attention score
        atten_scores = torch.bmm(query, key.transpose(1, 2)) #===>[bacth_size,seq_len,seq_len]
        atten_scores = F.softmax(atten_scores, dim=-1)

        atten_output = torch.bmm(atten_scores, value)

        return atten_output


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
        # print(x.shape)
        # print(self.pe[:x.size(0),:].shape)
        x=x+self.pe[:x.size(0),:]

        # print("x shape:",x.shape)
        return self.dropout(x)



class CapsNet(nn.Module):

    def __init__(self,vocab_size,d_model,input_size,num_classes,num_routing,is_used_Position_encode=True):

        super(CapsNet,self).__init__()

        self.is_used_pos=is_used_Position_encode

        self.vocab_size=vocab_size
        self.d_model=d_model

        self.input_size = input_size
        self.num_classes = num_classes
        self.routings = num_routing

        self.in_channel=input_size[2]

        self.src_embedding=nn.Embedding(self.vocab_size,self.d_model)
        self.pos_embedding=PositionalEncoding(self.d_model)

        #conv layer
        self.conv = nn.Conv1d(in_channels=self.in_channel, out_channels=32, kernel_size=7, stride=1, padding=0)

        self.attention2=Self_Attention(hidden_size=8,dropout_rate=0)
        self.relu=nn.ReLU(inplace=True)
        #primary capsule
        self.primary_caps=PrimaryCaps(in_channels=32,out_channels=16,dim_vector=8,kernel_size=7,stride=1,padding=0)
        #digit capsule
        #squence length =25
        self.digit_caps=DigitCaps(in_num_caps=25 * 16,in_dim_caps=8,out_num_caps=num_classes,out_dim_caps=8)

        self.length=Length()

        self.dropout1=nn.Dropout(0.7)
        self.dropout2=nn.Dropout(0.2)

    def forward(self,x):


        # use position encode:
        if self.is_used_pos:
            src_emb_out = self.src_embedding(x)
            x = self.pos_embedding(src_emb_out.transpose(0, 1)).transpose(0,1)  # (batch_size,seq_len,d_model)

        x = torch.permute(x, [0, 2, 1])
        # print("init x shape:",x.shape)
        out=self.relu(self.conv(x))
        out=self.dropout1(out)
        # print("first out shape:",out.shape)

        # out=self.dropout2(out)
        out=self.primary_caps(out)
        out=self.dropout2(out)
        # print("seconde out shape:",out.shape)

        out=self.digit_caps(out)
        # print("third out shape:",out.shape)
        out=self.attention2(out)

        # out=self.length(out)
        # print("final out shape:",out.shape)
        return out



class BiLSTM_Model(nn.Module):

    def __init__(self,vocab_size,d_model,hidden_size,num_classes=2,num_layers=1):
        super(BiLSTM_Model, self).__init__()


        self.vocab_size=vocab_size
        self.input_size=d_model
        self.d_model=d_model
        self.hidden_size=hidden_size
        self.num_classes=num_classes
        self.num_layers=num_layers

        self.src_embedding=nn.Embedding(self.vocab_size,d_model)
        self.pos_embedding=PositionalEncoding(self.d_model)

        self.BiLSTM_layer=nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout=nn.Dropout(0.5)

    def forward(self,X):

        src_emb_out = self.src_embedding(X)
        X= self.pos_embedding(src_emb_out.transpose(0, 1)).transpose(0, 1)  # (batch_size,seq_len,d_model)
        BiLSTM_output,(h,c)=self.BiLSTM_layer(X)
        output=self.dropout(BiLSTM_output)

        return output

class BiLSTM_CapsNet_Model(nn.Module):

    def __init__(self,vocab_size,d_model,Bi_hidden_size,Bi_num_classes,Bi_num_layers):

        super(BiLSTM_CapsNet_Model, self).__init__()

        self.d_model=d_model
        self.vocab_size=vocab_size
        self.hidden_size=Bi_hidden_size
        self.num_classes=Bi_num_classes
        self.num_layers=Bi_num_layers

        self.BiLSTM_Layer=BiLSTM_Model(
                                       vocab_size=self.vocab_size,
                                       d_model=self.d_model,
                                       hidden_size=self.hidden_size,
                                       num_classes=self.num_classes,
                                       num_layers=self.num_layers)

        self.attention=Self_Attention(hidden_size=hidden_size * 2,dropout_rate=0.)
        self.CapsNet_layer1=CapsNet(vocab_size=self.vocab_size,d_model=self.d_model,input_size=[1, 37, self.d_model], num_classes=self.num_classes, num_routing=3)
        self.CapsNet_layer2 = CapsNet(vocab_size=self.vocab_size,d_model=self.d_model,input_size=[1, 37,self.hidden_size * 2], num_classes=self.num_classes, num_routing=3,is_used_Position_encode=False)

        self.length = Length()
        self.flatten_layer=nn.Flatten()
        self.output_layer=nn.Linear(in_features=32,out_features=2)


    def forward(self,X):

        inputs=X

        CapsNet_out1=self.CapsNet_layer1(X) #torch.Size([128, 2, 8])
        # print("CapsNet_out1 shape:",CapsNet_out1.shape)
        BiLSTM_out=self.BiLSTM_Layer(inputs) #[batch_size,seq_len,hidden_size * 2]

        Attention_out=self.attention(BiLSTM_out)
        CapsNet_out2=self.CapsNet_layer2(Attention_out)#torch.Size([128, 2, 8])

        Cancate_out = torch.cat([CapsNet_out1, CapsNet_out2, ], dim=-1)
        flatten_out=self.flatten_layer(Cancate_out)
        Final_out = self.output_layer(flatten_out)

        return (inputs,CapsNet_out1,CapsNet_out2,flatten_out),Final_out


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


def train(model,epoch,train_loader,optimizer,criterion,device):
    epoch_loss = []
    epoch_acc = []
    epoch_auc = []

    # criterion = CapsuleLoss()
    train_loss=0.0
    model.train()
    for batch_id, data in enumerate(train_loader):

        x_data = data[0].to(device)
        y_data = data[1].to(device)
        y_data = torch.tensor(y_data, dtype=torch.long)

        _,y_predict = model(x_data)
        loss = criterion(y_predict,y_data)
        train_loss += loss.item() * x_data.size(0)

        acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),
                                     torch.argmax(y_predict, dim=1).detach().cpu().numpy())
        auc = metrics.roc_auc_score(y_data.detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

        epoch_loss.append(loss.detach().cpu().numpy())
        epoch_acc.append(acc)
        epoch_auc.append(auc)
        if (batch_id % 64 == 0):
            print("epoch is :{},batch_id is {},loss is {},acc is:{},auc is:{}".format(epoch, batch_id,
                                                                                      loss.detach().cpu().numpy(),
                                                                                      acc, auc))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # mean value
    avg_loss, avg_acc, avg_auc = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_auc)
    print("[train acc is:{}, loss is :{},auc is:{}]".format(avg_acc, avg_loss, avg_auc))

    #train loss
    train_loss /=len(train_loader.dataset)

    return train_loss


# validate function
def validate(model,test_loader,criterion,device):
    model.eval()
    with torch.no_grad():

        valid_acc = []
        valid_loss = []
        valid_auc = []

        y_true = []
        y_score = []
        y_predict_labels_list = []
        val_loss=0.0
        for batch_id, data in enumerate(test_loader):

            x_data = data[0].to(device)
            y_data = data[1].to(device)
            # y_data = torch.unsqueeze(y_data,dim=1)
            y_data = torch.tensor(y_data, dtype=torch.long)

            # y_data_label = torch.eye(2).index_select(dim=0, index=data[1]).to(device)
            _,y_predict = model(x_data)

            # the max possible label
            y_predict_label = torch.argmax(y_predict, dim=1)
            y_predict_labels_list.append(y_predict_label.detach().cpu().numpy())
            # calculate loss value
            loss = criterion(y_predict,y_data)

            val_loss+=loss * x_data.size(0)
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

        y_test_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        y_pred_label = np.concatenate(y_predict_labels_list)

        fpr, tpr, _ = metrics.roc_curve(y_test_true, y_score)
        res_auc = metrics.auc(fpr, tpr)

        #confusion matrix
        (TN,TP,FN,FP),(SN,SP,ACC,MCC,Pr,recall,F1Score)=Calculate_confusion_matrix(y_test_true,y_pred_label)
        print('--------------------------------------validate---------------------------------------')
        print("[Valid TP is {},FP is {},TN is {},FN is {}]".format(TP, FP, TN, FN))
        print(
            "Valid : SN is {},SP is {},Pr is {},ACC is {},F1-score is {},MCC is {},\n AUC is {}, recall is {}".
            format(SN, SP, Pr, ACC, F1Score, MCC, res_auc, recall))
        print('---------------------------------------------------------------------------------------------')

        val_loss /=len(test_loader.dataset)
        return avg_loss,val_loss


def indepedent_test(model,test_loader,criterion,device):
    model.eval()
    with torch.no_grad():

        valid_acc = []
        valid_loss = []
        valid_auc = []

        y_true = []
        y_score = []
        y_predict_labels_list = []
        for batch_id, data in enumerate(test_loader):

            x_data = data[0].to(device)
            y_data = data[1].to(device)
            # y_data = torch.unsqueeze(y_data,dim=1)
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
        #confusion matrix
        (TN,TP,FN,FP),(SN,SP,ACC,MCC,Pr,recall,F1Score)=Calculate_confusion_matrix(y_test_true,y_pred_label)
        print('--------------------------------------independent test---------------------------------------')

        print("[Valid TP is {},FP is {},TN is {},FN is {}]".format(TP, FP, TN, FN))
        print(
            "Valid : SN is {},SP is {},Pr is {},ACC is {},F1-score is {},MCC is {},\n AUC is {}, recall is {}".
            format(SN, SP, Pr, ACC, F1Score, MCC, res_auc, recall))
        print('---------------------------------------------------------------------------------------------')


def total_train(model,train_loader,test_loader,optimizer,train_criterion,device,early_stopping):

    for epoch in range(epochs):

        train_loss=train(model,epoch,train_loader,optimizer,train_criterion,device)
        avg_loss,val_loss=validate(model,test_loader,criterion,device)
        print(f'Epoch: {epoch+1}/{epochs}, Train_loss: is {train_loss:.4f}, Avg_loss:is {avg_loss:.4f}, Val_loss: is {val_loss:.4f}')

        early_stopping(model,avg_loss) # use val_loss;
        if early_stopping.early_stop:
            print("Early Stopping!")
            break

if __name__ == '__main__':


    vocab_size = len(word2id_dict)
    d_model = 16
    epochs = 1000
    batch_size = 128
    learn_rate = 0.001
    hidden_size = 64
    num_classes = 2
    num_layers = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    # DataLoader:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)


    model = BiLSTM_CapsNet_Model(vocab_size,d_model,hidden_size,num_classes,num_layers)
    print(model)
    model.to(device)
    #optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate)

    total_numbers,pos_numbers,neg_numbers=Count_samples.calculate_number(train_filepath)
    # cal the ratio:
    class_weights=torch.tensor([total_numbers/neg_numbers,total_numbers/pos_numbers])

    # reweight method only use in training
    train_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion = nn.CrossEntropyLoss()

    #save model:
    save_model_path = '../DL_weights/CapsNet-Khib-Final_Weights.pth'
    #early stopping:
    early_stopping=EarlyStopping(save_path=save_model_path,patience=30)

    total_train(model,train_loader,test_loader, optimizer,train_criterion,device,early_stopping)
    # load model to test
    model_path = '../DL_weights/CapsNet-Khib-Final_Weights.pth'
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    )
    indepedent_test(model,test_loader,criterion,device)

