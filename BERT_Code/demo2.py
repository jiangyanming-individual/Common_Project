import torch
from torch import nn
from d2l import torch as d2l

class BERT_Encoder(nn.Module):


    #num_heads == d_model
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,

                 max_len=1000,key_size=768,query_size=768,value_size=768, **kwargs):
        super(BERT_Encoder,self).__init__()

        self.token_embedding=nn.Embedding(vocab_size,num_hiddens)
        self.segment_embedding=nn.Embedding(2,num_hiddens)

        # encoder block
        self.blk=nn.Sequential()

        for i in range(num_layers):
            self.blk.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))

        self.positional_embedding=nn.Parameter(torch.randn(1,max_len,num_hiddens))

    def forward(self,tokens,segmens,valid_lens):

        X=self.token_embedding(tokens) + self.segment_embedding(segmens)
        X=X + self.positional_embedding.data[:,:X.shape[1],:]
        for blk in self.blk:
            X=blk(X,valid_lens)

        return X



class MaskLM(nn.Module):

    def __init__(self,vocab_size,num_hiddens,num_inputs=768,**kwargs):
        super(MaskLM, self).__init__()
        self.mlp=nn.Sequential(
            nn.Linear(num_inputs,num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens,vocab_size)
        )

    def forward(self,X,pred_positions):

        num_pred_positions=pred_positions.shape[1]
        pred_positions=pred_positions.reshape(-1)

        batch_size=X.shape[0]
        batch_idx=torch.arange(0,batch_size)


        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）, 复制3次
        batch_idx=torch.repeat_interleave(batch_idx,num_pred_positions)
        masked_X=X[batch_idx,pred_positions]
        masked_X=masked_X.reshape((batch_size,num_pred_positions,-1))

        mlm_Y_hat=self.mlp(masked_X)

        return mlm_Y_hat

class NextSentencePred(nn.Module):

    def __init__(self,num_inputs,**kwargs):

        super(NextSentencePred, self).__init__()
        self.output=nn.Linear(num_inputs,2)


    def forward(self,X):

        return self.output(X)


class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERT_Encoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)

        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :])) # cls是预测
        return encoded_X, mlm_Y_hat, nsp_Y_hat