import torch
from torch import nn




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



class BERTEncoder(nn.Module):

    #num_heads == d_model
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,

                 max_len=1000,key_size=768,query_size=768,value_size=768, **kwargs):
        super(BERTEncoder,self).__init__()

        self.token_embedding=nn.Embedding(vocab_size,num_hiddens)
        self.segment_embedding=nn.Embedding(2,num_hiddens)

        # encoder block
        self.blk=nn.Sequential()

        for i in range(num_layers):

            self.blk.add_module( f"{i}",EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))

        self.positional_embedding=nn.Parameter(torch.randn(1,max_len,num_hiddens))


    def forward(self,tokens,segmens,valid_lens):

        X=self.token_embedding(tokens) + self.segment_embedding(segmens)
        X=X + self.positional_embedding.data[:,:X.shape[1],:]
        for blk in self.blk:
            X=blk(X,valid_lens)

        return X

if __name__ == '__main__':

    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder=BERTEncoder(vocab_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout)

    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    encoded_X = encoder(tokens, segments, None)

    print(encoded_X.shape)