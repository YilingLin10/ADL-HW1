from typing import Dict

import torch
from torch.nn import Embedding
from torch.autograd import Variable
import torch.nn.functional as F

from math import floor


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        max_len: int
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        # TODO: model architecture
        # embed.shape = ([4117, 300])
        self.embed_size = embeddings.size(dim=1)
        # LSTM layer
        self.lstm = torch.nn.LSTM(
            input_size=self.embed_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bidirectional=bidirectional,
            batch_first=True)
        # Conv2d layer
        # self.convolution = torch.nn.Conv2d(in_channels=1,
        #                             out_channels=64,
        #                             kernel_size=(1, 2*hidden_size))
        # Conv1d layer
        # self.convolution1 - output(batch_size, 100, sentence_len)
        self.convolution1 = torch.nn.Conv1d(in_channels=max_len,out_channels=hidden_size, kernel_size=1)
        # self.convolution2 = torch.nn.Conv1d(in_channels=max_len,out_channels=256, kernel_size=)
        # self.convolution3 = torch.nn.Conv1d(in_channels=max_len,out_channels=150, kernel_size=4)
        # fully-connected layer (bidirectional --> *2)
        self.dense = torch.nn.Linear(self.hidden_size, self.num_class)
        # dropout
        self.dropout = torch.nn.Dropout(self.dropout)
        

    # @property
    # def encoder_output_size(self) -> int:
    #     # TODO: calculate the output dimension of rnn
    #     raise NotImplementedError

    def forward(self, batch):
        # TODO: implement model forward
        # 1. embedding: (batch_size, sentence_len) --> (batch_size, sentence_len, embedding_size)
        embeded = self.embed(batch)
        batch_size = batch.size(0)
        # 2. initialize hidden state and cell state for lstm (2*num_layers, batch_size, hidden_size)
        h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size))
        
        # 3. propagate input through LSTM
        # output (batch_size, sequence_len, 2*hidden_size)
        output, (hn, cn) = self.lstm(embeded, (h0, c0))
        output = self.dropout(output)
        # 4. 1D convolution
        # output1 (batch_size, 100, sentence_len)
        output1 = F.relu(self.convolution1(output))
        # output2 = F.relu(self.convolution2(output))
        # output3 = F.relu(self.convolution3(output))
        # max_pool1d
        maxpool = torch.max(output1, dim=2)[0]
        # output2 = torch.max(output2, dim=2)[0]
        # output3 = torch.max(output3, dim=2)[0]
        # cat
        # output = torch.cat((output1, output2, output3), dim=1)
        # 5. 加上dropout
        # output = self.dropout(maxpool)
        # 6. dense layer (128, 300) --> (128, 150)
        output = self.dense(maxpool)
        # 7. log_softmax (128, 150) --> (128, 150)
        output = F.log_softmax(output, dim=1)
        return output
    
    @property
    def conv_output_size(h_in, w_in, filter_size) -> int:
        # default parameters
        stride = 1
        pad = 0
        dilation = 1
        # calculate the output dimension of cnn
        h_out = floor(( (h_in + 2*pad - dilation * (filter_size - 1)-1) / stride) + 1)
        w_out = floor(( (w_in + 2*pad - dilation * (self.embed_size - 1)-1) / stride) + 1)
        
        return h_out, w_out
        
        
