from typing import Dict

import torch
from torch.nn import Embedding
from torch.autograd import Variable
import torch.nn.functional as F


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
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
        self.gru = torch.nn.GRU(
            input_size=self.embed_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True)
        # fully-connected layer (bidirectional --> *2)
        self.dense = torch.nn.Linear(self.hidden_size * 2, self.num_class)
        # dropout
        self.dropout = torch.nn.Dropout(self.dropout)

    def forward(self, batch):
        # TODO: implement model forward
        # 1. embedding: (batch_size, sentence_len) --> (batch_size, sentence_len, embedding_size)
        batch = self.embed(batch)
        
        # 2. initialize hidden state and cell state for lstm (2*2, 128, 512)
        batch_size = batch.size(0)
        h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size))
        
        # 3. propagate input through LSTM
        output, hn = self.gru(batch, h0)
        # 4. output.shape = ()
        # 4. use the last output from lstm (batch_size, 2*hidden_size)
        # TODO: 加上 pooling
        output = torch.cat((output[:,-1,:self.hidden_size], output[:,0,-self.hidden_size:]),dim=1)
        
        # TODO: 加上dropout
        output = self.dropout(output)
        # 5. dense layer (128, 1024) --> (128, 150)
        output = self.dense(output)
        # 6. log_softmax (128, 150) --> (128, 150)
        output = F.log_softmax(output, dim=1)
        return output

class SeqTagClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqTagClassifier, self).__init__()
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
            dropout=dropout,
            batch_first=True)
        # fully-connected layer (bidirectional --> *2)
        self.dense = torch.nn.Linear(self.hidden_size * 2, self.num_class)
        # dropout
        self.dropout = torch.nn.Dropout(self.dropout)

    def forward(self, batch):
        # TODO: implement model forward
        # 1. embedding: (batch_size, sentence_len) --> (batch_size, sentence_len, embedding_size)
        batch = self.embed(batch)
        
        # 2. initialize hidden state and cell state for lstm (2*2, 128, 512)
        batch_size = batch.size(0)
        h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size))
        # 3. propagate input through LSTM
        output, (hn, cn) = self.lstm(batch, (h0, c0))
        # 4. reshape output from (batch_size, max_len, 2* hidden_size) to (batch_size * max_len, 2* hidden_size)
        output = output.contiguous().view(-1, 2* self.hidden_size)
        # 5. dropout
        output = self.dropout(output)
        # 5. dense layer (batch_size* max_len, 2* hidden_size) --> (batch_size* max_len, num_class)
        output = self.dense(output)
        # 6. log_softmax (batch_size* max_len, num_class) --> (batch_size* max_len, num_class)
        output = F.log_softmax(output, dim=1)
        return output