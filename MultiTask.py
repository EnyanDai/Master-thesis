# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:54:35 2018

@author: Enyan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,device):
        super(MultiTaskLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim,padding_idx=vocab_size-1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim , bidirectional = True)
        self.embdropout = torch.nn.Dropout(p = 0.1)
        self.lstmdropout = torch.nn.Dropout(p = 0.1)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)
        self.hidden2link1 = nn.Linear(2*hidden_dim,2)
        self.hidden2link2 = nn.Linear(2*hidden_dim,2)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim).to(self.device),
                torch.zeros(2, 1, self.hidden_dim).to(self.device))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = self.embdropout(embeds)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        lstm_out = self.lstmdropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        
        link_tag_space1 = self.hidden2link1(lstm_out.view(len(sentence),-1))
        link_tag_scores1 = F.log_softmax(link_tag_space1,dim = 1)
        
        link_tag_space2 = self.hidden2link2(lstm_out.view(len(sentence),-1))
        link_tag_scores2 = F.log_softmax(link_tag_space2,dim = 1)
        
        return tag_scores,link_tag_scores1,link_tag_scores2