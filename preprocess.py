# -*- coding: utf-8 -*-
"""
Created on Thu May 31 00:29:17 2018

@author: Enyan
"""

import json
import io
from data_load import data_loader
import torch
class vocabulary:
    def __init__(self,sent_list=[]):
        self.word_to_indx = {}
        for doc in sent_list:
            for sent in doc:
                for word in sent:
                    if word not in self.word_to_indx:
                        self.word_to_indx[word] = len(self.word_to_indx)
        self.UNK = len(self.word_to_indx)
    def prepare_sequence(self,seq):
        idxs=[]
        for word in seq:
            if word in self.word_to_indx:
                idxs.append(self.word_to_indx[word])
            else:
                idxs.append(self.UNK)
        return torch.tensor(idxs)
    def write(self,path):
        data = json.dumps(self.word_to_indx)
        with io.open(path,'w',encoding='UTF-8') as f:
            f.write(data)
    def load(self,path):
        with io.open(path,'r',encoding='UTF-8') as f:
            data=f.read()
        self.word_to_indx = json.loads(data)
        self.UNK = len(self.word_to_indx)
def main():
    test_path = './colon_artificial'
    train_path = './colon_artificial'
    
    types = 'EVENT'
    loader = data_loader(train_path,types=types)
    loader.write_sent_tokenizer('sent_tokenize.pickle')
    [sent_list,sent_labels],[train_span_list,train_labels_span] = loader.train_data()
    [test_list,test_labels],[test_span_list,test_labels_span] = loader.test_data(test_path)
    
    v = vocabulary(sent_list)
    #tag_to_indx = {"O":0,"B-TIMEX3":1,"IN-TIMEX3":2}
    v.write('vocabulary.dict')
if __name__ == "__main__":
    main()
        