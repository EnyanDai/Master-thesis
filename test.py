# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 19:40:35 2018

@author: Enyan
"""

from data_load import data_loader
from preprocess import vocabulary
import torch
import torch.nn as nn
from MultiTask import MultiTaskLSTM
from utils import tag,convert,metrics,linkMetrics
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_path = './colon_artificial/Dev'
train_path = './colon_artificial/train'
types = 'TIMEX3'
loader = data_loader(train_path,types=types,sent_tokenizer='sent_tokenize.pickle')
[sent_list,sent_labels],[train_span_list,train_labels_span],[train_linkSpans,train_linkLabels] = loader.train_data()
[test_list,test_labels],[test_span_list,test_labels_span],[test_linkSpans,test_linkLabels] = loader.test_data(test_path)
v = vocabulary()
tag_to_indx = {"O":0,"B-"+types:1,"IN-"+types:2}
#tag_to_indx = {"O":0,"B-EVENT":1,"IN-EVENT":2}
v.load('vocabulary.dict')
MultiModel = MultiTaskLSTM(25, 12, len(v.word_to_indx)+1, len(tag_to_indx),device)
MultiModel.load_state_dict(torch.load("./model/25+12+1e-06+TIMEX3MultiTask"))
MultiModel = MultiModel.to(device)

def getDataSet(sents,init_labels):
    MultiModel.eval()
    stackDatasetLabels = []
    stackDatasetInput = []
    for i in range(len(sents)):
        tmpLabels = []
        tmpInput = []
        for j in range(len(sents[i])):
            MultiModel.hidden = MultiModel.init_hidden()
            sentence_in = v.prepare_sequence(sents[i][j]).to(device)
            labels = tag.prepare_sequence(init_labels[i][j],tag_to_indx).to(device)
            tmpLabels.append(labels)
            
            output,source_scores,target_scores = MultiModel(sentence_in)
                
            inputTensor = [output,source_scores,target_scores]
            tmpInput.append(inputTensor)
        stackDatasetInput.append(tmpInput)
        stackDatasetLabels.append(tmpLabels)
    return stackDatasetInput,stackDatasetLabels
        
sent_list,sent_labels = getDataSet(sent_list,sent_labels)
test_list,test_labels = getDataSet(test_list,test_labels)

def test(threshold = 0.5):
    number = 0
    running_loss = 0.0
    acc = 0.0
    H = 0
    S = 0
    common = 0
    
    containLink = 0
    linkNumber = 0
    for i in range(len(test_list)):
        pred_span=[]
        for j in range(len(test_list[i])):
            labels = test_labels[i][j]
            n = len(test_list[i][j][0])
            number += n
            [output,source_scores,target_scores] = test_list[i][j] 
            loss = nn.functional.nll_loss(output,labels)
            _,pred = torch.max(output,dim=1)
            _,source_pred = torch.max(source_scores,dim=1)
            _,target_pred = torch.max(target_scores,dim=1) 
            
            source_scores = torch.exp(source_scores)
            target_scores = torch.exp(target_scores)
            for k in range(len(pred)):
                if(source_scores[k][1]>threshold or target_scores[k][1]>threshold):
                    pred[k]=1 if output[k][1]>output[k][2] else 2
            for indexs in convert(pred.data):
                pred_span.append([test_span_list[i][j][indexs[0]][0],test_span_list[i][j][indexs[1]][1]])
            acc += torch.sum(torch.eq(pred,labels).float()).data
            running_loss += n*loss.data
            
        S += len(pred_span)
        H += len(test_labels_span[i])
        common += metrics(pred_span,test_labels_span[i])
        
        tmpContainLink,tmpLinkNumber = linkMetrics(pred_span,test_linkSpans[i])
        containLink += tmpContainLink
        linkNumber +=tmpLinkNumber
    print(S,H,common)
    if(S != 0):
        precision = common/S
    else:
        precision = 0.0
    recall = common/H
    if(common==0):
        F1 = 0.0
    else:
        F1 = 2*recall*precision/float(recall+precision)
        
    print(containLink,linkNumber)
    
    print('loss: %.4f , acc: %.4f , precision: %.4f, recall: %.4f,F1: %.4f,LinkRecall: %.4f Testing'
              %(running_loss/number,acc/number,precision,recall,F1,containLink/linkNumber))
    return running_loss/number,acc/number,precision,recall,F1,containLink/linkNumber
test()