# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 23:08:10 2018

@author: Enyan
"""

from data_load import data_loader
from preprocess import vocabulary
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from MultiTask import MultiTaskLSTM
from utils import tag,convert,metrics,linkMetrics


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_path = './colon_artificial/Dev'
train_path = './colon_artificial/train'
#types = 'TIMEX3'
types = 'EVENT'
loader = data_loader(train_path,types=types,sent_tokenizer='sent_tokenize.pickle')
[sent_list,sent_labels],[train_span_list,train_labels_span],[train_linkSpans,train_linkLabels] = loader.train_data()
[test_list,test_labels],[test_span_list,test_labels_span],[test_linkSpans,test_linkLabels] = loader.test_data(test_path)
v = vocabulary()
#tag_to_indx = {"O":0,"B-"+types:1,"IN-"+types:2}
tag_to_indx = {"O":0,"B-EVENT":1,"IN-EVENT":2}
v.load('vocabulary.dict')



                   
def test(model,threshold = 0.5):
    number = 0
    running_loss = 0.0
    acc = 0.0
    H = 0
    S = 0
    common = 0
    
    containLink = 0
    linkNumber = 0
    model.eval()
    for i in range(len(test_list)):
        pred_span=[]
        for j in range(len(test_list[i])):
            model.hidden = model.init_hidden()
            sentence_in = v.prepare_sequence(test_list[i][j]).to(device)
            labels = tag.prepare_sequence(test_labels[i][j],tag_to_indx).to(device)
            n = len(test_list[i][j])
            number += n
            output,source_scores,target_scores = model(sentence_in)
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



def train(EMBEDDING_DIM,HIDDEN_DIM,weight_decay,test_path,train_path,types,epochs = 30,pretrained_path=None):
        
    name = str(EMBEDDING_DIM)+'+'+str(HIDDEN_DIM)+'+'+str(weight_decay)+'+'+types+'MultiTask+conditional'
    model = MultiTaskLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(v.word_to_indx)+1, len(tag_to_indx),device)
    if pretrained_path != None:
        model.load_state_dict(torch.load(pretrained_path))
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(),weight_decay=weight_decay)
    model = model.to(device)
    
    F1Max = 0.0
    for epoch in range(epochs):
        running_loss,acc,precision,recall,F1,linkRecall = test(model)
        if(F1 > F1Max ):
            F1Max = F1
            torch.save(model.state_dict(),"model/"+name)
            with open("model/"+name+".txt","w") as file:
                file.write('[%d] loss: %.4f , acc: %.4f , precision: %.4f, recall: %.4f,F1: %.4f,LinkRecall: %.4f Testing'
                           %(epoch+1,running_loss,acc,precision,recall,F1,linkRecall))
        
        number = 0
        running_loss = 0.0
        acc = 0.0
        H = 0
        S = 0
        common = 0
        model.train()
        for i in range(len(sent_list)):
            pred_span=[]
            for k in range(len(sent_list[i])):
                optimizer.zero_grad()
                model.hidden = model.init_hidden()
                
                sentence_in = v.prepare_sequence(sent_list[i][k]).to(device)
                labels = tag.prepare_sequence(sent_labels[i][k],tag_to_indx).to(device)
                link_labels = torch.tensor(train_linkLabels[i][k],dtype = torch.long).to(device)
                
                '''get weight for link_tag_loss
                '''
                link_weight = labels.gt(0).float().to(device)
                
                link_labels_source = link_labels[:,0].to(device)
                link_labels_target = link_labels[:,1].to(device)
                n = len(sent_list[i][k])
                number += n
                model=model.train()
                output,source_scores,target_scores = model(sentence_in)
                tag_loss = F.nll_loss(output,labels)
                source_loss = F.nll_loss(source_scores,link_labels_source,reduce=False)
                target_loss = F.nll_loss(target_scores,link_labels_target,reduce=False)
                
                if(link_weight.sum() > 0):
                    source_loss = (source_loss*link_weight).sum()/link_weight.sum()
                    target_loss = (target_loss*link_weight).sum()/link_weight.sum()
                else:
                    source_loss = 0.0
                    target_loss = 0.0
                loss = tag_loss+10*(source_loss+target_loss)
                
                _,pred = torch.max(output,dim=1)
                acc += torch.sum(torch.eq(pred,labels).float()).data
                
                loss.backward()
                optimizer.step()
                for indexs in convert(pred.data):
                    pred_span.append([train_span_list[i][k][indexs[0]][0],train_span_list[i][k][indexs[1]][1]])
                running_loss += n*loss.data
            S += len(pred_span)
            H += len(train_labels_span[i])
            common += metrics(pred_span,train_labels_span[i])
        #        print(pred)
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
        print('[%d] loss: %.4f , acc: %.4f , precision: %.4f, recall: %.4f,F1: %.4f'
              %(epoch+1,running_loss/number,acc/number,precision,recall,F1))
        
def loadTest(EMBEDDING_DIM,HIDDEN_DIM,pretrained_path):
    model = MultiTaskLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(v.word_to_indx)+1, len(tag_to_indx),device)
    model.load_state_dict(torch.load(pretrained_path))
    test(model)        
        
        
def main():
    emb_dim = [25]
    hid_dim = [12]
    weight_decay_list = [1e-6]
#    learning_rate = 0.1
    for EMBEDDING_DIM in emb_dim:
        for HIDDEN_DIM in hid_dim:
            for weight_decay in weight_decay_list:
                train(EMBEDDING_DIM,HIDDEN_DIM,weight_decay,test_path,train_path,types,epochs=30)
if __name__ == "__main__":
    main()