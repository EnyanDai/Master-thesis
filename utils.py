# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 23:11:40 2018

@author: Enyan
"""
import torch
class tag:
    def prepare_sequence(seq,tag_to_indx):
        idxs=[]
        for word in seq:
            if word in tag_to_indx:
                idxs.append(tag_to_indx[word])
            else:
                idxs.append(tag_to_indx['O'])
        return torch.tensor(idxs,dtype=torch.long)

def convert(pred):
    result=[]
    i=0
    while(i<len(pred)):
        value = pred[i]
        if(value==1):
            indexs=[i,i]
            i += 1
            while((i<len(pred)) and (pred[i]==2)):
                indexs[1]=i
                i += 1
            result.append(indexs)
        else:
            i += 1
    return result

def metrics(pred,ground_truth):
    i=0
    for p in pred:
        if(ground_truth.count(p)>0):
            i += 1
    return i;

def linkMetrics(pred,linkSpans):
    n = 0
    total = len(linkSpans)
    for link in linkSpans:
        if('Source' in link.keys()):
            source = link['Source']
            if(pred.count([source['B'],source['E']]) <= 0):
                n += 1
                continue
        if('Target' in link.keys()):
            target = link['Target']
            if(pred.count([target['B'],target['E']]) <= 0):
                n += 1
                continue
            
    return total-n,total
