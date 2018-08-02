# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 23:37:03 2017

@author: Enyan
"""
import os,random,io
import nltk
import  xml.dom.minidom
import re
import pickle

def getfiles(path):
    paths=os.listdir(path)
    path_list=[]
    for p in paths:
        tmp=[]
        data_path=path+'/'+p+'/'+p
        tmp.append(data_path)
        for inside_path in os.listdir(path+'/'+p):
            if(inside_path.find('Temporal')>0 and (inside_path.find('gold')>0)):
                tmp.append(path+'/'+p+'/'+inside_path)
        path_list.append(tmp)
#    print("shuffle data")
#    random.shuffle(path_list)
    return path_list

def getLink(root):
    root = root.getElementsByTagName('annotations')[0]
    relations=root.getElementsByTagName('relation')
    relationLabels = set()
    for relation in relations:
        relation_type = relation.getElementsByTagName('Type')[0].childNodes[0].nodeValue
        if(relation_type == 'CONTAINS'):
            sourceId = relation.getElementsByTagName('Source')[0].childNodes[0].nodeValue
            targetId = relation.getElementsByTagName('Target')[0].childNodes[0].nodeValue
            relationLabels.add(sourceId+','+targetId)
    labels = []
    for relation in relationLabels:
        labels.append(relation.split(','))
    return labels

def getLinkSpanFromEvent(linkLabels,EventLabels):
    linkSpans=[]
    for link in linkLabels:
        Span = {}
        for event in EventLabels:
            sourceId = link[0]
            targetId = link[1]
            if(event['id']==sourceId):
                Span['Source'] = {'B':event['B'],'E':event['E']}
            if(event['id']==targetId):
                Span['Target'] = {'B':event['B'],'E':event['E']}
        if(len(Span) > 0):
            linkSpans.append(Span)
    return linkSpans

def getEvent(root,types):
#    root = root.getElementsByTagName('annotations')[0]
    annotations=root.getElementsByTagName('entity')
    labels={}
    for entity in annotations:
        annlist=entity.childNodes
        ann={}
        for element in annlist:
            if(element.nodeName=='type'):
                ann['type']=str(element.childNodes[0].nodeValue)
            if(element.nodeName=='properties'):
                properties={}
                for prop in element.childNodes:
                    if(prop.nodeType==1):
                        properties[prop.nodeName]=prop.childNodes[0].nodeValue
                ann['properties']=properties
            if(element.nodeName=='span'):
                span=element.childNodes[0].nodeValue
                ann['B'] = min(int(re.split('[,;]',span)[0]),int(re.split('[,;]',span)[1]))
                ann['E'] = max(int(re.split('[,;]',span)[0]),int(re.split('[,;]',span)[1]))
            if(element.nodeName=='id'):
                ann['id'] = str(element.childNodes[0].nodeValue)
        if(ann['type'] == types and 'B' in ann):        
            labels[ann['id']]=ann
    return sorted(labels.values(),key=lambda s:s['B'])
def getEventSpan(labels):
    spans=set()
    for label in labels:    
        spans.add(str(label['B'])+','+str(label['E']))
    EventSpans=[]
    for span in spans:
        EventSpans.append({'B':int(re.split(',',span)[0]),'E':int(re.split(',',span)[1])})
    return EventSpans
    
def word_token(text):
    tokenizer = nltk.tokenize.RegexpTokenizer('\w+|[^\w\s]')
    return tokenizer.tokenize(text)
def span_word_token(text):
    tokenizer = nltk.tokenize.RegexpTokenizer('\w+|[^\w\s]')
    return tokenizer.tokenize(text),tokenizer.span_tokenize(text);
def inspan(left,right):
    a=max(left[0],right[0])
    b=min(left[1],right[1])
    return b>a
def find_label_index(spans,label):
    label_indexs=[]
    for i,span in enumerate(spans):
        if(inspan(span,label)):
            label_indexs.append(i)
    return label_indexs
            
        
#data_path,label_path = './colon_artificial/ID001_clinic_001/ID001_clinic_001','./colon_artificial/ID001_clinic_001/ID001_clinic_001.Temporal-Relation.gold.completed.xml'
#types = 'EVENT'
def preprocess(data_path,label_path,sent_tokenizer=None,types=None):
    dom=xml.dom.minidom.parse(label_path)
    root = dom.documentElement
    EventLabels=getEvent(root,types)
    labels = getEventSpan(EventLabels)
    
    linkLabels = getLink(root)
    linkSpans = getLinkSpanFromEvent(linkLabels=linkLabels,EventLabels=EventLabels)
    
    fo = io.open(data_path,'r',encoding='UTF-8')
    content = fo.read()
    content = content.lower()
    fo.close()
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    if(sent_tokenizer!=None):
        tokenizer = sent_tokenizer
    sent_list = []
    span_list = []
    sentences = tokenizer.tokenize(content)
    indexs = tokenizer.span_tokenize(content)
    start = 0
    sent_labels=[]
    link_labels=[]
    i = 0
    for sent in sentences:
        words,spans = span_word_token(sent)
        sent_list.append(words)
        label=[]
        link_label = []
        total_spans = []
        start = indexs[i][0]
        for s in spans:
            total_spans.append([start+s[0],start+s[1]])
            label.append('O')
            link_label.append([0,0])
        span_list.append(total_spans)
        sent_labels.append(label)
        link_labels.append(link_label)
        #start = content[end:].index(sent)+end
        #end = start + len(sent)
        i += 1
    n = 0
    labels_span=[]
    for label in labels:
        labels_span.append([label['B'],label['E']])
        for i in range(len(span_list)):
            if(inspan(indexs[i],[label['B'],label['E']])):
                result=find_label_index(span_list[i],[label['B'],label['E']])
                if(len(result) > 0):
                    sent_labels[i][result[0]] = 'B-'+types
                    n += 1 
                if( len(result) > 1):
                    for label_index in result[1:]:
                        sent_labels[i][label_index] = 'IN-'+types
                break
    if(n!=len(labels)):
        print('\n\nFailed\n\n')
    for link in linkSpans:
        if('Source' in link.keys()):
            source = link['Source']
            for i in range(len(span_list)):
                if(inspan(indexs[i],[source['B'],source['E']])):
                    result = find_label_index(span_list[i],[source['B'],source['E']])
                    for label_index in result:
                        link_labels[i][label_index][0] = 1
                    break
        if('Target' in link.keys()):
            target = link['Target']
            for i in range(len(span_list)):
                if(inspan(indexs[i],[target['B'],target['E']])):
                    result = find_label_index(span_list[i],[target['B'],target['E']])
                    for label_index in result:
                        link_labels[i][label_index][1]= 1
                    break

    return [sent_list,sent_labels],[span_list,labels_span],[linkSpans,link_labels]
        



def printdataset(path,sent_list,sent_labels):
    file = open(path, "w")
    for i in range(len(sent_list)):
        for j in range(len(sent_list[i])):
            for k in range(len(sent_list[i][j])):
                file.write(sent_list[i][j][k]+'\t'+sent_labels[i][j][k]+'\n')
            file.write('\n')
        file.write('\n')
    file.close()
def findindex(indexs,span1,span2):
    i= 0
    for [start,end] in indexs:
        a = max(start,span1)
        b = min(end,span2)
        if(a < b):
            return i
        i +=1
    return -1

class data_loader():
    def __init__(self,train_path=None,types='TIMEX3',sent_tokenizer=None):
        self.train_path = train_path
        if(train_path!=None):
            path_list = getfiles(train_path)
        contents = ''
        self.types = types
        if(sent_tokenizer==None):    
            for [data_path,label_path] in path_list:
                fo = io.open(data_path,'r',encoding='UTF-8')
                contents = fo.read()+'\n\n'+contents
                fo.close()
            sent_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer(contents)
            self.sent_tokenizer = sent_tokenizer
        else:
            self.load_sent_tokenizer(sent_tokenizer)
       
        
    def write_sent_tokenizer(self,path):
        out = open(path,"wb")
        pickle.dump(self.sent_tokenizer, out)
        out.close()
    def load_sent_tokenizer(self,path):
        self.sent_tokenizer = nltk.data.load(path)
        
    def train_data(self):
        
        if(self.train_path==None):
            print("No train data path is told in the init")
            return
        path_list = getfiles(self.train_path)
        sent_list=[]
        sent_labels=[]
        spans=[]
        labels_spans=[]
        linkSpans = []
        linkLabels = []
        
        for [data_path,label_path] in path_list:
#            print(data_path+' is being processed.')
            [sent,label],[span,labels_span],[linkSpan,linkLabel]=preprocess(data_path,label_path,self.sent_tokenizer,self.types)
            sent_list.append(sent)
            sent_labels.append(label)
            spans.append(span)
            labels_spans.append(labels_span)
            linkSpans.append(linkSpan)
            linkLabels.append(linkLabel)
        printdataset('./train',sent_list,sent_labels)
        return [sent_list,sent_labels],[spans,labels_spans],[linkSpans,linkLabels]
    def test_data(self,test_path):
        path_list = getfiles(test_path)
        sent_list=[]
        sent_labels=[]
        spans=[]
        labels_spans=[]
        linkSpans = []
        linkLabels = []
        contents = ''
        for [data_path,label_path] in path_list:
            fo = io.open(data_path,'r',encoding='UTF-8')
            contents = fo.read()+'\n\n'+contents
            fo.close()
        for [data_path,label_path] in path_list:
#            print(data_path+' is being processed.')
            [sent,label],[span,labels_span],[linkSpan,linkLable]=preprocess(data_path,label_path,self.sent_tokenizer,self.types)
            sent_list.append(sent)
            sent_labels.append(label)
            spans.append(span)
            labels_spans.append(labels_span)
            linkSpans.append(linkSpan)
            linkLabels.append(linkLable)
        printdataset('./test',sent_list,sent_labels)                 
        return [sent_list,sent_labels],[spans,labels_spans],[linkSpans,linkLabels]
