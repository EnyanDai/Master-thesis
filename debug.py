# -*- coding: utf-8 -*-
"""
Created on Thu May 31 21:09:32 2018

@author: Enyan
"""
import re
import io,os
import xml

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
        else:
#            print("Not found")
    return linkSpans
            
            
    
contents = ''
fo = io.open('train.txt','r',encoding='UTF-8')
contents = fo.read()
fo.close()
contents = contents.split('\n')
TIMEX3 = []
EVENT =[]
LINK = []
nt=0
n = 0
for c in contents:
    element = re.split('\s+',c)
#    print(len(element))
    if(len(element)>1):
        if(element[1]=='EVENT'):
            nt += int(element[2])
            TIMEX3.append([element[0],int(element[2])])
#for c in contents:
#    element = re.split('\s+',c)
##    print(len(element))
#    if(len(element)>1):
#        if(element[1]=='TLINK:Type:CONTAINS'):
#            nt += int(element[2])
#            LINK.append([element[0],int(element[2])])
from data_load import getEvent,getEventSpan
path='./colon_artificial/train'
for [p,number] in TIMEX3:
    for inside_path in os.listdir(path+'/'+p):
        if(inside_path.find('Temporal')>0 and inside_path.find('gold')>0):
            dom=xml.dom.minidom.parse(path+'/'+p + '/'+inside_path)
            root = dom.documentElement
            lables=getEvent(root,'EVENT')
            lables +=getEvent(root,'TIMEX3')
            linkLabels = getLink(root)
            linkSpans = getLinkSpanFromEvent(linkLabels,lables)
            if(len(linkLabels)!=len(linkSpans)):
                print(len(linkLabels),len(linkSpans))
            lables = getEventSpan(lables)
            n += len(lables)
#            if(number!=len(lables)):
#                print(p,number,len(lables))
#for [p,number] in LINK:
#    for inside_path in os.listdir(path+'/'+p):
#        if(inside_path.find('Temporal')>0 and inside_path.find('gold')>0):
#            dom=xml.dom.minidom.parse(path+'/'+p + '/'+inside_path)
#            root = dom.documentElement
#            lables=getLink(root)
#            n += len(lables)
#            if(number!=len(lables)):
#                print(p,number,len(lables))
