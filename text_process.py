# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:01:53 2017

@author: zhou
"""
#preprocessing each line label str
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
def extract_articles(in_file):
    with open(in_file,"r") as f:
        clusters=f.read().strip("%%%").split("\n%%%")
    docs=[]
    labels=[]
    for i,cluster in enumerate(clusters):
        for text in cluster.strip().split('\n')[1:]:
            docs.append(text)
            labels.append(i)
    return docs,labels
        
def text_processing(list_text):
    text_contain=[]
    stop_to_clearn=set(list(stopwords.words('english'))+['\n',',','.'])
    regex_non_alphanumeric=re.compile('[^0-9a-zA-Z]')
    #NLP Stemming,Lemmatization(maybe better)
    stemmer=PorterStemmer()
    
    for index,text in enumerate(list_text):
        single_text=[]
        for item in text.strip().split():
            item=regex_non_alphanumeric.sub('',item)
            item=item.lower()
            #item=stemmer.stem(item)
            single_text.append(item)
        clearned_list=[elem for elem in single_text if elem not in stop_to_clearn]
        text_contain.append(' '.join(clearned_list))
    return text_contain


            
docs,labels=extract_articles('acl2017dataset/story_clusters.txt')            
text_list=text_processing(docs)            
            

            
            
            
        
    
    
    
    
