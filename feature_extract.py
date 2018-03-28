# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:48:28 2017

@author: zhou
"""
from numpy import dot,array
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import gensim
import text_process
def load_model(model_path):
    #model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    model=gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False) 
    return model
    
class Content_extract():
    def __init__(self,method,model,with_weight,use_idf):
        self.method=method
        self.model=model
        self.with_weight=with_weight
        self.use_idf=use_idf
        self.vectorizer=TfidfVectorizer(stop_words='english',use_idf=self.use_idf)
    def compute_vector(self,tfidf_vector):
        essays_vector=[]
        tokens=self.vectorizer.get_feature_names()
        #print len(tokens)
        #5600
        #print tokens
        for article in tfidf_vector:
            #print article.shape
            context_vector=[]
            for i,token in enumerate(tokens):
                if token in self.model:
                #print token
                    word_vector=self.model[token]
                    #print word_vector.shape
                    weight=article[0,i]
                    #print weight.shape
                    word_vector= weight * word_vector 
                    #print word_vector
                    context_vector.append(word_vector)   
            essays_vector.append(sum(context_vector))
        return essays_vector
                    
    def fit(self,articles):
        #vectorizer=TfidfVectorizer(stop_words='english',use_idf=self.use_idf)
        tfidf_keywords=self.vectorizer.fit_transform(articles)
        #print tfidf_keywords.shape
        #(1983, 5600)
        content=self.compute_vector(tfidf_keywords)
        return content
     
       
docs,labels=text_process.extract_articles('acl2017dataset/story_clusters.txt')            
text_list=text_process.text_processing(docs) 
wordvector_model=load_model('glove.6B.300d1') 
content=Content_extract(method=1,model=wordvector_model,with_weight=True,use_idf=True).fit(text_list) 
        
        
        
        
    