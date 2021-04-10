from nltk.corpus import stopwords 
from nltk import download 
download('stopwords')

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os, sys
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

import wikipedia 

import torch


from pyemd import emd
import gensim
from gensim.models import Word2Vec


stop_words = stopwords.words('english')

"""
model stores word2vec for WMD and uses Google-News-vector Word2vec model
"""
model = gensim.models.KeyedVectors.load_word2vec_format('D:/GoogleNews-vectors-negative300.bin.gz', binary=True)  
model.init_sims(replace=True)

"""
embed uses 'USE' model for converting sentances to WordVec and then similarity is calculates using cosine_sim
"""
embed = hub.load("D:/USE")


def cos_sim(input_vectors):
    similarity = cosine_similarity(input_vectors)
    return similarity


# List of some negative words which will change the semantics of the sentance
negative = ["not" , "without","against","bad","useless","no","dislike","hate"]

"""
semantic_similairty function will find the similairty between actual answer and given answer and also taken into account the 
semantics and negations present in the sentance
"""
def semantic_similarity(actual_answer , given_answer) :
    actual = actual_answer.lower().split(".")
    given = given_answer.lower().split(".")
    
    
    
    sim_checker = actual 
    
    not_matching_semantics = list()
    
    semantic_1 = 0   # Actual_answer
    semantic_2 = 0   # Given_answee
    
    actual_embed_list = list()
    given_embed_list = list()
    
    
    
    for z in range(len(actual)) :
        list_actual = list()  
        list_actual.append(actual[z])
        actual_embed_list.append(embed(list_actual))
        #print(actual_embed_list[z].shape)
    
    for z in range(len(given)) :
        
        semantic_1 = 0
        semantic_2 = 0 
        list_given = list()
        list_given.append(given[z])
        embed_z = embed(list_given)
        
        
        sim_check = sim_checker.copy() 
        sim_check.append(given[z]) 
        
        sen_em = embed(sim_check)
        
        similarity_matrix = cos_sim(np.array(sen_em))
        
        similarity_matrix_df = pd.DataFrame(similarity_matrix) 
        
        cos_list = list(similarity_matrix_df[len(similarity_matrix_df) - 1]) 
        cos_list = cos_list[:len(cos_list)-1]
        #print(cos_list)
        
        index = cos_list.index(max(cos_list))
        
        actual_check = actual[index]
        actual_check = actual_check.split()
        for i in range(len(actual_check) - 1) :
            if(actual_check[i] in negative and actual_check[i+1] in negative) :
                semantic_1 += 1 
            elif(actual_check[i] in negative and actual_check[i+1] not in negative) :
                semantic_1 -= 1 
                
        
        


        answer_given = given[z].split()
        for i in range(len(answer_given) - 1) :
            if(answer_given[i] in negative and answer_given[i+1] in negative) :
                semantic_2 += 1 
            elif(answer_given[i] in negative and answer_given[i+1] not in negative) :
                semantic_2 -= 1 

                
        
        if(semantic_1 == 0 and semantic_2 == 0) :
            
            """
            Well and good
            """
        elif(semantic_1 < 0  and semantic_2 >= 0) :
            not_matching_semantics.append(list([actual[index],given[z]]))
            embed_z*=(-1)
            
        
        elif(semantic_1 >= 0 and semantic_2 < 0 ) :
            not_matching_semantics.append(list([actual[index],given[z]]))
            embed_z*=(-1)
            
        
        
        #print(semantic_1,semantic_2,actual[index],given[z])
        
        
        given_embed_list.append(embed_z)
        
        
        
    
    #print(np.array(actual_embed_list).shape)
    actual_embed = actual_embed_list[0] 
    #print(actual_embed.shape) 
    
    for i in range(len(actual_embed_list)-1) :
        #print(actual_embed_list[i+1].shape)
        actual_embed += actual_embed_list[i+1]
        
    given_embed = given_embed_list[0] 
    for i in range(len(given_embed_list) - 1) :
        given_embed += given_embed_list[i+1] 
        
        
    
            
    actual_embed = np.array(actual_embed).reshape(512)
    given_embed = np.array(given_embed).reshape(512) 
    sem_checker = list([actual_embed,given_embed]) 
    answer = pd.DataFrame(cos_sim(sem_checker))
            
        
    return not_matching_semantics , answer[0][1]



"""
WMD function will calculate Word's mover distance between actual and given answer using Google-News-Vector Word2Vec model
"""
def WMD(actual_answer , given_answer,model) :
    
    actual_answer = actual_answer.lower().split()
    actual_answer = [w for w in actual_answer if w not in stop_words]
    
    
    given_answer = given_answer.lower().split()
    given_answer = [w for w in given_answer if w not in stop_words]
    
    
    return model.wmdistance(given_answer,actual_answer)
    

"""
This function will return the final score (similairty) between actual and given answer 
"""
def score(given_answer , actual_answer,model) :
    
    given_answer1 = given_answer
    actual_answer1 = actual_answer
    
    given_answer2 = given_answer
    actual_answer2 = actual_answer

    not_macthing , similarity = semantic_similarity(actual_answer1,given_answer1)
    distance = WMD(actual_answer2,given_answer2,model)
    

    
    if(similarity > 0) :
        if(distance == 0) :
            return 1 
        return similarity/distance
    else :
        return -1 


# page1 = wikipedia.page("Republic of India")
# a = page1.summary

# page2 = wikipedia.page("Demographics of India")
# b = page2.summary


# page3 = wikipedia.page("Demographics of United States")
# c = page3.summary


# score(a,b,model)
# score(a,c,model)
# score(c,b,model)