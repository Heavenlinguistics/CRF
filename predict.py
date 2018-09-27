# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 08:00:43 2018

@author: Administrator

Predict the most likely tagging of  an instance using viterbi algorithm.

Reference
----------
    Function "predict()" here is a modified version of the function "argmax()"
    from Github project: Simple implementation of Conditional Random Fields (CRF)
    in Python (https://github.com/timvieira/crf/blob/master/crf/basecrf.py) by
    Tim Vieira.
    "argmax()" is modified to be "predict()" in current program, such that it 
    is consistent with our previously defined variables and functions.
    
"""
import numpy as np
from training import cal_mx
#Define the the range of output tags.If it is an NER task, then the range of
#tags would be:
tags_range=['B_PER','I_PER','B_LOC','I_LOC','B_T','I_T','B_ORG','I_ORG','O']

def predict(test_data,i,feats_dict,template):
    """
    Parameters
    ----------
    test_data : List
                "test_data" is the test dataset which has the same structure 
                as "train_data" used in previous functions.
                
    i         : Int
                "i" is used to locate a particular instance (usually a sentence)
                in "test_data"
    Returns 
    ----------
    max_prob_path: List
                   The most likely tag sequence of the input instance.
    
    """
    #Get the length of instance
    n=len(test_data[0][i])
    mx,a0=cal_mx(test_data,i,feats_dict,template)
    #Initialize "v" with probabilities of the first tag 
    v=a0
    trace=[]
    max_prob_path=[]
    for i in range (n-1):
        tmp=[]
        #"u" here is used to store maximum probabilities of current local path. 
        u=[]
        for y in tags_range:
            y_id=tags_range.index(y)
            w=v+mx[i,:,y_id]
            #Get the maximum probability of current local path given that the 
            #tag of current token is "y"
            j=np.argmax(w)
            u.append(w[j])
            #Each element in "tmp" is the index of the tag for the previous token 
            #which maximizes the probability of current local path.
            tmp.append(j)
        v=u
        v=np.array(v)
        trace.append(tmp)
    #Get the index of the tag for the last token
    last_i=np.argmax(v)
    last_tag=tags_range[last_i]
    max_prob_path.append(last_tag)
    #Once the index of the tag for the last token has been found, we can backtrack 
    #tags for the previous tokens through list "trace".
    for j in reversed(range(n-1)):
        tag_id=trace[j][last_i]
        curr_tag=tags_range[tag_id]
        max_prob_path.append(curr_tag)
        last_i=tag_id
    max_prob_path.reverse()   
    return max_prob_path

"""
#Run a test

words=[u'我爱你中国。',u'今天中午我和他吃饭了。']
pos=['nvnnnw','nnnnncnvntw']
tags=[['O','O','O','B_ORG','I_ORG','O'],['B_T','I_T','I_T','I_T','O','O','O',
      'O','O','O','O']]
train_data=[tags,words,pos]
feats_dict={'O_O/-1_O/-2_':[1,0],'B_T_I_T/-1_':[1,0],'I_ORG_'+u'\u4e2d'+'/-1_'+u'\u56fd'+'/0_':[1,0],'O_'+u'\u7231'+'/0_':[1,0]}
template=['0:-1','0:-1_0:-2','1:0','0:-1_0:0','1:-1_1:0']
max_prob_path=predict(train_data,0,feats_dict,template)
print max_prob_path

"""