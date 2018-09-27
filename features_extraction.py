# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 09:49:32 2018

@author: Administrator

"""
def feats_extract (train_data,template):   
        """
       Extract features from training data.
    
       Parameters
       ----------
       train_data:List
                  "train_data" stores all sorts of available information to
                  infer the output tag. 
                  Note that "train_data[0]" should always be the list of tags 
                  of the training data. For instance, in an NER task, training
                  data is usually in the form of
                  "[[list of naming entity tags],[list of pos tags],[list of words]]" 
                  i.e. 3 sorts of information are used to predict the naming 
                  entity tags.
                  
       template: List
                 self-defined feature template where each element stands for a 
                 type of feature.
                 each element is in the form of:
                 "Sort of information:Distance from current token_Sort of 
                 information:Distance from current token_Sort of information:
                 Distance from current token..." eg.
                 "0:-1_1:0"
                 Sort of information is represented by the index of that sort in 
                 list "train_data". Distance from current token is a negative or
                 positive number(but note that it's part of a string here,not int 
                 type).Negative number is used to extract information from preceding 
                 tokens while positive number is used to extract information from 
                 posterior tokens. Finally, if "0" is used as distance from current 
                 token, that would mean we use information about the current token
                 to infer its naming entity tag. eg. In an NER task,
                 If train data is of the form:
    
                 "[[list of naming entity tags],[list of pos tags],[list of words]]"
                 
                 "0:-1" means that we use the naming entity tag of the previous 
                 word to predict the output.
                 "0:-1_1:0" above is an example of composite feature which means
                 previous naming entity tag and pos tag of the current token are
                 both used to predict the output.
                 Note that in a composite feature, if it involves the tag of 
                 the previous token, i.e. if it represents a transition function,
                 the notation representing the previous tag should always be 
                 placed as the beginning of the expression.Hence in the above 
                 example instead of "1:0_0:-1",we use "0:-1_1:0", for "0:-1" 
                 represents the previous naming entity tag.
                 
                 
        
        Returns
        ----------
        feats_dict: Dictionary
                    A key stands for a feature function and its value is frequency 
                    of this feature. 
                    .
        """ 
        feats_dict={}
        #Get the number of lines(usually a line is a sentence.)
        m=len(train_data[0])
        for i in range (m):
            #Get the number of tokens in line i 
            n=len(train_data[0][i])
            for j in range (n):
                for form in template:
                    token_feat=obt_feat(form,train_data,i,j)
                    if token_feat!='':
                                f_func=train_data[0][i][j]+'_'+token_feat
                                if f_func not in feats_dict.keys():
                                    feats_dict[f_func]=1
                                else:
                                    feats_dict[f_func]+=1
                 
        return feats_dict
    
def obt_feat(form,train_data,i,j):
     """
     Generate a feature of a particular token using a form in the template.
     
     Parameters
     ----------
     form     : String
                "form" is an element of the feature template which represents 
                a particular type of feature. eg. 
                "0:-1" stands for the set of features defined by previous tag. 
                Using a feature form and information of a particular token, an
                exact feature of that token is generated.
    train_data: List
               "train_data" is the same as that defined in "feats_extract()".
    
    i         : Int
                "i" is the index of a line (usually a sentence) in list
                "train_data"
    j         : Int
                "j" is the index of a particular token in a line.
                "i" and "j" specify a token in the training data.
                
    Returns
    ----------
    token_feat: String
                If a token has a feature of the form specified, then a feature
                string is returned.
                Otherwise an empty string is returned.
                
    
     """
     token_feat=''
     atoms=form.split('_')
     #Check if "0:-1" has been placed at the beginning of the feature expression
     if '0:-1' in form and atoms[0]!='0:-1':
         print 'Error! "0:-1" should be placed at the beginning of a feature expression. '
         return
     for a in atoms:
            a=a.split(':')
            #Try to get the index of the atomic feature which comprises a whole 
            #feature with other atoms(if the form doesn't stand for a composite 
            #feature, then the atomic feature itself is the whole feature)
            target=j+int(a[1])
            #Check if the index is out of range
            if target<0 or target >len(train_data[0][i])-1:
                return ''
            else:
                info_tp=int(a[0])
                # Get feature string and add add a[1] as suffix to specify
                # the token of each atomic feature 
                token_feat+=train_data[info_tp][i][target]+'/'+a[1]+'_'  
     return token_feat

def shrink(feats_dict,freq):
        """
        Shrink the "feats_dict" according to frequency of the feature in 
        "feats_dict".
        
        """
        for key in feats_dict.keys():
            if feats_dict[key]<=freq:
                del feats_dict[key]
        if feats_dict=={}:
            print "Error! The feature dictionary is empty!"
            return 
        else:
            return feats_dict
        
"""
Run a test

words=['世纪钟的底座为长城城垛形状,钟体宽4.5米,高5.8米,钟体面积为26.1平方米。'.decode('utf-8')]
pos=['nnnunnvnnnnnnwnnammmqwammmqwnnnnvmmmmqqqw']
tags=[['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B_LOCATION', 'I_LOCATION', 'O', 
       'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 
       'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 
       'O', 'O', 'O']]

train_data=[tags,words,pos]
# get  5 types of features which include: 1)the naming entity tag of the previous 
#token ,2)the word of current token, 3)the pos of the previous word, 4) a 
#composite feature consisting the previous word and the current word, 5) a 
#composite feature consisting of the naming entity tag and  pos of the previous 
#token
template=['0:-1','1:0','2:-1','1:-1_1:0','0:-1_ 2:-1']
feats_dict=feats_extract(train_data,template)
feats_dict=shrink(feats_dict,1)
print feats_dict

"""








