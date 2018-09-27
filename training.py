# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 09:34:36 2018

@author: Administrator 

Train a model of the training data. i.e. Find the best weight for each feature
function in "feats_dict".

Use MLE method to construct the objective funtion: f=-l(θ), where,
l(θ)=∑i ∑t ∑k θkfk(yt^(i) ,y(t−1)^(i),xt^(i)−∑i logZ(x^(i)). And calculate 
gradients for f.
Use L-BFGS to optimize parameters of "f".

References
----------
    John Lafferty, Andrew McCallum, Fernando C.N. Pereira. Conditional Random 
    Fields: Probabilistic Models for Segmenting and Labeling Sequence Data, 
    (2001), Proc. 18th International Conf. on Machine Learning, pp. 282–289.
    
    Charles Sutton and Andrew McCallum. An Introduction to Conditional Random 
    Fields, (2011), Foundations and Trends in Machine Learning, Vol.4, No.4,
    pp. 267–373.   

"""
import numpy as np
from features_extraction import obt_feat
from lbfgs import Lbfgs

#Define the the range of output tags.If it is an NER task, then the range of
#tags would be:
tags_range=['B_PER','I_PER','B_LOC','I_LOC','B_T','I_T','B_ORG','I_ORG','O']
m=len(tags_range)

def cal_mx(train_data,i,feats_dict,template):
    """
   Calculate the matrix which is needed in the calculation of forward and
   backward probabilities.
   
   Parameters
   ----------
   train_data:List
              "train_data" is the same as that used in function "feats_extract()"
              in module "features_extraction.py"
     
    i         : Int
               "i" is an index which indicates the position where all the 
               available information of a particular instance is stored. eg.
               "i=0" means that we are calculating "mx" for the first line (usually
               a sentence) with information "train_data[k][0] (k=0,...,N-1)",
               where "N" is the number of sorts of information.
               
   feats_dict: Dictionary
               The key is a string that stands for a feature function. Its 
               value is an array, where the first element is the weight of 
               corresponding feature function and the second element is gradient.
               
               The reason of storing weight and gradient together is that we may
               have to use gradients and weights at the same time in following
               programs. To avoid creating another dictionary which requires 
               extra storage, we choose to store gradient and weight in an array.
                         
   template  : List
               "template" here is the same as that in function "feats_extract()"
               in module "features_extraction.py". 
               
  
               
    Returns
    ----------
    mx       : Ndarray
               Each element is the log-pobability log(Mi(yp,y|x)),where
               Mi(yp,y|x) = exp(Λi(yp,y|x)) 
               Λi(yp,y|x) = ∑k λkfk(ei,Y|ei = (yp,y),x)+ ∑k µkgk(vi,Y|vi = y,x)
               See John Lafferty et al.(2001) for details.
    
    """
    #Get length of the instance
    n=len(train_data[0][i]) 
    mx=np.zeros((n,m,m))
    for y in tags_range:
        y_id=tags_range.index(y)
        for j in range(n):
            for form in template:
                    feat=obt_feat(form,train_data,i,j)
                    #if "feat" involves the  tag of for the previous token, i.e.
                    #the feature function is a transition 
                    #function:
                    if feat!='':
                        if '0:-1' in form:
                             feat=feat.split('/')
                             del feat[0]
                             feat='/'+'/'.join(feat)
                             for yp in tags_range:
                                yp_id=tags_range.index(yp)
                                f_func=y+'_'+yp+feat
                                #Check if the condition for the feature function
                                # to be 1 is satisfied. If satisfied, add the
                                # corresponding weight.
                                if f_func in feats_dict.keys():
                                    mx[j][yp_id][y_id]+=feats_dict[f_func][0]
                        #if it is a state function
                        else:
                            f_func=y+'_'+feat
                            if f_func in feats_dict.keys():
                                state_c=np.array([feats_dict[f_func][0]])
                                state_c=state_c.repeat(m)
                                mx[j,:,y_id]+=state_c
    #"a0" stores the probabilities of different naming entity tags for the first token.
    a0=mx[0,0,:]
    #The tag before the first token has not been defined here (Unlike in John 
    #Lafferty et al.(2001) it is defined to be a tag "start"). Hence, M0(yp,y|x)
    #for the first token is meaningless, and mx[0] should be deleted.
    mx=np.delete(mx,0,axis=0)
    return mx,a0
 
def cal_alpha(mx,a0,n):
    """
    Calculate the array of forward unnormalized log-probabilities for an instance.
    
    "alpha[i,j]" is the forward unnormalized log-probability of having a state
    sequence x0,x1,...,x(i-1) and that the naming entity tag at position i is
    tags_range[j].
    
    We directly use "a0" calculated above as alpha[0], which should be an array
    that stores the probabilities of different naming entity tags for the first
    token.
    
    """
    alpha=np.zeros((n,m))
    alpha[0:]=a0
    for i in range (1,n):
        for j in range(m):
            #"mx[i-1,:,j]" concerns with (i+1)th token in the given instance.
            #"alpha[i-1,:]" concerns with i th token in the given instance.
            alpha[i,j]=logsumexp(alpha[i-1,:]+mx[i-1,:,j])
    return alpha

def cal_beta(mx,n):
    """
    Calculate the array of backward unnormalized log-probabilities.
    
    "beta"[i,j] is the backward unnormalized log-probability that given a naming 
    entity tag "tags_range[j]" at position i, we have the state sequence from 
    position i+1 to the final position x(i+1),...,x(n-1), where "n" is the length 
    of the given instance. In addition, we define "xn" to be the state for the 
    token following the last token in the instance. "xn" is assumed to be "null".
    
    For the last token in the instance, since the state of the token following 
    it, i.e."xn" must be "null", the array of log-probabilities beta[n-1] should be 
    an array of zeros.
    """
    beta=np.zeros((n,m))
    ins_range=range(n-1)
    ins_range.reverse()
    for i in ins_range:
        for j in range(m):
            beta[i,j]=logsumexp(beta[i+1,:]+mx[i,j,:])
    return beta

def logsumexp(arr):
    """
    Calculate the log of sum of exponentials of an array.
    
    """
    b=arr.max()
    return b + np.log((np.exp(arr-b)).sum())
    
def cal_Z(alpha,n):
     """
     Calculate the log normalization log(p(x)), where p(x)= ∑yT αT(yT). 
     (See Charles Sutton et al.(2011), pp.315 for details.)
     
     """
     return logsumexp(alpha[n-1,:])    

def cal_exp(train_data,i,n,feats_dict,template,mx,alpha,beta,Z):
    """
    Calculate model expectation for each feature function.
    
    Parameters
    ----------
    feats_dict: Dictionary
                Value of each key is an array where the element at index 1 is 
                initialized as zero and the element at index 0 is the weight of 
                the corresponding feature function.
    
    Returns
    ----------
    feats_dict: Dictionary
                Value of each key is an array where the element at index 1 is
                the model expectation for the corresponding feature function.
    """
    for y in tags_range:
        y_id=tags_range.index(y)
        for j in range(n):
            #Calculate log(p(yt|x)) where p(yt|x)=at(yt)*bt(yt)/Z(x). See  
            #Charles Sutton et al.(2011), pp.318.
            p=alpha[j,y_id]+beta[j,y_id]-Z
            p=np.exp(p)
            for form in template:
                    feat=obt_feat(form,train_data,i,j)
                    if feat!='':
                        #For the transition funcitons
                        if '0:-1' in form:
                            feat=feat.split('/')
                            del feat[0]
                            feat='/'+'/'.join(feat)
                            for yp in tags_range:
                              yp_id=tags_range.index(yp)
                              f_func=y+'_'+yp+feat
                              if f_func in feats_dict.keys():
                                 #Calculate p(y(t-1),yt|x)
                                 q=alpha[j-1,yp_id]+mx[j-1,yp_id,y_id]+beta[j,y_id]-Z
                                 q=np.exp(q)
                                 feats_dict[f_func][1]+=q
                        #For the state functions
                        else:
                            f_func=y+'_'+feat
                            if f_func in feats_dict.keys():
                                feats_dict[f_func][1]+=p
    return feats_dict
                               
def cal_g(train_data,i,n,feats_dict,template,mx,alpha,beta,Z,freq_ar,K):
    """
    Calculate gradient for each feature.

    Parameters
    ----------
    freq_ar  : Ndarray
                "freq" stores frequency of each feature, which can be seen as 
                observation expectation for that feature.
                
    K        : List
               key list of "feats_dict". 
        
    Returns   
    ----------
    g        : Ndarray
               "g" is of shape "(n,)" ,where "n" is the number of feature functions
               in "feats_dict".
               "g[j]" (0<=j<=n-1) is the gradient for feature function "K[j]"(K[j]
               is a key of "feats_dict").
                
    
    """         
    feats_dict=cal_exp(train_data,i,n,feats_dict,template,mx,alpha,beta,Z)
    g=d_to_arr(feats_dict,K,1)
    g=g-freq_ar
    return g

def d_to_arr(dic,K,i):
    """
    Extract values from a dictionary and store it in an array .
    
    """
    li=[]
    for key in K:
        li.append(dic[key][i])
    arr=np.array(li)
    return arr

def arr_to_d(dic,arr,K):
    """
    Append values to a dictionary.
    
    """
    ln=len(K)
    for j in range(ln):
        key=K[j]
        dic[key]=arr[j]
    return dic

    
def g_f(theta,params):      
    """
    Calculate value of the objective function and its gradients.
    
    Parameters
    ----------
    theta     : Ndarray
                "theta" stores weights of feature functions in "feats_dict" .
    
    params    : List
                "params=[train_data,feats_dict,template,freq_ar,K]" 
    
    Returns
    ----------
    g          : Ndarray
                "g" stores the gradients given parameters "theta".
    
    f          : Float
                 "f" is the objective function value for the training data, given
                 parameters "theta".

    """
    #Append weights in "theta" to feats_dict and initialize each expectation to be 
    #zero again.
    ln=len(theta)
    g=np.zeros((ln,))
    theta=np.stack((theta,g),axis=1)
    params[1]=arr_to_d(params[1],theta,params[4])
    #Get the number of instances
    l=len(params[0][0])
    f=0
    for i in range(l):
        n=len(params[0][0][i])
        mx,a0=cal_mx(params[0],i,params[1],params[2])
        alpha=cal_alpha(mx,a0,n)
        beta=cal_beta(mx,n)
        Z=cal_Z(alpha,n)
        #Sum the gradients of all instances in the training data
        g+=cal_g(params[0],i,n,params[1],params[2],mx,alpha,beta,Z,params[3],params[4])
        #Calculate the value of the objective function f=-l(θ), where,
        #l(θ)=∑i ∑t ∑k θkfk(yt^(i) ,y(t−1)^(i),xt^(i)−∑i logZ(x^(i)).See Charles
        #Sutton et al.(2011), pp.333 for details.
        yp=params[0][0][i][0]
        s=a0[tags_range.index(yp)]
        for j in range(1,n):
            y=params[0][0][i][j]
            yp_id=tags_range.index(yp)
            y_id=tags_range.index(y)
            s+=mx[j-1,yp_id,y_id]
            yp=y
        f+=Z-s
    return g,f

def L2_regularize(theta,g,f,sigma):
    
    """
    Perform an L2-regularization on both the objective function value and its
    gradients.
    
    Parameters
    ----------
    sigma     : Float
                "sigma" is a parameter that determines how much to penalize large
                weights.
    Returns
    ----------
    f         : Float
                "f" returned is the L2-regularized objective function value. 
    g         : Ndarray
                "g" returned stores the L2-regularized parameters.
    
    """
    s=np.linalg.norm(theta)
    s*=s
    p=sigma*sigma
    f+=float(1)/(2*p)*s
    g/=p
    return g,f
 
def model(train_data,feats_dict,template,K,freq_ar,sigma):
    """
    
    Minimize the objective function value using L-BFGS.
    
    Returns
    ----------
    x or"0": Ndarray or int
             If the minimization succeeds, i.e.parameters found passed the 
             convergence test, then return an array "x" that stores the 
             parameters found. If maximum number of iterations has been 
             reached and appropriate parameters have not been found yet, 
             then return 0.
                           
    """
    params=[train_data,feats_dict,template,freq_ar,K]
    #Get the number of parameters to be optimized.
    n=len(K)
    #call the Lbfgs class
    lbfgs=Lbfgs(n)
    #initialize the array of parameters and the approximate Hessian inverse
    x,h=lbfgs.lbfgs_init()   
    g,fx=g_f(x,params)
    g,fx=L2_regularize(x,g,fx,sigma)
    gnorm=np.linalg.norm(g)
    xnorm=np.linalg.norm(x)
    #Early exit if "x" is already a minimizer
    if gnorm <= lbfgs.epsilon*max(xnorm,1):
       return x
    h=np.mat(h)
    u=[]
    k=0
    while k < lbfgs.max_iterations:
        
        g=np.transpose(np.mat(g))
        #Calculate the direction
        drt=-h*g
        drt=np.array(np.transpose(drt))[0]
        g=np.array(np.transpose(g))[0]
        g_old=g
        #Perform a line search to find the appropriate step
        step,g,fx=lbfgs.linesearch(fx,g,drt,x,g_f,params)
        x_old=x
        x=x+step*drt
        g,fx=L2_regularize(x,g,fx,sigma)
        gnorm=np.linalg.norm(g)
        xnorm=np.linalg.norm(x)
        # If passed gradient convergence test, then return "x"
        if gnorm <= lbfgs.epsilon*max(xnorm,1):
          return x
        si=x-x_old
        yi=g-g_old
        crr_tpl=(si,yi)
        #Update the corrections
        if k <=lbfgs.m-1:
            u.append(crr_tpl)
        else:
            for i in range(lbfgs.m-1):
                u[i]=u[i+1]
            u[lbfgs.m-1]=crr_tpl
        #Update the approximate Hessian inverse
        h=lbfgs.update_hk(k,u,n)
        k+=1
    print 'Maximum number of iterations has been reached.'
    return 0
    


        
        









    
    
              