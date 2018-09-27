# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 07:29:11 2018

@author: Administrator 

Create an L-BFGS class.

References
----------

    Dong C Liu and Jorge Nocedal. On the Limited Memory BFGS Method for Large 
    Scale Optimization, (1989), Mathematical Programming 15, pp. 503-528.	

"""
import numpy as np

class Lbfgs():
    def __init__(self,n):
        """
        Initialize parameters used in L-BFGS
        
        """
        
        self.m=6 #The limited memory. The default value of m is 6.
        self.n=n #The number of parameters that is to be optimized
        self.max_iterations=1500 #The maximum number of iterations of L-BFGS
        self.epsilon=1e-5 #A Parameter used in the convergence test: ||g||<=self.epsilon*max(xnorm,1)
        self.max_linesearch=20 #The maximum number of iterations of line search
        self.min_step=1e-20 #The minimum line search step that is allowed 
        self.max_step=1e+20#The maximum line search step that is allowed
        self.ftol=1e-4 #A parameter that controls the accuracy of line search. 0<self.ftol<0.5.
        self.wolfe=0.9 #A parameter used in the construction of wolfe condition. self.ftol<self.wolfe<1.0.
        
        
    def lbfgs_init(self):
        """
        Initialize the array of parameters and the approximate Hessian inverse.
        
        """
        x0=np.ones((self.n,))
        H0=np.eye(self.n,self.n)
        return x0,H0
    
    def update_hk(self,k,u,n):
        """
        Update the approximate Hessian inverse.
        
        Parameters
        ----------
        k         : Int
                    "k" is a number that indicates the current iteration. Number
                    "k" indicates the (k+1)th iteration.
        u         : List
                    Elements in "u" are tuples {(si,yi)|k-minor<=i<=k, minor=min(k,m-1)}
                    which are corrections needed in the update of the approximate 
                    Hessian inverse.
        n         : Int
                    The number of parameters to be optimized
                    
        Returns
        ----------
        h_new     : Matrix
                    The updated approximate Hessian inverse
                      
        """
        vt=np.mat(np.eye(n,n))
        v=np.mat(np.eye(n,n))
        minor=min(k,self.m-1)
        h_new=np.mat(np.zeros((n,n)))
        #rho=1/y's
        rho=1/np.dot(u[minor][0],u[minor][1])
        u_indices=range(minor+1)
        u_indices.reverse()
        #Use iterative method to update the approximate Hessian inverse. 
        
        #According to Dong C Liu et al.(1989), 
        #         h_new=(V'k*...*V'(k-minor))*H0*(V(k-minor)*...*Vk)
        #              +rho(k-minor)*(V'k*...*V'(k-minor+1))*S(k-minor)*S'(k-minor)*(V(k-minor+1)*...*Vk)
        #              +rho(k-minor+1)*(V'k*...*V'(k-minor+2))*S(k-minor+1)S'(k-minor+1)*(V(k-minor+2)*...*Vk)
        #              .
        #              .
        #              .
        #              +rho(k)*Sk*S'k
        #where "V'" is the transpose of "V".
        #The "vt" and "v" which is initialized to be identity matrices are
        #updated to be (V'k*...*V'(k-minor+i)) and (V(k-minor+i)*...*Vk) in  
        #(i+1)th iteration
        for i in u_indices: 
            ms=np.mat(u[i][0])
            ms_t=ms
            ms=np.transpose(ms)
            h_new+=rho*vt*ms*ms_t*v
            my=np.transpose(np.mat(u[i][1]))#my是个n*1的矩阵
            v_new=np.mat(np.eye(n,n))-rho*my*ms_t
            vt*=np.transpose(v_new)
            v=v_new*v
            rho=1/np.dot(u[i-1][0],u[i-1][1])
        #By iterations above, we've got
        #            h =rho(k-minor)*(V'k*...*V'(k-minor+1))*S(k-minor)*S'(k-minor)*(V(k-minor+1)*...*Vk)
        #              +rho(k-minor+1)*(V'k*...*V'(k-minor+2))*S(k-minor+1)S'(k-minor+1)*(V(k-minor+2)*...*Vk)
        #              .
        #              .
        #              .
        #              +rho(k)*Sk*S'k
        #Then the updated approximate Hessian inverse should be h_new=h+(V'k*...*V'(k-minor))*H0*(V(k-minor)*...*Vk)
        h_new+=vt*np.mat(np.eye(n,n))*v
        return h_new
        
   
    def linesearch(self,fx_init,g,drt,xk,g_f,*args):
        """
        Use line search to find out the best step.
        
        Parameters
        ----------
        fx_init   : Float
                    The objective function value 
        drt       : Ndarray
                    An array of search directions
        xk        : Ndarray
                    An array that stores parameters that is to be optimized
        g_f       : Function
                    A function that returns the objective function value and its
                    gradients
        *args     : Tuple
                    A tuple that stores the arguments for "g_f()"
        
        Returns
        ----------  
        step      : Float
                    The appropriate step  to update "xk"
        g_new     : Ndarray
                    The array of gradients at the updated "xk"
        fx_new    : The objective function value at the updated "xk"
        
        Reference
        ----------
        The line search routine in current program is a modified version of
        "LineSearch.h" in LBFGS++ (http://yixuan.cos.name/LBFGSpp/doc/index.html)
        by yixuan.
        Line search in current program sets Wolfe condition only as limitation, 
        for L-BFGS algorithm introduced by Dong C Liu et al.(1989)only requires
        the satisfaction of Wolfe condition.
        
        """
        step=1
        icr=2.1
        dcr=0.5
        dg_init=np.dot(g,drt)
        if dg_init>0:
            print'The moving direction is increasing the objective function value. '
        dg_test=self.ftol*dg_init
        for i in range(self.max_linesearch):
            x_new=xk+step*drt
            g_new,fx_new=g_f(x_new,args[0])
            # Check if Wolfe condition is satisfied
            if fx_new > fx_init+step*dg_test:
                width=dcr
            else:
                if np.dot(g_new,drt)<self.wolfe*dg_init:
                    width=icr
                else:
                    break
            step*=width
        if(i >=self.max_linesearch-1):
            print'The line search routine reached the maximum number of iterations.'
        if(step<self.min_step):
            print'The line search step becomes smaller than the minimum value allowed.'
        if(step>self.max_step):
            print'The line search step becomes larger than the maximum value allowed.'
        return step,g_new,fx_new
          
    
    

        
                