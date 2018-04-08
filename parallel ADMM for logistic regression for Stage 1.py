
import numpy as np
import scipy
import random
import time
import multiprocessing
#from cvxpy import *
#ADMM for logistic regression

# n samples and real_beta is p dimension


#beta is n*p matrix
#z is 1*p vector
#u is n*p matrix
def ADMM_Logis_Regre(y,X,lam_0,rho=1,regularization=1,ep_abs=1e-4,ep_rel=1e-2):
    (n,p) = X.shape
    stop = False
    beta = np.zeros(n*p).reshape(n,p)
    z = np.zeros(1*p).reshape(1,p)[0]
    u = np.zeros(n*p).reshape(n,p)
    turn=0
    while((1-stop)):
        (beta,z,u,r,s) = ADMM_logis_update(beta,z,u,y,X,rho,n,p,lam_0)
        stop = ADMM_logis_stop(ep_abs,ep_rel,r,s,n,p,beta,z,u,rho)
        turn = turn + 1
        if(turn%1==0):
            print(turn)
    return(beta,z,u,r,s)
    
    
#update parameters
def ADMM_logis_update(beta,z,u,y,X,rho,n,p,lam_0):
    beta_new = ADMM_logis_update_beta(beta,u,z,n,rho,y,X)
    
    z_new = (beta_new.sum(0) + u.sum(0))/float(n)
    z_tem = abs(z_new)-lam_0/float(n*rho)
    z_new = np.sign(z_new) * z_tem * (z_tem>0)

    s = z_new - z
    
    r = beta_new - np.ones(n).reshape(n,1) * z_new
    u_new = u + r
    return(beta_new,z_new,u_new,r,s)
    
#target function of beta
def target_beta(beta,*args):
    u = args[0]
    z = args[1]
    n = args[2]
    rho = args[3]
    y = args[4]
    x = args[5]
    return(np.log(1+np.e**(-y*np.dot(x,beta))) + (rho/2.0) * np.sum((beta-z+u)**2))
    
#gradient of target function of beta
#def grad_target_beta(beta,*args):
#    u = args[0]
#    z = args[1]
#    n = args[2]
#    rho = args[3]
#    y = args[4]
#    x = args[5]
#    return()
    
#abtain beta by L-BFGS      
# ? parallel ?     
def ADMM_logis_update_beta(beta,u,z,n,rho,y,X):
    beta_new = []
    pool = multiprocessing.Pool(processes=4)
#    time_start=time.time()
#    for i in range(0,n):
#        beta_tem = scipy.optimize.minimize(target_beta,beta[i,:],args=(u[i,:],z,n,rho,y[i],X[i,:]),method='L-BFGS-B').x
#        beta_new.append(beta_tem)
    for i in range(n):
        r = pool.apply_async(beta_update, (i,beta[i,:],u[i,:],z,n,rho,y[i],X[i,:]))
        beta_new.append(r.get())
    pool.close()
    pool.join()
#    time_end=time.time()
#    print('totally cost',time_end-time_start)
    beta_new = np.array(beta_new)
    return(beta_new)
        
def beta_update(i,beta,u,z,n,rho,y,X):
    return(scipy.optimize.minimize(target_beta,beta,args=(u,z,n,rho,y,X),method='L-BFGS-B').x)

#stopping Criteria
def ADMM_logis_stop(ep_abs,ep_rel,r,s,n,p,beta,z,u,rho):
    e_pri = (n*p)**(0.5) * ep_abs + ep_rel * (max(np.sum(beta**2),np.sum(n*z**2)))**(0.5)
    e_dual = (p)**(0.5) * ep_abs + ep_rel * rho * (np.sum(u**2))**(0.5)/(n)**(0.5)
    stop = (np.sum(r**2) <= e_pri**2)&(np.sum(s**2) <= e_dual**2)
    return(stop)

def target_function(beta,y,X,rho):
    (n,p) = X.shape
    sum_tar = 0
    for i in range(n):
        sum_tar = sum_tar + np.log(1+np.e**(-y[i]*np.dot(X[i,:],beta))) + (rho/2.0) * np.sum((beta)**2)
    return(sum_tar)


if __name__ == '__main__':
    
    #######experience  
#    np.random.seed(1)
    n = 1000
    p = int(n*0.3)
    beta_real = np.array([np.random.uniform(-3,3) for i in range(p)]).reshape(1,p)
    beta_real = beta_real * (abs(beta_real)>=1)
    X = np.random.normal(0,1,n*p).reshape(n,p)
    
    y1 = np.sign(np.dot(X,beta_real.T).T + np.random.normal(0,0.1,n))[0]
    y2 = 1/(1+np.e**np.dot(X,beta_real.T).T)[0]
    y2 = np.array([np.random.binomial(1,i) for i in y2])
    y2 = 2*(y2-0.5)
    
    y=y2
    
    lam_0 = np.sum(np.dot(X.T,y)**2)**(0.5)
    
    #real 
    target_function(beta_real.T,y,X,lam_0)
    
    #ADMM
    time_start=time.time()
    res = ADMM_Logis_Regre(y,X,lam_0=1/float(n)*lam_0,rho=1,regularization=1,ep_abs=1e-4,ep_rel=1e-2)
    time_end=time.time()
    print('totally cost of ADMM:',time_end-time_start)
    
    print(beta_real)
    print(res[1])
    target_function(res[1],y,X,lam_0)
    
    #cvxpy
    
    
    
    
    
    
    
        
