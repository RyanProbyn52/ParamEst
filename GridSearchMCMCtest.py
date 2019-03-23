
# coding: utf-8

# In[ ]:


#Importing libraries
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random


# In[ ]:


#Initialising data
x = [1, 2, 3, 4, 5]
y = [3.83, 8.75, 10.98, 14.18, 17.22]


# In[ ]:


#Initialising constants
sigma = 0.71
chisqmin = 1e10
a = np.linspace(-2.5,8,200)
b = np.linspace(0,8,200)
chi_sq = np.zeros((len(a),len(b)))
delchisq = np.zeros((len(a),len(b)))
chisqmin = 1e10
#Grid search method
#Iterating through entire grid
for i in range(0,len(b)):
    for j in range(0,len(a)):       
        for k in range(0,5): #chisq calculation iterating through all data points
            chi_sq[i,j] = chi_sq[i,j] + ((y[k] - (a[j] + b[i]*float(x[k])))/sigma)**2
            print(chi_sq[i,j])
    if chi_sq[i,j] < chisqmin:#min chisq value important for contour lines
        chisqmin = chi_sq[i,j]
        #print(chisqmin)
for i in range(0,len(b)):
    for j in range(0,len(a)):
        delchisq[i,j] = chi_sq[i,j] - chisqmin
        #print(delchisq[i,j])


# In[ ]:


cont = [0,2.3, 6.7, 11.8] #contour lines representing regions of confidence


# In[ ]:


cp = plt.contourf(a, b, chi_sq, cont)
plt.colorbar(cp)
plt.title('Filled Contours Plot')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.show()


# In[ ]:


#MCMC method
#intitialising values
nmcmc = 100000
chisq = np.zeros((nmcmc))
a = np.zeros((nmcmc))
b = np.zeros((nmcmc))
accept = 0     #counter to compute acceptance ratio
a[0] = 5   #specify initial value of a
sig_a = 0.001     #specify width of proposal density along a axis
b[0] = 20   #specify initial value of b
sig_b = 0.001     #specify width of proposal density along b axis
#resetting matrix values
for i in range(0,nmcmc):    
    a[i] = 0
    b[i] = 0
    chisq[i] = 0


# In[ ]:


#starting mcmc
for k in range(0,5):#calculating chi-square value for first data point - necessary since next loop relies on a previous data point for each iteration
    chisq[0] = chisq[0] + ((y[k] - (a[0] + b[0]*x[k]))/sigma)**2

print(chisq[0])    
for i in range(1,nmcmc):
    a_trial = a[i-1] + random.randint(-101,101)*sig_a
    b_trial = b[i-1] + random.randint(-101,101)*sig_b
    chisq_trial = 0
    for k in range(0,5):      #calculate chisqr for new data point    
        chisq_trial = chisq_trial + ((y[k] - (a_trial + b_trial*x[k]))/sigma)**2
    log_Lratio = 0.5*(chisq[i-1] - chisq_trial) #calculating logratio value
    if log_Lratio >= 0: #accepting value if uphill
        a[i] = a_trial
        b[i] = b_trial
        chisq[i] = chisq_trial
        accept = accept + 1
        
    else:
        ratio = np.exp(log_Lratio);   #note that ratio is less than unity        
        test_uniform = random.randint(1,1001)/1000;   #generate uniform random number in (0,1)        
        if test_uniform <= ratio:  #in this case again accept trial point as new point
            a[i] = a_trial
            b[i] = b_trial
            chisq[i] = chisq_trial
            accept = accept + 1
        else:     
            a[i] = a[i-1]
            b[i] = b[i-1]
            chisq[i] = chisq[i-1]
        
accept_ratio = accept/nmcmc
print(accept_ratio)
minval = np.argmin(chisq)
print(minval)
apar = a[minval]
bpar = b[minval]


# In[ ]:


plt.scatter(a,b,0.1)
plt.show()
print("most probable parameter estimation is: a =",apar,"and b =",bpar)


# In[ ]:


#now can use estimated parameter as starting point to generate regions of confidence
#initialising values
nmcmc = 1000000
a_b_chi = np.zeros((nmcmc, 3)) #inititallising array with 3xnmcmc for a, b and chisq
accept = 0     #counter to compute acceptance ratio
a_b_chi[0,0] = apar   #specify initial value of a
sig_a = 0.01     #specify width of proposal density along a axis
a_b_chi[0,1] = bpar   #specify initial value of b
sig_b = 0.01     #specify width of proposal density along b axis

#starting mcmc
for k in range(0,5):      #calculate chi-square for first data point    
    a_b_chi[0,2] = a_b_chi[0,2] + ((y[k] - (a_b_chi[0,0] + a_b_chi[0,1]*x[k]))/sigma)**2

print(chisq[0])    
for i in range(1,nmcmc):
    a_trial = a_b_chi[i-1,0] + random.randint(-101,101)*sig_a
    b_trial = a_b_chi[i-1,1] + random.randint(-101,101)*sig_b
    chisq_trial = 0
    for k in range(0,5):      #calculating chi-square for new data point    
        chisq_trial = chisq_trial + ((y[k] - (a_trial + b_trial*x[k]))/sigma)**2
    log_Lratio = 0.5*(a_b_chi[i-1,2] - chisq_trial) #calculating logratio value
    if log_Lratio >= 0: #accepting value if 'downhill'
        a_b_chi[i,0] = a_trial
        a_b_chi[i,1] = b_trial
        a_b_chi[i,2] = chisq_trial
        accept = accept + 1
        
    else:
        ratio = np.exp(log_Lratio)   #ratio is number between 0 and 1 (larger if change in chi-square values is large)      
        test_uniform = random.uniform(0,1)   #generate uniform random number in (0,1)        
        if test_uniform <= ratio:  #in this case again accept trial point as new point
            a_b_chi[i,0] = a_trial
            a_b_chi[i,1] = b_trial
            a_b_chi[i,2] = chisq_trial
            accept = accept + 1
        else:     #else reject and return to previous point
            a_b_chi[i,0] = a_b_chi[i-1,0]
            a_b_chi[i,1] = a_b_chi[i-1,1]
            a_b_chi[i,2] = a_b_chi[i-1,2]
        
accept_ratio = accept/nmcmc
print(accept_ratio)


# In[ ]:


sortabc = np.array(sorted(abc, key=lambda x : x[2])) # sorting list based on chisquared values
sig1 = round(nmcmc*0.6827) #1sig
sig2 = round(nmcmc*0.9545) #2sig
sig3 = round(nmcmc*0.9973) #3sig


# In[ ]:


plt.scatter(sortabc[:sig1,0],sortabc[:sig1,1],0.1, color='k')
plt.scatter(sortabc[sig1:sig2,0],sortabc[sig1:sig2,1],0.1)
plt.scatter(sortabc[sig2:sig3,0],sortabc[sig2:sig3,1],0.1)
plt.scatter(sortabc[sig3:,0],sortabc[sig3:,1],0.1)
plt.show()

