
# coding: utf-8

# In[ ]:


#Importing libraries
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy.integrate import quad
import time
from scipy.interpolate import spline
from numpy import linspace, meshgrid
from matplotlib.mlab import griddata


# In[ ]:


#function which calculates dlum using single Riemann sum and rounded z values (only works if z values ARE rounded)
#rounded numbers must match the n_int significant figres 
#MOST BASIC,INNACURATE AND SLOWEST APPROACH
def lum_1(gen_z, omm, oml, omk, c_0, h0, n_sn, dlum):
    #initialising constants
    z_max = max(gen_z)
    n_it = int(z_max/0.001)
    width = z_max/(n_it)
    const = (abs(omk))**0.5
    tot = 0
    counter = 0
    if omk < 0:     #if statements select calculation based on curvature density
        for i in range(0,n_it+1):
            E_z = (1/((omm*(1+i*width)**3+omk*(1+i*width)**2+oml)**(0.5)))*width
            tot = tot + E_z
            try: #try and except stops the code from sampling beyond the maximum gen_z value
                while round(i*width, 4) == round(gen_z[counter],4): #condition so dlum is calculated when z-step reaches a generated z value
                    dlum[counter] = (c_0/(h0*const))*(1.0+gen_z[counter])*math.sin(const*(tot-E_z))
                    counter = counter + 1
            except IndexError:
                break
    elif omk > 0:
        for i in range(0,n_it+1):
            E_z = (1/((omm*(1+i*width)**3+omk*(1+i*width)**2+oml)**(0.5)))*width
            tot = tot + E_z
            try:
                while round(i*width,4) == round(gen_z[counter],4):
                    dlum[counter] = (c_0/(h0*const))*(1.0+gen_z[counter])*math.sinh(const*(tot-E_z))
                    counter = counter + 1 
            except IndexError:
                break

    else:
        for i in range(0,n_it+1):
            E_z = (1/((omm*(1+i*width)**3+omk*(1+i*width)**2+oml)**(0.5)))*width
            tot = tot + E_z
            try:
                while round(i*width,4) == round(gen_z[counter],4):
                    dlum[counter] = (c_0/h0)*(1.0+gen_z[counter])*(tot-E_z)
                    counter += 1
            except IndexError:
                break    

    return dlum


# In[ ]:


#calculates dlum for a given z value (requires script to loop over all z values with this function inside)
#SLOW BUT ACCURATE
#defining function to be integrated
def integrand(z, omm, oml,omk):
    return (omm*(1+z)**3+omk*(1+z)**2+oml)**(-0.5)

#defining distance luminostiy function using inbuilt integration function
def lum_2(z, omm, oml, omk, c_0, h0):
    #integrating
    cd = quad(integrand, 0, z, args=(omm,oml,omk))
    const = (abs(omk))**0.5 #defining constant used in dlum calculation
    #calculating dlum for respecive omk values
    if omk < 0:
        dlum = (c_0/(h0*const))*(1.0+z)*math.sin(const*cd[0])
        
    elif omk > 0:
        dlum = (c_0/(h0*const))*(1.0+z)*math.sinh(const*cd[0])
        
    else:
        dlum = (c_0/h0)*(1.0+z)*cd[0]

    return dlum


# In[ ]:


#function approximates a dlum funtion as multiple line segments for given density parameters
#function takes z array and for each z value in array, the corresponding dlum is taken the distance modulus is calculated
#FASTEST APPROACH

#defining integration function
def integrand(z, omm, oml, omk):
        return (omm*(1+z)**3+omk*(1+z)**2+oml)**(-0.5)

def lum_3(gen_z, omm, oml, omk, c_0, h0, n_sn):    
    #initialising arrays and constants
    gen_dlum = np.zeros(n_sn)
    z_max = max(gen_z) #extracting maximum z value (define outside function since only needs to be calculated once?)
    n_int = 50 #specifying number of line segments
    z_arr = np.zeros(n_int)
    grad = np.zeros(n_int-1)
    c = np.zeros(n_int-1)
    dlum_int = np.zeros(n_int)
    counter = 0
    
    for i in range(n_int): #setting points at which line segments start/finish
        z_arr[i] = i*(z_max/n_int) 
    const = (abs(omk))**0.5
    #calculating dlum for evenly spaced z values so line segments can be generated
    if omk < 0:
        for i in range(1,n_int):
            cd = quad(integrand, 0, z_arr[i], args=(omm,oml,omk))
            dlum_int[i] = (c_0/(h0*const))*(1.0+z_arr[i])*math.sin(const*cd[0])
    elif omk > 0:
        for i in range(1,n_int):
            cd = quad(integrand, 0, z_arr[i], args=(omm,oml,omk))
            dlum_int[i] = (c_0/(h0*const))*(1.0+z_arr[i])*math.sinh(const*cd[0])
    else:
        for i in range(1,n_int):
            cd = quad(integrand, 0, z_arr[i], args=(omm,oml,omk))
            dlum_int[i] = (c_0/h0)*(1.0+z_arr[i])*cd[0]
            
    #generating line segments to approximate dlum(z) function
    for i in range(1,n_int):
        grad[i-1] = (dlum_int[i]-dlum_int[i-1])/(z_arr[i]-z_arr[i-1])
        c [i-1] = dlum_int[i]-grad[i-1]*z_arr[i]
    
    for i in range(1,n_int):#for loop means the ith line segment will be used for the until z value passes valid range
        try: #try and except stops the code from sampling beyond the maximum gen_z value
            while gen_z[counter] <= (i+1)*z_max/n_int:
                gen_dlum[counter] = grad[i-1]*gen_z[counter]+c[i-1]
                counter += 1
        except IndexError:
                break
    return gen_dlum
    


# In[ ]:


#function which uses inbuild integration but rather than integrating from for each new z value, continues from previous z value
#WORKS WELL IF POINTS ARE ROUNDED SINCE SINGLE CALCULATION IS VALID FOR MUTLIPLE gen_z
def lum_4(gen_z, omm, oml, omk, c_0, h0, n_sn):
    dlum = np.zeros(n_sn)
    #integrating for first z point (needs to be done seperately to avoid indexing errors)
    cd,err = quad(integrand, 0, gen_z[0], args=(omm,oml,omk))
    
    const = (abs(omk))**0.5 #defining constant for dlum calculation
    #calculating dlum for respecive omk values
    if omk < 0:
        dlum[0] = (c_0/(h0*const))*(1.0+gen_z[0])*math.sin(const*cd)        
    elif omk > 0:
        dlum[0] = (c_0/(h0*const))*(1.0+gen_z[0])*math.sinh(const*cd)        
    else:
        dlum[0] = (c_0/h0)*(1.0+gen_z[0])*cd
    
    #loop over rest of z values (different calculation for varying omk values)
    if omk < 0:
        for i in range(1,n_sn): #looping through all z values
            if gen_z[i] == gen_z[i-1]: #if statment campares new point to last, if equal assign same dlum value (saves coputational cost)
                dlum[i] = dlum[i-1]

            else:
                temp = quad(integrand, gen_z[i-1], gen_z[i], args=(omm,oml,omk)) #integrates area between current and next point
                cd = cd + temp[0] #cumulative uses previous calculations for next
                dlum[i] = (c_0/(h0*const))*(1.0+gen_z[i])*math.sin(const*cd)
                
    elif omk > 0:
        for i in range(1,n_sn):
            if gen_z[i] == gen_z[i-1]:
                dlum[i] = dlum[i-1]

            else:
                temp = quad(integrand, gen_z[i-1], gen_z[i], args=(omm,oml,omk))
                cd = cd + temp[0]
                dlum[i] = (c_0/(h0*const))*(1.0+gen_z[i])*math.sinh(const*cd)

        
    else:
        for i in range(1,n_sn):
            if gen_z[i] == gen_z[i-1]:
                dlum[i] = dlum[i-1]

            else:
                temp = quad(integrand, gen_z[i-1], gen_z[i], args=(omm,oml,omk))
                cd = cd + temp[0]
                dlum[i] = (c_0/h0)*(1.0+gen_z[i])*cd


    return dlum
    


# In[ ]:


def integrand(z, omm, oml,omk):
    return (omm*(1+z)**3+omk*(1+z)**2+oml)**(-0.5)

#defining distance luminostiy function using inbuilt integration function
def D_c(z, omm, oml, omk, c_0, h0):
    #integrating
    cd = quad(integrand, 0, z, args=(omm,oml,omk))
    const = (abs(omk))**0.5 #defining constant used in dlum calculation
    #calculating dlum for respecive omk values
    if omk < 0:
        dc = (c_0/(h0*const))*math.sin(const*cd[0])
        
    elif omk > 0:
        dc = (c_0/(h0*const))*math.sinh(const*cd[0])
        
    else:
        dc = (c_0/h0)*cd[0]

    return dc


# In[ ]:


#specifying model
omm = 0.3     #matter density
oml = 0.7      #dark energy density
omk = 0.00      #curvature density
h0 = 70.0         #Hubble counstant
c_0 = 3*(10**5)   #speed of light kms-1
#generating uniform comoving z using rejection sampling
z_min = 1
z_max = 8
h0 = 70.0         #Hubble counstant
c_0 = 3*(10**5)   #speed of light kms-1
nit = 100
n_sn = 100
z_arr = np.zeros(nit)
dc_arr = np.zeros(nit)
dc_grad = np.zeros(nit)
dc_c = np.zeros(nit)
gen_z = np.zeros(n_sn)

for i in range(nit):
    z_arr[i] = z_min + (z_max-z_min)*i/nit
    dc_arr[i] = D_c(z_arr[i],omm,oml,omk,c_0,h0)
for i in range(1,nit):
    dc_grad[i-1] = (dc_arr[i]-dc_arr[i-1])/(z_arr[i]-z_arr[i-1])
    dc_c[i-1] = dc_arr[i]-dc_grad[i-1]*z_arr[i]

dc_min = D_c(z_min,omm,oml,omk,c_0,h0)
dc_max = D_c(z_max,omm,oml,omk,c_0,h0)

counter = 0
#bug sometimes requires this section to run multiple times to fill entire array!!
while counter <= n_sn:
    x = random.uniform(-dc_max,dc_max)
    y = random.uniform(-dc_max,dc_max)
    z = random.uniform(-dc_max,dc_max)
    dc = (x**2+y**2+z**2)**0.5

    if dc < dc_max:
        if dc > dc_min:
            q=0
            try:
                while dc>dc_arr[q]:#determining which line generated u value falls on
                    q = q+1
                    
                
                gen_z[counter] = ((dc-dc_c[q-1])/dc_grad[q-1])
                #print(gen_z[counter])
                counter = counter + 1
            except IndexError:
                break


# In[ ]:


print(gen_z)


# In[ ]:


gen_z = sorted(gen_z) #sorting z values necessary from error sampling and some of the dlum methods


# In[ ]:


#cdf of generated points
num_bins = n_sn
counts, bin_edges = np.histogram(gen_z, bins=num_bins)
cdf = np.cumsum(counts)
plt.step(bin_edges[1:],cdf/(num_bins))
plt.title("CDF of uniform co-moving z", fontsize=18)
plt.xlabel('Redshift (z)', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.savefig('cdfsnls.png', bbox_inches='tight')
plt.show()


# In[ ]:


#approximating error graph from delesning paper Shapiro et. al (select lensed or delensed)
#delensed
grad_err = (0.026-0.01)/(3-0.5)
c_err = 0.01-grad_err*0.5
#normal errors
#grad_err = (0.052-0.0465)/(3-2.5)
#c_err = 0.052-grad_err*3


gen_err_mu = np.zeros(n_sn)
gen_var_mu = np.zeros(n_sn)
pred_dlum = np.zeros(n_sn)
pred_mu = np.zeros(n_sn)

#caluclating distance modulus for genrated data using specified model
pred_dlum = lum_3(gen_z, omm, oml, omk, c_0, h0, n_sn)
#generating errors from real data best fit
for i in range(n_sn):
    #pred_dlum[i] = lum_2(gen_z[i], omm, oml, omk, c_0, h0)
    pred_mu[i] = 5*np.log10(pred_dlum[i])+25
    err_dlum = pred_dlum[i]*(gen_z[i]*grad_err+c_err)
    gen_err_mu = (5/np.log(10))*(err_dlum/pred_dlum[i])
    gen_var_mu[i] = gen_err_mu**2


# In[ ]:


#intitalising mcmc parameters
#BUG FOR LUM_3 - REQUIRES KERNEL TO RUN TWICE FOR CORRECT OUTPUT 
nmcmc = 20000
abc = np.zeros((nmcmc, 4)) #inititallising array with 3xnmcmc for omm, oml, omk and chisq
#initialising values   
abc[0,0] = omm  #mass density value
abc[0,1] = oml   #dark energy density value
abc[0,2] = 1-(abc[0,0]+abc[0,1])  #curvture density value (omm + oml + omk = 1)
sig_a = 0.005     #proposal density along a axis
sig_b = 0.005     #proposal density along b axis

accept = 0        #counter to compute acceptance ratio
chisq = 0         #initial chi squared value
chisq_trial = 0

start = time.time()
abc[0,3] = 0
gen_dlum = lum_3(gen_z, omm, oml, omk, c_0, h0, n_sn)
for i in range(0,(n_sn)):     
    gen_mu = 5*np.log10(gen_dlum[i])+25
    err_dlum = gen_dlum[i]*(gen_z[i]*grad_err+c_err)
    gen_err_mu = (5/np.log(10))*(err_dlum/gen_dlum[i])
    gen_var_mu = gen_err_mu**2
    arg = ((gen_mu-pred_mu[i])**2)/gen_var_mu  
    abc[0,3] = abc[0,3]+ (chisq+arg)

#starting mcmc chain
for i in range(1,nmcmc):
    omk = abc[i-1,2] + random.uniform(-1,1)*sig_a #setting new trial points using randomly generated number
    oml = abc[i-1,1] + random.uniform(-1,1)*sig_b #setting new trial points using randomly generated number
    omm = 1-(omk+oml) #condition that densitites must add up to one
    chisq = 0         
    chisq_trial = 0
    gen_dlum = lum_3(gen_z, omm, oml, omk, c_0, h0, n_sn)
    for isn in range(0,n_sn):
        gen_mu = 5*np.log10(gen_dlum[isn])+25
        err_dlum = gen_dlum[isn]*(gen_z[isn]*grad_err+c_err)
        gen_err_mu = (5/np.log(10))*(err_dlum/gen_dlum[isn])
        gen_var_mu = gen_err_mu**2
        arg = ((gen_mu-pred_mu[isn])**2)/gen_var_mu  
        abc[0,3] = abc[0,3]+ (chisq+arg)
        chisq_trial = chisq_trial+arg
    log_Lratio = 0.5*(abc[i-1,3]-chisq_trial) #calculating logratio value
    if log_Lratio >= 0: #accepting value if uphill
        abc[i,0] = omm
        abc[i,1] = oml
        abc[i,2] = omk
        abc[i,3] = chisq_trial
        accept = accept + 1
        print(i)

    else:
        ratio = np.exp(log_Lratio)
        test_uniform = random.uniform(0,1)   #generate uniform random number in (0,1)        
        if test_uniform <= ratio:  #in this case again accept trial point as new point
            abc[i,0] = omm
            abc[i,1] = oml
            abc[i,2] = omk
            abc[i,3] = chisq_trial
            accept = accept + 1
        else:     #return to previous point if rejected
            abc[i,0] = abc[i-1,0] 
            abc[i,1] = abc[i-1,1]
            abc[i,2] = abc[i-1,2]
            abc[i,3] = abc[i-1,3]

end = time.time()#function for timing process
print(end-start)#function for timing process
accept_ratio = accept/nmcmc
print(accept_ratio)


# In[ ]:


sortabc = np.array(sorted(abc, key=lambda x : x[3])) # sorting list based on chisquared values
sig1 = round(nmcmc*0.6827)
sig2 = round(nmcmc*0.9545)
sig3 = round(nmcmc*0.9973)


# In[ ]:


reg1_omk = sortabc[:sig1,2]
reg1_oml = sortabc[:sig1,1]
reg2_omk = sortabc[sig1:sig2,2]
reg2_oml = sortabc[sig1:sig2,1]
reg3_omk = sortabc[sig2:sig3,2]
reg3_oml = sortabc[sig2:sig3,1]
reg4_omk = sortabc[sig3:,2]
reg4_oml = sortabc[sig3:,1]


# In[ ]:


#RENAME PLOT BEFORE SAVING!!!!!!!!!!!
#scatter plot showing contours
plt.scatter(reg1_omk,reg1_oml,0.1, color='k')
plt.scatter(reg2_omk,reg2_oml,0.1)
plt.scatter(reg3_omk,reg3_oml,0.1)
plt.scatter(reg4_omk,reg4_oml,0.1)
plt.title("%i BBHM" % n_sn, fontsize=18)
plt.xlabel('$\Omega_k$', fontsize=12)
plt.ylabel('$\Omega_\Lambda$', fontsize=12)
plt.savefig('%iBBHM.png'% n_sn, bbox_inches='tight')
plt.show()


# In[ ]:


mean = np.mean(abc[:,2])
var = np.var(abc[:,2])
sd = var**0.5
print(sd)

