
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
from IPython.display import clear_output


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
def integrand(z, omm, oml):
    return (omm*(1+z)**3+omk*(1+z)**2+oml)**(-0.5)

#defining distance luminostiy function using inbuilt integration function
def lum_2(z, omm, oml, omk, c_0, h0):
    #integrating
    cd = quad(integrand, 0, z, args=(omm,oml))
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
    gen_mu= np.zeros(n_sn)
    z_max = max(gen_z) #extracting maximum z value (define outside function since only needs to be calculated once?)
    n_int = 100 #specifying number of line segments
    z_arr = np.zeros(n_int)
    grad = np.zeros(n_int-1)
    c = np.zeros(n_int-1)
    mu = np.zeros(n_int)
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
        try: #try and except, stops the code from sampling beyond the maximum gen_z value
            while gen_z[counter] <= (i+1)*z_max/n_int:
                gen_dlum = grad[i-1]*gen_z[counter]+c[i-1]
                gen_mu[counter] = 5*np.log10(gen_dlum)+25
                counter += 1
        except IndexError:
                break
    return gen_mu
    


# In[ ]:


#function which uses inbuild integration but rather than integrating from for each new z value, continues from previous z value
#WORKS WELL IF POINTS ARE ROUNDED SINCE SINGLE CALCULATION IS VALID FOR MUTLIPLE gen_z
def lum_4(gen_z, omm, oml, omk, c_0, h0, n_sn, dlum):
    
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


#plotting redshift histogram of SNLS+others data
z = pd.read_csv('/home/2064198p/SNzdat.csv', header=None, usecols=[0], sep=',').values.flatten()
plt.hist(z,bins=12)   
plt.show()


# In[ ]:


#generating cdf of SNLS
num_bins = len(z)
counts, bin_edges = np.histogram(z, bins=num_bins)
cdf = np.cumsum(counts)
plt.step(bin_edges[1:],cdf/(num_bins))
plt.title("CDF of real data", fontsize=18)
plt.xlabel('Redshift (z)', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.savefig('cdfreal.png', bbox_inches='tight')
plt.show()


# In[ ]:


#creating array of coordinates of 'points' from cdf
n_sn = 500000 #specifying number of supernovae to be generated
x = np.zeros(num_bins)#i.e a point at each bin edge
y = np.zeros(num_bins)
gen_z = np.zeros(n_sn)
for i in range(0,num_bins):
    x[i] = bin_edges[i]
    y[i] = cdf[i]/num_bins 
grad = np.zeros(num_bins-1) #for n points there will be n-1 lines connecting
c = np.zeros(num_bins-1)

#loop to generate line segments between each point
for i in range(1,num_bins):
    grad[i-1] = (y[i]-y[i-1])/(x[i]-x[i-1])
    c [i-1] = y[i]-grad[i-1]*x[i]

for i in range(0, n_sn):
    u = random.uniform(y[0], 1)#generating random u between 0 and 1
    q = 0
    while u>y[q]:#determining which line generated u value falls on
        q = q+1
    #gen_z[i] = math.floor(((u-c[q-1])/grad[q-1])*1000)/1000#generating z value based on line segment (rounded to 3 decimals)
    gen_z[i] = ((u-c[q-1])/grad[q-1]) #same, but no rounding


# In[ ]:


#generating cdf for generated z values (for comparison)
num_bins = 1000
counts, bin_edges = np.histogram(gen_z, bins=num_bins)
cdf = np.cumsum(counts)
plt.step(bin_edges[1:],cdf/(n_sn))
plt.title("CDF of mock data", fontsize=18)
plt.xlabel('Redshift (z)', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.savefig('cdfmock.png', bbox_inches='tight')
plt.show()


# In[ ]:


#histogram for generated data to compare to SLS
plt.hist(gen_z,bins=12)   
plt.show()


# In[ ]:


#plotting errors from SLS with best fit line
z_snls = pd.read_csv('/home/2064198p/snls.csv', header=None, usecols=[0], sep=',').values.flatten()
err_mu = pd.read_csv('/home/2064198p/snls.csv', header=None, usecols=[8], sep=',').values.flatten()
p = np.poly1d(np.polyfit(z_snls, err_mu, 4)) #2nd order polynomial fit to data (maybe improve)
t = np.linspace(0, 1, 200)
plt.plot(z_snls,err_mu,'o',t, p(t), '-')
plt.show()


# In[ ]:


#sorting data from lowest to highest z value (important for some of dlum functions)
gen_z = sorted(gen_z)


# In[ ]:


#specifying model
omm = 0.3      #matter density
oml = 0.7      #dark energy density
omk = 0.00      #curvature density
h0 = 70.0         #Hubble counstant
c_0 = 3*(10**5)   #speed of light kms-1

#caluclating distance modulus for generated data using specified model
pred_mu = lum_3(gen_z, omm, oml, omk, c_0, h0, n_sn)
  
#generating errors from real data best fit
gen_err_mu = np.zeros(n_sn)
gen_var_mu = np.zeros(n_sn)
for i in range(0,n_sn): 
    gen_err_mu[i] = p(gen_z[i])
    gen_var_mu[i] = gen_err_mu[i]**2


# In[ ]:


#intitalising mcmc parameters
nmcmc = 10000
abc1 = np.zeros((nmcmc, 4)) #inititallising array with 3xnmcmc for omm, oml, omk and chisq
#initialising values   
abc1[0,0] = omm  #mass density value
abc1[0,1] = omk   #dark energy density value
abc1[0,2] = 1-(abc1[0,0]+abc1[0,1])  #curvture density value (omm + oml + omk = 1)
sig_a = 0.001      #proposal density along a axis
sig_b = 0.001      #proposal density along b axis

accept = 0        #counter to compute acceptance ratio
chisq = 0         #initial chi squared value
chisq_trial = 0
sig_int = 0.14
var_int = sig_int**2
abc1[0,3] = 0
gen_mu = lum_3(gen_z, omm, oml, omk, c_0, h0, n_sn)
for i in range(0,(n_sn)):    
    var_tot = var_int+gen_var_mu[i]
    arg = ((gen_mu[i]-pred_mu[i])**2)/var_tot  
    abc1[0,3] = abc1[0,3]+ (chisq+arg)

#starting mcmc chain
for i in range(1,nmcmc):
    #print(i)
    omm = abc1[i-1,0] + random.uniform(-1,1)*sig_a #setting new trial points using randomly generated number
    oml = abc1[i-1,1] + random.uniform(-1,1)*sig_b #setting new trial points using randomly generated number
    omk = 1-(omm+oml) #condition that densitites must add up to one
    chisq = 0         
    chisq_trial = 0
    gen_mu = lum_3(gen_z, omm, oml, omk, c_0, h0, n_sn)
    for isn in range(0,n_sn):
        var_tot = var_int+gen_var_mu[isn]
        arg = ((gen_mu[isn]-pred_mu[isn])**2)/var_tot  
        chisq_trial = chisq_trial+arg
    log_Lratio = 0.5*(abc1[i-1,3]-chisq_trial) #calculating logratio value
    if log_Lratio >= 0: #accepting value if uphill
        abc1[i,0] = omm
        abc1[i,1] = oml
        abc1[i,2] = omk
        abc1[i,3] = chisq_trial
        accept = accept + 1
    else:
        ratio = np.exp(log_Lratio)
        test_uniform = random.uniform(0,1)   #generate uniform random number in (0,1)        
        if test_uniform <= ratio:  #in this case again accept trial point as new point
            abc1[i,0] = omm
            abc1[i,1] = oml
            abc1[i,2] = omk
            abc1[i,3] = chisq_trial
            accept = accept + 1
        else:     #return to previous point if rejected
            abc1[i,0] = abc1[i-1,0] 
            abc1[i,1] = abc1[i-1,1]
            abc1[i,2] = abc1[i-1,2]
            abc1[i,3] = abc1[i-1,3]

accept_ratio = accept/nmcmc
print(accept_ratio)


# In[ ]:


temp = pd.DataFrame(data=abc1.astype(float)) #saving data to file
temp.to_csv('%iSN_1z.csv'% n_sn, sep=' ', header=False, float_format='%.15f', index=False)


# In[ ]:


sortabc = np.array(sorted(abc1, key=lambda x : x[3])) # sorting list based on chisquared values
sig1 = round(nmcmc*0.6827)
sig2 = round(nmcmc*0.9545)
sig3 = round(nmcmc*0.9973)


# In[ ]:


#scatter plot showing contours
plt.scatter(sortabc[:sig1,2],sortabc[:sig1,1],0.1, color='k')
plt.scatter(sortabc[sig1:sig2,2],sortabc[sig1:sig2,1],0.1)
plt.scatter(sortabc[sig2:sig3,2],sortabc[sig2:sig3,1],0.1)
plt.scatter(sortabc[sig3:,2],sortabc[sig3:,1],0.1)
plt.title("%i Supernovae" % n_sn, fontsize=18)
plt.xlabel('$\Omega_k$', fontsize=12)
plt.ylabel('$\Omega_\Lambda$', fontsize=12)
plt.savefig('%iSN1z_rs.png' % n_sn, bbox_inches='tight')
plt.show()


# In[ ]:


mean = np.mean(abc1[:,2])
var = np.var(abc1[:,2])
sd = var**0.5
print(sd)

