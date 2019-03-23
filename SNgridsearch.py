
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
#requires later code to be altered since this outputs distance modulus array

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
        try: #try and except stops the code from sampling beyond the maximum gen_z value
            while gen_z[counter] <= (i+1)*z_max/n_int:
                gen_dlum = grad[i-1]*gen_z[counter]+c[i-1]
                gen_mu[counter] = 5*np.log10(gen_dlum)+25
                counter += 1
        except IndexError:
                break
    return gen_mu
    


# In[ ]:


#Initialising Supernovae data
z = pd.read_csv('/home/2064198p/snls.csv', header=None, usecols=[0], sep=',').values.flatten()
appm = pd.read_csv('/home/2064198p/snls.csv', header=None, usecols=[1], sep=',').values.flatten()
err_appm = pd.read_csv('/home/2064198p/snls.csv', header=None, usecols=[2], sep=',').values.flatten()
s = pd.read_csv('/home/2064198p/snls.csv', header=None, usecols=[3], sep=',').values.flatten()
err_s = pd.read_csv('/home/2064198p/snls.csv', header=None, usecols=[4], sep=',').values.flatten()
c = pd.read_csv('/home/2064198p/snls.csv', header=None, usecols=[5], sep=',').values.flatten()
err_c = pd.read_csv('/home/2064198p/snls.csv', header=None, usecols=[6], sep=',').values.flatten()
mu = pd.read_csv('/home/2064198p/snls.csv', header=None, usecols=[7], sep=',').values.flatten()
err_mu = pd.read_csv('/home/2064198p/snls.csv', header=None, usecols=[8], sep=',').values.flatten()
sigma_int=0.14
var_int=sigma_int**2


# In[ ]:


#initialising constants
omm = 0.3      #matter density
oml = 0.7      #dark energy density
omk = 0.00      #curvature density
h0 = 70.0         #Hubble counstant
c_0 = 3*(10**5)   #speed of light kms-1
grid = 50
chisqmin = 10**10
oml = np.linspace(0,2,grid)
omm = np.linspace(0,1,grid)
chi_sq = np.zeros((grid,grid))
delchisq = np.zeros((grid,grid))

#iterates through grid NaN solution normal and corresponds to a 'no big bang' solution
for i in range(grid):
    for j in range(grid):
        chisq = 0
        omk = 1-(oml[i]+omm[j])
        for k in range(n_sn):
            dlum = lum_2(z[k], omm[j], oml[i], omk, c_0, h0) #calculating luminosity distance for each object
            gen_mu = 5*np.log10(dlum)+25 #converting to distance modulus
            var_tot = var_int+err_mu[k]**2 #variance of kth object
            arg = ((gen_mu-mu[k])**2)/var_tot 
            chi_sq[i,j] = chi_sq[i,j] + arg #computing chi-square values
            
chisqmin = np.nanmin(chi_sq) #finding minimum value ignoring NaNs      
for i in range(grid):
    for j in range(grid):
        delchisq[i,j] = chi_sq[i,j] - chisqmin #important step to determine confidence regions       


# In[ ]:


cont = [2.3, 6.3, 11.8] #plotting regions of 1 sigma, 2 sigma, and 3 sigma confidence

plt.contour(omm, oml, delchisq,cont)
plt.title('Confidence plot of SNLS')
plt.ylabel('$\Omega_\Lambda$')
plt.xlabel('$\Omega_m$')
plt.show()

