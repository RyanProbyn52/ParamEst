
# coding: utf-8

# In[1]:


#Importing libraries
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
import time


# In[2]:


#assign values here
incr = 0.0001
z = 1         #redshift
omm = 0.3        #mass density
oml = 0.7        #vacuum density
omk = 0.0        #curvature parameter
c = 3*10**5    #speed of light (Km/s)
h0 = 67      #hubble constant (Km/s/Mpc)
ncount = round((z/incr))  #number of iterations for integral approximation


# In[20]:


#luminosity distance calculation for flat universe
if omk == 0:
    #resetting comoving distance counter
    cd = 0
    #for loop to approximate numerical integration from 0 to z
    for count in range(1,ncount):
        zcount = (count)*incr
        Ez = (omm*(1+zcount)**3+oml)**0.5
        cd = cd + incr/Ez

    dlum = (c/h0)*(1+z)*cd
    
    if dlum < 0:
        dlum = 0

#luminosity distance calculation for open universe        
if omk > 0:
    const = omk**0.5
    #resetting comoving distance counter
    cd = 0
    #for loop to approximate numerical integration from 0 to z
    for count in range(1,ncount):
        zcount = count*incr
        Ez = (omm*(1+zcount)**3+oml+omk*(1+zcount)**2)**0.5
        cd = cd + incr/Ez

    dlum = (c/h0)*(1+z)*math.asinh(const*cd)/const
    
    if dlum < 0:
        dlum = 0

#luminosity distance calculation for closed universe          
if omk < 0:
    const = abs(omk)**0.5
    #resetting comoving distance counter
    cd = 0
    #for loop to approximate numerical integration from 0 to z
    for count in range(1,ncount):
        zcount = count*incr
        Ez = (omm*(1+zcount)**3+oml+omk*(1+zcount)**2)
        
        #filter negative solutions
        if Ez <= 0:
            dlum = 0
        else: 
            #take sqrt here in case solution was negative
            Ez = Ez**0.5
            cd = cd + incr/Ez
            dlum = (c/h0)*(1+z)*math.asin(const*cd)/const   
    
print("Luminosity distance is: %0.2f" % dlum,"Mpc")        
      

